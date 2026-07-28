# weight_transfer 核心流程：vLLM 先准备什么，后面真正同步时做什么

`weight_transfer` 的功能本质是：

> vLLM 已经加载了一个模型；外部 trainer 后续把新权重传进来；vLLM 不重启服务，而是把当前模型参数在线更新掉。

要理解这个功能，最好按时间顺序看两件事：

1. vLLM 服务初始化时，提前准备了什么。
2. 外部真正发起 weight transfer 时，每一步发生什么。

## 一、vLLM 服务初始化时的准备动作

vLLM 启动时不会立刻传权重，也通常不会立刻和 trainer 建通信组。它做的是把“将来可接收权重”的能力装好。

启动时大致流程：

```text
vLLM serve / LLM 初始化
    ↓
解析 weight_transfer_config
    ↓
正常加载模型到 worker
    ↓
每个 worker 创建 WeightTransferEngine
    ↓
engine 持有当前 model 对象
    ↓
HTTP / Python API / engine RPC 准备好
    ↓
等待外部 trainer 后续调用 init/update
```

### 1. 启动参数里必须启用 weight_transfer_config

如果没有传 `weight_transfer_config`，vLLM 就是普通推理服务，不会创建 weight transfer engine。

upstream vLLM 示例：

```bash
VLLM_SERVER_DEV_MODE=1 \
vllm serve /path/to/model \
  --weight-transfer-config '{"backend": "nccl"}'
```

也可以是：

```bash
--weight-transfer-config '{"backend": "ipc"}'
--weight-transfer-config '{"backend": "sparse_nccl"}'
```

vllm-ascend 中，用户也可能继续写 upstream 风格：

```bash
--weight-transfer-config '{"backend": "nccl"}'
--weight-transfer-config '{"backend": "ipc"}'
```

但 Ascend patch 会把它们映射到：

```text
nccl → HCCLWeightTransferEngine
ipc  → NPUIPCWeightTransferEngine
```

### 2. vLLM 正常加载模型

weight transfer 不是替代模型加载。

服务启动时，vLLM 仍然会先正常加载初始模型：

```text
checkpoint / model path
    ↓
vLLM worker model object
    ↓
GPU / NPU 上已有一份当前模型参数
```

后续 weight transfer 更新的是这份已经加载好的模型参数。

### 3. 每个 worker 创建对应的 WeightTransferEngine

worker 初始化时会检查：

```text
vllm_config.weight_transfer_config is not None
```

如果存在，就创建 engine：

```python
WeightTransferEngineFactory.create_engine(
    weight_transfer_config,
    vllm_config,
    device,
    model,
)
```

这一步非常关键，因为 engine 会拿到当前 worker 的：

```text
weight_transfer_config
vllm_config
device
model
```

其中最重要的是 `model`。

也就是说，启动阶段创建出来的 `WeightTransferEngine` 已经能访问当前 worker 上的模型对象。后面收到新权重时，它才有能力找到对应 parameter 并写进去。

### 4. factory 根据 backend 选择具体实现

upstream vLLM：

```text
backend="nccl"        → NCCLWeightTransferEngine
backend="ipc"         → IPCWeightTransferEngine
backend="sparse_nccl" → SparseNCCLWeightTransferEngine
```

vllm-ascend：

```text
backend="hccl"    → HCCLWeightTransferEngine
backend="npu_ipc" → NPUIPCWeightTransferEngine
```

同时 vllm-ascend 还 patch 了 upstream backend 名：

```text
backend="nccl" → HCCLWeightTransferEngine
backend="ipc"  → NPUIPCWeightTransferEngine
```

### 5. 如果是 HTTP serving，要暴露 dev-mode endpoint

HTTP server 场景下，需要开启：

```bash
VLLM_SERVER_DEV_MODE=1
```

这样外部才能调用这些 endpoint：

```text
/init_weight_transfer_engine
/start_weight_update
/update_weights
/finish_weight_update
/pause
/resume
/get_world_size
```

这些 endpoint 只是入口。服务刚启动时，它们处于等待调用状态。

### 6. 启动时通常还没真正建立通信组

这点容易误解。

vLLM 初始化时通常只是：

```text
模型加载好了
WeightTransferEngine 创建好了
engine 持有 model 了
API endpoint 准备好了
```

但 NCCL/HCCL 通信组通常还没建立。

真正建立通信组一般发生在外部 trainer 调：

```text
/init_weight_transfer_engine
```

之后。

比如 NCCL/HCCL backend 需要外部传入：

```text
master_address
master_port
rank_offset
world_size
```

vLLM worker 收到这些信息后，才加入 trainer 发起的通信组。

## 二、真正发生 weight_transfer 时的动作

服务已经启动并准备好之后，外部 trainer/controller 才开始真正同步权重。

一次典型流程：

```text
外部 trainer/controller
    ↓
1. pause vLLM generation
    ↓
2. init_weight_transfer_engine
    ↓
3. start_weight_update
    ↓
4. update_weights，一次或多次
    ↓
5. finish_weight_update
    ↓
6. resume vLLM generation
```

不是所有示例都一定显式调用完整六步，但核心含义是这个顺序。

### 1. pause：让 vLLM 进入适合更新的状态

在线服务里，更新权重前通常先暂停 generation：

```text
/pause
```

目的：避免 worker 正在 forward 的同时，另一个流程把 model parameter 改掉。

如果推理请求和权重写入交错，可能出现：

```text
同一个 request 的前半段用旧权重
后半段用新权重
```

这是需要避免的。

### 2. init_weight_transfer_engine：真正初始化通信 backend

外部调用：

```text
/init_weight_transfer_engine
```

vLLM worker 会执行：

```text
parse init_info
    ↓
weight_transfer_engine.init_transfer_engine(init_info)
```

不同 backend 的 init 行为不同。

NCCL/HCCL：

```text
trainer 提供 master address / port / world size / rank 信息
    ↓
vLLM worker 加入 NCCL/HCCL 通信组
```

IPC/NPU IPC：

```text
可能没有复杂通信组
主要准备 IPC handle 解析、设备匹配、后续 copy 所需状态
```

重点：

> vLLM 服务启动时只是创建 engine；真正通信初始化通常在这个阶段发生。

### 3. start_weight_update：开始一次权重更新

外部调用：

```text
/start_weight_update
```

worker 侧告诉 engine：

```text
这一轮权重更新要开始了
```

这个阶段可以做：

```text
准备接收 buffer
清理上一轮状态
进入 update 状态
等待后续 update_weights
```

### 4. update_weights：真正传权重

这是数据真正进来的阶段。

外部调用一次或多次：

```text
/update_weights
```

为什么可能多次？

因为模型参数很多，可能需要：

```text
按参数逐个传
按 layer 分批传
packed 后分 buffer 传
只传 sparse patch
```

#### NCCL / HCCL backend

流程：

```text
trainer 当前参数 tensor
    ↓ broadcast
vLLM worker 接收 tensor / packed buffer
    ↓
写入本地 model parameter
```

这里 HTTP/RPC 传的更多是 metadata，例如：

```text
参数名
shape
dtype
是否 packed
buffer 配置
```

真正大 tensor 通过 NCCL/HCCL broadcast 传。

#### IPC / NPU IPC backend

流程：

```text
trainer 侧 tensor storage
    ↓ 生成 IPC handle
metadata / handle 传给 vLLM
    ↓
vLLM worker 打开 handle
    ↓
copy 到本地 model parameter
```

这里传进 vLLM 的不是完整 checkpoint，而是 IPC handle 和必要 metadata。worker 通过 handle 拿到数据，再写入自己的模型参数。

#### sparse NCCL backend

流程：

```text
changed indices + new values
    ↓
vLLM worker 收到 sparse patch
    ↓
target_param.view(-1)[indices] = values
```

它不是完整权重同步，而是局部 patch。

### 5. worker 把收到的数据写进当前模型参数

这是最核心的动作。

无论数据怎么传过来，最后都要落到当前 worker 的 model parameter 上。

抽象流程：

```text
收到 update_info
    ↓
解析参数名、shape、dtype、patch 信息
    ↓
找到当前 model 里的目标 parameter
    ↓
copy_ / unpack / in-place patch
    ↓
当前模型参数被更新
```

典型写入方式：

```text
received_tensor → target_param.copy_(received_tensor)
```

packed tensor：

```text
packed buffer
    ↓ unpack
param A / param B / param C
    ↓
分别 copy_ 到对应 model parameter
```

sparse patch：

```text
indices + values
    ↓
target_param.view(-1)[indices] = values
```

这里要处理的关键问题包括：

```text
参数名是否匹配
shape 是否匹配
dtype 是否匹配
当前 worker 持有的是完整参数还是 shard
TP / PP / EP 下应该更新哪部分
quantized / draft / skip tensor 等特殊权重怎么处理
```

### 6. finish_weight_update：结束这一轮更新

外部调用：

```text
/finish_weight_update
```

worker 侧告诉 engine：

```text
这一轮权重都传完了
```

这个阶段可以做：

```text
同步 device
清理 buffer
做后处理
重置 update 状态
确认当前模型可继续推理
```

### 7. resume：恢复 generation

如果前面 pause 了，最后调用：

```text
/resume
```

vLLM 继续用新权重处理后续请求。

## 三、完整时间线

```text
阶段 A：vLLM 服务启动

vLLM 启动
  ↓
解析 --weight-transfer-config
  ↓
正常加载初始模型
  ↓
worker 创建 WeightTransferEngine
  ↓
engine 持有当前 model
  ↓
暴露 HTTP / Python API / engine RPC
  ↓
等待外部 trainer

阶段 B：外部 trainer 发起一次权重同步

trainer 更新出新权重
  ↓
外部调用 pause，可选但在线服务常用
  ↓
外部调用 init_weight_transfer_engine
  ↓
vLLM worker 初始化 NCCL/HCCL/IPC/NPU IPC backend
  ↓
外部调用 start_weight_update
  ↓
外部调用 update_weights，一次或多次
  ↓
backend 通过 NCCL/HCCL/IPC/NPU IPC/sparse patch 把数据送到 worker
  ↓
worker 把收到的数据写入当前 model parameter
  ↓
外部调用 finish_weight_update
  ↓
外部调用 resume
  ↓
vLLM 使用新权重继续推理
```

## 四、为什么不是重启服务

重启服务的路径是：

```text
trainer 保存 checkpoint
    ↓
停止 vLLM
    ↓
重新启动 vLLM
    ↓
重新加载 checkpoint
    ↓
重新初始化 worker / 通信 / 显存状态
```

weight transfer 的路径是：

```text
trainer 当前内存/GPU/NPU 权重
    ↓
NCCL / HCCL / IPC / NPU IPC / sparse patch
    ↓
vLLM worker 当前 model parameter
```

它绕过了 checkpoint 落盘和服务重启。

## 五、对应代码位置

upstream vLLM：

```text
vllm/config/weight_transfer.py
vllm/distributed/weight_transfer/base.py
vllm/distributed/weight_transfer/factory.py
vllm/distributed/weight_transfer/nccl_engine.py
vllm/distributed/weight_transfer/ipc_engine.py
vllm/distributed/weight_transfer/sparse_nccl_engine.py
vllm/distributed/weight_transfer/packed_tensor.py
vllm/v1/worker/gpu_worker.py
vllm/entrypoints/llm.py
vllm/v1/engine/async_llm.py
vllm/entrypoints/serve/dev/rlhf/api_router.py
```

vllm-ascend：

```text
vllm_ascend/distributed/weight_transfer/hccl_engine.py
vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py
vllm_ascend/distributed/weight_transfer/packed_tensor.py
vllm_ascend/patch/platform/patch_weight_transfer_engine.py
vllm_ascend/worker/worker.py
```

## 一句话总结

vLLM 初始化时只是提前准备好：

```text
配置已启用
模型已加载
WeightTransferEngine 已创建
engine 已持有 model
API 已暴露
```

真正 weight transfer 时才发生：

```text
初始化通信 backend
开始更新
传输权重数据
worker 写入当前 model parameter
结束更新
恢复推理
```
