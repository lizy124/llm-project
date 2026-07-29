# weight_transfer 核心流程

`weight_transfer` 的本质是：vLLM 已经加载了一个模型，外部 trainer 后续把新权重传进来，vLLM 不重启服务，而是在线更新当前 worker 持有的 model parameter。

理解这个功能按时间顺序看两件事：

1. vLLM 服务初始化时提前准备了什么。
2. 外部真正发起 weight transfer 时，每一步发生什么。

## 一、vLLM 启动时准备什么

启动时大致流程：

```text
vLLM serve / LLM 初始化
  ↓
解析 weight_transfer_config
  ↓
正常加载初始模型
  ↓
每个 worker 创建 WeightTransferEngine
  ↓
engine 持有当前 model 对象
  ↓
HTTP / Python API / engine RPC 准备好
  ↓
等待外部 trainer 后续调用 init/update
```

### 1. 启用 weight_transfer_config

如果没有传 `weight_transfer_config`，vLLM 就是普通推理服务，不会创建 weight transfer engine。

upstream vLLM 示例：

```bash
VLLM_SERVER_DEV_MODE=1 \
vllm serve /path/to/model \
  --weight-transfer-config '{"backend": "nccl"}'
```

vllm-ascend 中可以使用 Ascend 原生 backend：

```bash
--weight-transfer-config '{"backend": "hccl"}'
--weight-transfer-config '{"backend": "npu_ipc"}'
```

也兼容 upstream backend name：

```text
nccl -> HCCLWeightTransferEngine
ipc  -> NPUIPCWeightTransferEngine
```

### 2. 正常加载初始模型

weight transfer 不替代模型加载。vLLM 启动时仍然先从 checkpoint / model path 加载初始模型，后续更新的是这份已经在 worker 上的参数。

### 3. 创建 WeightTransferEngine

worker 初始化时检查：

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

关键点是 engine 会拿到当前 worker 的 `model`。后面收到新权重时，engine 才能找到对应 parameter 并写进去。

### 4. 暴露调用入口

HTTP serving 场景下需要开启：

```bash
VLLM_SERVER_DEV_MODE=1
```

这样外部可以调用：

```text
/init_weight_transfer_engine
/start_weight_update
/update_weights
/finish_weight_update
/pause
/resume
/get_world_size
```

服务启动时通常只是 engine 和 endpoint 准备好了。NCCL/HCCL 通信组一般还没建立，真正初始化通常发生在 `/init_weight_transfer_engine`。

## 二、一次 weight transfer 发生什么

典型在线更新流程：

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

### 1. pause

在线服务里，更新权重前通常先暂停 generation，避免同一个 request 的前半段使用旧权重、后半段使用新权重。

### 2. init_weight_transfer_engine

外部调用 `/init_weight_transfer_engine` 后，worker 执行：

```text
parse init_info
  ↓
weight_transfer_engine.init_transfer_engine(init_info)
```

不同 backend 的 init 行为不同：

```text
NCCL/HCCL:
  trainer 提供 master address / port / world size / rank 信息
  vLLM worker 加入通信组

IPC/NPU IPC:
  准备 IPC handle 解析、设备匹配和后续 copy 所需状态
```

### 3. start_weight_update

外部调用 `/start_weight_update`，worker 通知 engine 进入一轮权重更新。这个阶段可以准备接收 buffer、清理上一轮状态或初始化 layerwise reload。

### 4. update_weights

这是数据真正进来的阶段，可能调用一次或多次。

NCCL / HCCL：

```text
trainer 当前参数 tensor
  ↓ broadcast
vLLM worker 接收 tensor / packed buffer
  ↓
写入本地 model parameter
```

HTTP/RPC 主要传 metadata，例如参数名、shape、dtype、packed buffer 配置；大 tensor 通过 NCCL/HCCL broadcast 传。

IPC / NPU IPC：

```text
trainer 侧 tensor storage
  ↓ 生成 IPC handle
metadata / handle 传给 vLLM
  ↓
vLLM worker 打开 handle
  ↓
copy 到本地 model parameter
```

Sparse NCCL：

```text
changed indices + new values
  ↓
vLLM worker 收到 sparse patch
  ↓
target_param.view(-1)[indices] = values
```

### 5. 写入当前模型参数

无论数据通过哪种 backend 传过来，最后都要落到当前 worker 的 model parameter 上：

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
received_tensor -> target_param.copy_(received_tensor)
```

packed tensor 会先从 packed buffer unpack 出多个参数，再分别 `copy_`。sparse patch 则只更新 flat indices 对应位置。

需要处理的关键问题包括：参数名、shape、dtype、当前 worker 持有完整参数还是 shard、TP/PP/EP 下更新哪部分，以及 quantized / draft / skip tensor 等特殊权重。

### 6. finish_weight_update 和 resume

外部调用 `/finish_weight_update`，worker 通知 engine 当前这一轮权重更新结束。这个阶段可以同步 device、清理 buffer、做后处理、重置状态。

如果前面 pause 了，最后调用 `/resume`，vLLM 继续用新权重处理后续请求。

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
pause generation，可选但在线服务常用
  ↓
init_weight_transfer_engine
  ↓
vLLM worker 初始化 NCCL/HCCL/IPC/NPU IPC backend
  ↓
start_weight_update
  ↓
update_weights，一次或多次
  ↓
backend 把数据送到 worker
  ↓
worker 写入当前 model parameter
  ↓
finish_weight_update
  ↓
resume generation
  ↓
vLLM 使用新权重继续推理
```

## 四、对应代码位置

upstream vLLM：

```text
vllm/config/weight_transfer.py
vllm/distributed/weight_transfer/base.py
vllm/distributed/weight_transfer/factory.py
vllm/distributed/weight_transfer/nccl_engine.py
vllm/distributed/weight_transfer/ipc_engine.py
vllm/distributed/weight_transfer/sparse_nccl_engine.py
vllm/distributed/weight_transfer/packed_tensor.py
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

vLLM 启动时只是把配置、模型、engine 和 API 准备好。真正 weight transfer 时，外部 trainer 才触发 backend 初始化、权重传输、参数写入、结束更新和恢复推理。
