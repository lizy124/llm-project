# vllm-ascend weight_transfer 重构设计

## 背景

vllm-ascend 的 `weight_transfer` 不是一套独立设计的系统，而是把 upstream vLLM 的 weight transfer 能力迁移到 Ascend NPU 场景：

```text
upstream nccl -> Ascend hccl
upstream ipc  -> Ascend npu_ipc
```

上层仍复用 vLLM 的 `WeightTransferConfig`、`WeightTransferEngineFactory`、HTTP API、LLM/AsyncLLM/worker 调用链。重构目标不是重新设计协议，而是降低后续对齐 upstream vLLM 的成本。

## 当前结构

upstream vLLM 的核心目录：

```text
vllm/distributed/weight_transfer/
  base.py
  factory.py
  nccl_engine.py
  ipc_engine.py
  packed_tensor.py
```

vllm-ascend 的核心目录：

```text
vllm_ascend/distributed/weight_transfer/
  __init__.py
  hccl_engine.py
  npu_ipc_engine.py
  packed_tensor.py

vllm_ascend/patch/platform/patch_weight_transfer_engine.py
```

当前实现可以理解为：copy upstream implementation -> port to Ascend backend -> patch factory 接入 -> 后续跟随 upstream 同步。

粗粒度相似度也支持这个判断：

| vllm-ascend 文件 | upstream 对应文件 | 相似度 | 结论 |
|---|---|---:|---|
| `hccl_engine.py` | `nccl_engine.py` | 0.810 | NCCL engine 骨架 port 到 HCCL/NPU |
| `npu_ipc_engine.py` | `ipc_engine.py` | 0.545 | IPC 流程保留，但 NPU handle/device 细节差异较多 |
| `packed_tensor.py` | `packed_tensor.py` | 0.302 | 职责对应 upstream packed tensor 工具 |
| `rlhf_http_hccl.py` | `rlhf_http_nccl.py` | 0.839 | 示例基本是替换 backend/device |
| `rlhf_http_npu_ipc.py` | `rlhf_http_ipc.py` | 0.729 | 示例结构沿用 upstream IPC 示例 |

## 主要问题

当前重复不是简单代码行数问题，而是边界不清。

### 1. Lifecycle 分散

upstream 已经演进到四阶段 API：

```text
init_weight_transfer_engine
start_weight_update
update_weights
finish_weight_update
```

Ascend 侧 HCCL 和 NPU IPC 分别实现 `start_weight_update` / `finish_weight_update`。如果 upstream 后续增加 pre/post hook、返回状态、错误恢复或 chunk session id，Ascend 需要逐个 backend 同步。

### 2. Trainer-side send 编排重复

HCCL 和 NPU IPC 都包含：

- 参数遍历。
- name / shape / dtype metadata 收集。
- packed / unpacked 分支。
- HTTP / Ray / callable send mode。
- post-send sync。

这些是公共编排，真正不同的是 HCCL broadcast 和 NPU IPC handle 构造。

### 3. Backend 注册入口分散

当前同时维护：

```text
hccl / npu_ipc              # Ascend 原生 backend name
nccl / ipc                  # upstream-compatible alias
```

注册分散在 `__init__.py` 和 `patch_weight_transfer_engine.py`，后续 upstream factory API 或 backend 类型变化时容易漏改。

### 4. 示例和测试复制 HTTP lifecycle

RLHF HTTP 示例、e2e 测试都重复 pause/init/start/update/finish/resume 流程。upstream endpoint schema 或调用顺序变化时，需要多处同步。

### 5. Backend 文件过厚

backend 文件同时承担 lifecycle、rank/world size、process group、packed/unpacked、trainer send、HTTP/Ray 分发、device/UUID/handle 等职责。reviewer 很难快速判断哪些是 upstream 同步逻辑，哪些是真正的 Ascend 差异。

## 重构边界

目标边界：

```text
upstream 公共协议和流程
  -> registry / lifecycle / trainer_send / HTTP helper / test helper

Ascend 通信差异
  -> hccl_engine / npu_ipc_engine / packed transport

示例和测试模板
  -> 示例 helper 和 e2e helper
```

不建议直接 fork upstream `base.py` / `factory.py`。vllm-ascend 继续复用 upstream 的 `WeightTransferEngine` 和 `WeightTransferEngineFactory`，只在 Ascend patch 和 helper 层做适配。

也不建议把 HCCL 和 NPU IPC 合成一个大 engine。HCCL 是 broadcast 通信组，NPU IPC 是 handle 共享和 rebuild，底层模型不同，过度抽象会产生大量 backend if/else。

## 目标结构

推荐结构：

```text
vllm_ascend/distributed/weight_transfer/
  __init__.py
  registry.py
  lifecycle.py
  trainer_send.py
  device_mapping.py
  hccl_engine.py
  npu_ipc_engine.py
  packed_tensor.py
```

职责：

- `registry.py`：唯一 backend 注册和 alias 入口。
- `lifecycle.py`：start/finish lifecycle policy。
- `trainer_send.py`：trainer-side send 编排、metadata/payload 公共逻辑。
- `device_mapping.py`：`ASCEND_RT_VISIBLE_DEVICES`、logical/physical NPU、host-ip UUID。
- `hccl_engine.py`：HCCL communicator、broadcast、process group 初始化。
- `npu_ipc_engine.py`：NPU IPC handle encode/decode、receive rebuild。
- `packed_tensor.py`：保留 packed 公共 helper 和现有 transport 入口；更大拆分可后续单独做。

示例和测试：

```text
examples/rl/weight_transfer_http_utils.py
tests/e2e/pull_request/weight_transfer_utils.py
```

## 落地顺序

推荐按风险从低到高推进：

1. 收敛 registry 和 alias。
2. 抽 HTTP 示例 helper 和 e2e 测试 helper。
3. 抽 lifecycle policy 和 device mapping。
4. 抽 trainer_send orchestration。
5. 最后再考虑 packed tensor 文件级拆分。

当前完整重构视图已经完成到第 4 步，并做了低风险 packed tensor helper 抽取。没有重写 HCCL / NPU IPC 的核心 transport loop。

## 验收标准

### 行为等价

重构前后必须保持：

- HCCL backend 可以完成一次完整 weight update。
- NPU IPC backend 可以完成一次完整 weight update。
- pause / resume 正常工作。
- init / start / update / finish 顺序正常。
- packed=False 和 packed=True 路径正常。
- HTTP send mode 正常。
- Ray 或 callable send mode 如果已有覆盖，保持正常。
- 错误输入仍返回预期错误，而不是静默成功。

推荐验证：

```text
pytest tests/ut/distributed/weight_transfer
pytest tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
pytest tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

无 Ascend 环境时，最低要求是纯 Python UT、import collection、registry / lifecycle / device_mapping / trainer_send mock 测试通过，NPU 相关 e2e 留给 CI。

### 接口兼容

必须继续支持：

```text
hccl    -> HCCLWeightTransferEngine
npu_ipc -> NPUIPCWeightTransferEngine
nccl    -> HCCLWeightTransferEngine
ipc     -> NPUIPCWeightTransferEngine
```

这部分应有 registry UT 覆盖，不应只依赖 e2e。

### 边界清晰

重构后 reviewer 应该能按职责定位：

```text
backend 注册和 alias -> registry.py
start / finish 生命周期差异 -> lifecycle.py
HTTP 示例调用模板 -> weight_transfer_http_utils.py
测试服务和 endpoint 模板 -> weight_transfer_utils.py
device / physical chip id / host key -> device_mapping.py
trainer send 编排 -> trainer_send.py
HCCL 通信细节 -> hccl_engine.py
NPU IPC handle 编解码 -> npu_ipc_engine.py
packed 公共和 transport 差异 -> packed_tensor.py 或后续 packed/
```

如果公共 lifecycle、HTTP endpoint 调用、trainer send payload 仍然在多个 backend 或示例中复制，说明问题没有真正解决。

### 回归风险

需要确认：

- `import vllm_ascend` 不会提前 import Ray 或初始化 NPU 通信资源。
- registry lazy-load 行为保持不变。
- 无 NPU 环境下，纯 Python 测试不会因为 import `torch_npu` / HCCL 失败而整体失败。
- 错误路径不会吞异常。
- 日志仍能定位 backend、rank、device、send mode。

## 最终判断

这次重构的价值在于集中 upstream API 同步点，而不是简单减少代码量。

如果后续 upstream 修改 `start_weight_update`、`finish_weight_update`、`update_weights` payload、HTTP endpoint schema 或 trainer send args，理想修改点应该集中在 `lifecycle.py`、`trainer_send.py`、`weight_transfer_http_utils.py` 和 e2e helper，而不是同时修改 HCCL backend、NPU IPC backend、两个示例和多个测试文件。
