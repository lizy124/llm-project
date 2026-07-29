# vllm-ascend weight_transfer 代码对比与重构方案

## 背景和目标

当前 vllm-ascend 的 weight_transfer 适配，本质目标不是重新设计一套权重传输系统，而是把 upstream vLLM 的 weight_transfer 能力迁移到 Ascend NPU 场景：

- upstream `nccl` backend 对应 Ascend `hccl` backend。
- upstream `ipc` backend 对应 Ascend `npu_ipc` backend。
- 上层仍复用 vLLM 的 `WeightTransferConfig`、`WeightTransferEngineFactory`、HTTP API、LLM/AsyncLLM/worker 调用链。

重构的根本目的：**后续对齐 upstream vLLM 时更轻。upstream 改 API 时，只需要改公共 lifecycle 或注册层，而不是在 HCCL、NPU IPC、示例、测试里到处同步一遍。**

本文基于本地代码对比：

- upstream vLLM：`D:/lzy/code/pd_pool_mtp/vllm`
- vllm-ascend：`D:/lzy/code/pd_pool_mtp/vllm-ascend`
- 当前 vllm-ascend HEAD：`864512208`

## 当前结构对比

### upstream vLLM

核心目录：`vllm/distributed/weight_transfer/`

```text
__init__.py
base.py
factory.py
ipc_engine.py
nccl_engine.py
packed_tensor.py
```

关键职责：

- `base.py`：定义 `WeightTransferEngine`、`WeightTransferInitInfo`、`WeightTransferUpdateInfo`、API request dataclass、parse/validate 基础逻辑。
- `factory.py`：定义 `WeightTransferEngineFactory`，集中 lazy register `nccl` / `ipc`。
- `nccl_engine.py`：NCCL dense/sparse 权重同步实现。
- `ipc_engine.py`：CUDA IPC 权重同步实现，包含 HTTP/Ray/send_mode、packed/unpacked 发送逻辑。
- `packed_tensor.py`：NCCL/IPC packed tensor 工具。

### vllm-ascend

核心目录：`vllm_ascend/distributed/weight_transfer/`

```text
__init__.py
hccl_engine.py
npu_ipc_engine.py
packed_tensor.py
```

额外 patch：

```text
vllm_ascend/patch/platform/patch_weight_transfer_engine.py
```

关键职责：

- `__init__.py`：注册 `hccl` / `npu_ipc` backend。
- `patch_weight_transfer_engine.py`：覆盖 upstream factory 中的 `nccl` / `ipc`，让 upstream backend name 在 Ascend 环境下落到 `HCCLWeightTransferEngine` / `NPUIPCWeightTransferEngine`。
- `hccl_engine.py`：按 NCCL engine 的结构适配 HCCL。
- `npu_ipc_engine.py`：按 IPC engine 的结构适配 NPU IPC。
- `packed_tensor.py`：Ascend 侧 packed broadcast / packed NPU IPC 工具。

## 代码相似度和复制痕迹

对当前文件做粗粒度文本相似度对比，结果如下：

| vllm-ascend 文件 | upstream 对应文件 | 相似度 | 结论 |
|---|---|---:|---|
| `vllm_ascend/distributed/weight_transfer/hccl_engine.py` | `vllm/distributed/weight_transfer/nccl_engine.py` | 0.810 | 高度相似，基本是 NCCL engine 骨架 port 到 HCCL/NPU |
| `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py` | `vllm/distributed/weight_transfer/ipc_engine.py` | 0.545 | 中等相似，保留 IPC 流程，但 NPU handle/device 细节改动较多 |
| `vllm_ascend/distributed/weight_transfer/packed_tensor.py` | `vllm/distributed/weight_transfer/packed_tensor.py` | 0.302 | 相似度较低，但职责仍对应 upstream packed tensor 工具 |
| `examples/rl/rlhf_http_hccl.py` | `examples/rl/rlhf_http_nccl.py` | 0.839 | 高度相似，示例基本是替换 backend/device |
| `examples/rl/rlhf_http_npu_ipc.py` | `examples/rl/rlhf_http_ipc.py` | 0.729 | 高度相似，示例结构沿用 upstream IPC 示例 |

这个结果说明：当前实现不是纯粹从零重写，也不是只在 upstream 原文件上打几个小 patch，而是典型的 **copy upstream implementation -> port to Ascend backend -> patch factory 接入 -> 后续跟随 upstream 同步**。

这种方式适合快速落地，但长期会带来同步成本。

## 方法级别对比

### HCCL vs NCCL

upstream `NCCLWeightTransferEngine` 方法：

```text
__init__
init_transfer_engine
receive_weights
receive_sparse_weights
shutdown
trainer_send_weights
trainer_send_sparse_weights
trainer_init
_stateless_init_process_group
```

Ascend `HCCLWeightTransferEngine` 方法：

```text
__init__
start_weight_update
finish_weight_update
init_transfer_engine
receive_weights
shutdown
trainer_send_weights
trainer_init
_stateless_init_process_group
```

共性明显：

- init info / update info dataclass 结构相近。
- rank/world size 计算流程相近。
- stateless process group 初始化流程相近。
- packed vs unpacked receive/send 分支相近。
- trainer-side send 权重流程相近。

主要差异：

- NCCL 换成 HCCL/PyHcclCommunicator。
- CUDA device/stream 换成 NPU device/stream。
- HCCL 当前没有 sparse 权重更新支持。
- Ascend HCCL 侧补了 `start_weight_update` / `finish_weight_update`，调用 layerwise reload 初始化和收尾。

判断：HCCL engine 当前承担了大量“生命周期模板 + backend 通信细节”混合职责，后续 upstream 改生命周期或 trainer_send 参数时，会被迫手动同步。

### NPU IPC vs CUDA IPC

upstream `IPCWeightTransferEngine` 方法：

```text
__init__
parse_update_info
init_transfer_engine
receive_weights
shutdown
trainer_send_weights
_is_rank_zero
_all_gather_and_merge_handles
_post_send_sync
_send_unpacked
_send_packed
_do_send
```

Ascend `NPUIPCWeightTransferEngine` 方法：

```text
__init__
parse_update_info
init_transfer_engine
start_weight_update
finish_weight_update
receive_weights
shutdown
trainer_send_weights
_is_rank_zero
_all_gather_and_merge_handles
_post_send_sync
_send_unpacked
_send_packed
_do_send
```

共性明显：

- `IPCTrainerSendWeightsArgs`、`IPCWeightTransferUpdateInfo` 直接从 upstream IPC engine 继承。
- `parse_update_info` 的 HTTP pickle/base64 处理流程类似 upstream。
- trainer send 被拆成 `_send_unpacked`、`_send_packed`、`_do_send`。
- Ray/HTTP/callable `send_mode` 分支沿用 upstream 设计。
- all-gather handles、rank0 send、post-send sync 的结构一致。

主要差异：

- CUDA IPC handle rebuild 换成 `torch_npu.multiprocessing.reductions.rebuild_npu_tensor`。
- 通过 `ASCEND_RT_VISIBLE_DEVICES` 和 `torch.accelerator.current_device_index()` 生成物理 NPU UUID。
- IPC handle 从 CUDA device 语义改成 `{host_ip}-{physical_chip_id}` 匹配。
- `torch.cuda.synchronize()` / CUDA stream 相关逻辑换成 `torch.npu.synchronize()` / NPU device index。

判断：NPU IPC 不是简单替换几处字符串，但它仍然复制了 upstream IPC 的流程骨架。适合把“send orchestration”和“NPU handle 编解码”拆开。

## 当前重复和维护风险

### 风险 1：生命周期 API 分散在多个 backend 中

upstream 已经从早期 `init/update` 演进到明确的四阶段：

```text
init_weight_transfer_engine
start_weight_update
update_weights
finish_weight_update
```

Ascend 当前在 HCCL/NPU IPC 中分别实现或 no-op：

- HCCL：`start_weight_update` / `finish_weight_update` 调 reload。
- NPU IPC：`start_weight_update` / `finish_weight_update` no-op。

如果 upstream 后续继续调整 lifecycle，例如增加 pre/post hook、返回状态、错误恢复、chunk session id，Ascend 侧需要逐个 backend 同步。

### 风险 2：trainer_send 编排逻辑重复

HCCL 和 NPU IPC 都有 trainer-side send API：

- 参数遍历。
- dtype/shape/name 收集。
- packed/unpacked 分支。
- 发送后 sync。
- HTTP/Ray/callable send_mode。

这些流程里只有底层通信和 handle 构造不同。当前写在 backend 文件中，导致 upstream 改 trainer args 或 send payload 结构时，多个 backend 都要改。

### 风险 3：factory 注册入口分散

当前有两套接入逻辑：

```text
vllm_ascend/distributed/weight_transfer/__init__.py
  register_engine() 注册 hccl/npu_ipc

vllm_ascend/patch/platform/patch_weight_transfer_engine.py
  覆盖 nccl/ipc 到 HCCL/NPU IPC
```

这会产生两个维护点：

- 正式 backend name：`hccl` / `npu_ipc`
- upstream-compatible alias：`nccl` / `ipc`

后续 upstream 如果开放更稳定的 out-of-tree backend 注册，或者 `WeightTransferConfig.backend` 类型再次变化，这里容易重复调整。

### 风险 4：示例与测试同步成本高

示例相似度很高：

- `rlhf_http_hccl.py` vs `rlhf_http_nccl.py`：0.839
- `rlhf_http_npu_ipc.py` vs `rlhf_http_ipc.py`：0.729

这说明大量 HTTP helper、pause/resume、start/update/finish、world size、request 封装都在复制。upstream 示例改 API 时，Ascend 示例需要手工同步。

测试也存在类似问题：e2e 启动服务、构造 config、调用 endpoint、检查权重更新这些流程应尽量复用 helper，而不是每个 backend 一份。

### 风险 5：backend 文件过厚，真正的 NPU 差异不突出

现在 backend 文件同时包含：

- dataclass 定义。
- lifecycle。
- rank/world size 计算。
- process group 初始化。
- packed/unpacked 传输。
- trainer send。
- HTTP/Ray 分发。
- NPU device/UUID/handle 细节。

这会让后续 review 很难快速判断：哪些是 upstream 同步逻辑，哪些是 Ascend 必须保留的差异。

## 重构必要性判断

有必要重构，理由如下：

1. **代码来源和结构强依赖 upstream**：HCCL/NPU IPC 不是完全独立设计，而是基于 upstream NCCL/IPC port。既然形态依赖 upstream，就应该把同步点集中起来。
2. **相似度高，重复不是偶然**：HCCL engine 和 HCCL 示例相似度超过 0.8，说明后续 upstream 变更会持续造成重复同步。
3. **当前职责边界不清**：backend 同时承载 lifecycle、通信、payload、示例流程，导致每次 API 变更影响面过大。
4. **未来扩展会放大重复**：如果后续要做 `sparse_hccl`、stateful trainer client、更多 RLHF async 示例，继续复制 backend 会进一步增加维护成本。
5. **重构收益明确**：把公共 lifecycle、注册、trainer send orchestration、示例 helper 抽出来，可以让 Ascend backend 只保留 NPU/HCCL/NPU IPC 差异。

不建议一次性大规模重写。这个模块涉及 trainer/inference 多进程、NPU IPC handle、HCCL group、HTTP/Ray、e2e，风险不低。应分阶段、可验证地重构。

## 重构目标

目标不是简单减少行数，而是建立以下边界：

```text
upstream API 变化
  -> 改公共 lifecycle / adapter / registration
  -> backend 少改或不改

Ascend 通信差异
  -> 留在 hccl / npu_ipc communicator 中
  -> 不污染公共流程

示例和测试流程
  -> helper 统一封装 HTTP lifecycle
  -> backend 示例只声明 backend config 和少量差异
```

最终希望形成：

- backend 文件更薄。
- patch 文件只做注册入口，不承载实现细节。
- 示例和测试只表达场景，不重复 HTTP/Ray 调用模板。
- upstream vLLM API 改动时，修改点集中。

## 推荐重构方案

### 阶段 1：收敛注册与 alias 逻辑

优先级：高。风险：低。

当前问题：`register_engine()` 和 `patch_weight_transfer_engine.py` 分散维护 backend name 和 alias。

建议新增或改造统一注册入口：

```text
vllm_ascend/distributed/weight_transfer/registry.py
```

职责：

```python
def register_ascend_weight_transfer_engines(
    include_upstream_aliases: bool = True,
    override_existing: bool = True,
) -> None:
    ...
```

注册关系：

```text
hccl -> HCCLWeightTransferEngine
npu_ipc -> NPUIPCWeightTransferEngine
nccl -> HCCLWeightTransferEngine      # Ascend compatibility alias
ipc -> NPUIPCWeightTransferEngine     # Ascend compatibility alias
```

`vllm_ascend/distributed/weight_transfer/__init__.py` 只暴露：

```python
from .registry import register_ascend_weight_transfer_engines
```

`patch_weight_transfer_engine.py` 只调用：

```python
register_ascend_weight_transfer_engines(include_upstream_aliases=True)
```

收益：

- 后续 upstream factory 行为变化时，只改 registry/patch 一处。
- `hccl/npu_ipc` 和 `nccl/ipc` alias 关系集中管理。
- 减少直接散落访问 `_registry` 的位置。

注意：upstream 当前 `register_engine()` 对重复注册会报错，而 alias 覆盖需要替换 `_registry`。因此 registry 里要显式区分 register 和 override，避免隐藏行为。

### 阶段 2：抽 HTTP lifecycle 示例 helper

优先级：高。风险：低。

当前问题：`rlhf_http_hccl.py`、`rlhf_http_npu_ipc.py` 和 upstream 示例高度相似，重复了 endpoint 封装。

建议新增：

```text
examples/rl/weight_transfer_http_utils.py
```

封装：

```python
generate_completions(url, prompt)
init_weight_transfer_engine(url, init_info)
start_weight_update(url)
update_weights(url, update_info)
finish_weight_update(url)
pause_generation(url)
resume_generation(url)
get_world_size(url)
run_weight_update_lifecycle(...)
```

示例文件保留：

- backend config。
- init/update info 构造。
- trainer send 调用。
- HCCL vs NPU IPC 的差异参数。

收益：

- upstream HTTP API 改 payload 或 endpoint 时，示例只改 helper。
- 示例更短，更容易看出 backend 差异。
- 可被 e2e 测试复用。

### 阶段 3：抽测试 helper

优先级：中高。风险：低到中。

建议新增：

```text
tests/e2e/pull_request/weight_transfer_utils.py
```

或按现有测试目录规范放置公共工具。

封装：

- 启动/等待 vLLM server。
- 构造 `--weight-transfer-config`。
- 调用 pause/start/update/finish/resume。
- 通用权重变更断言。
- HCCL/NPU IPC backend 参数模板。

收益：

- backend 测试只保留场景差异。
- API 变更时测试修改点集中。
- 后续添加 sparse_hccl 或 async lifecycle 测试成本更低。

### 阶段 4：抽 backend lifecycle mixin/base

优先级：中。风险：中。

当前问题：HCCL/NPU IPC backend 都直接实现 lifecycle，后续 upstream 生命周期变化会多处同步。

建议新增：

```text
vllm_ascend/distributed/weight_transfer/lifecycle.py
```

候选抽象：

```python
class AscendWeightTransferLifecycleMixin:
    def start_weight_update(self) -> None:
        ...

    def finish_weight_update(self) -> None:
        ...
```

或者更明确地拆成策略：

```python
class WeightUpdateLifecyclePolicy:
    def start(self, model): ...
    def finish(self, model, model_config): ...

class LayerwiseReloadLifecyclePolicy(WeightUpdateLifecyclePolicy): ...
class NoopLifecyclePolicy(WeightUpdateLifecyclePolicy): ...
```

HCCL 使用 `LayerwiseReloadLifecyclePolicy`，NPU IPC 使用 `NoopLifecyclePolicy`。

收益：

- upstream start/finish 语义变化时集中改 lifecycle policy。
- backend 文件减少 reload import 和生命周期细节。
- 明确表达 HCCL 和 NPU IPC 的 lifecycle 差异。

注意：这里不要急着替换 upstream `WeightTransferEngine`，而是在 Ascend backend 内部先抽轻量 policy，降低风险。

### 阶段 5：抽 trainer_send orchestration

优先级：中。风险：中到高。

当前问题：NPU IPC 的 `_send_unpacked`、`_send_packed`、`_do_send` 和 upstream IPC 结构非常像；HCCL 的 trainer send 也有 packed/unpacked 分支。这里是后续 upstream API drift 的高风险区。

建议分两层抽：

```text
vllm_ascend/distributed/weight_transfer/trainer_send.py
```

公共 orchestration：

```python
@dataclass
class WeightMetadata:
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

collect_weight_metadata(iterator, tensor_getter)
send_update_payload(send_mode, url, llm_handle, update_info)
rank_zero_only_send(...)
post_send_sync(...)
```

backend-specific hooks：

```python
class HandleProducer:
    def make_handle(self, tensor) -> Any: ...
    def current_device_key(self) -> str: ...
```

NPU IPC 保留：

- `npu_generate_uuid()`。
- `reduce_tensor()` / `rebuild_npu_tensor()`。
- physical chip id 映射。
- packed NPU IPC producer/consumer。

HCCL 保留：

- HCCL communicator broadcast。
- HCCL process group init。

收益：

- HTTP/Ray/callable send_mode 改动集中。
- metadata 收集和 payload 结构集中。
- NPU IPC backend 更突出 handle 编解码差异。

注意：这一阶段涉及行为风险，需要有 UT + e2e 覆盖后再做。

### 阶段 6：重整 packed_tensor 职责

优先级：中。风险：中到高。

当前 `packed_tensor.py` 同时包含：

- HCCL packed broadcast producer/consumer。
- NPU IPC packed producer/consumer。
- buffer/chunk 控制。
- tensor size metadata。

建议拆分为：

```text
vllm_ascend/distributed/weight_transfer/packed/
  __init__.py
  common.py
  hccl.py
  npu_ipc.py
```

职责：

- `common.py`：chunk planning、metadata、buffer size 校验。
- `hccl.py`：HCCL packed broadcast producer/consumer。
- `npu_ipc.py`：NPU IPC packed producer/consumer。

收益：

- packed 逻辑按 transport 拆开，避免一个文件同时处理两类 backend。
- 后续如果 upstream packed tensor 协议变化，可以先改 common。
- 后续 sparse/chunked lifecycle 更容易接入。

注意：这个阶段应该晚于 trainer_send 抽象，因为两者耦合较强。

## 建议后的目标结构

推荐目标结构：

```text
vllm_ascend/distributed/weight_transfer/
  __init__.py
  registry.py
  lifecycle.py
  trainer_send.py
  device_mapping.py
  hccl_engine.py
  npu_ipc_engine.py
  packed/
    __init__.py
    common.py
    hccl.py
    npu_ipc.py
```

其中：

- `registry.py`：唯一 backend 注册/alias 入口。
- `lifecycle.py`：start/finish policy。
- `trainer_send.py`：trainer-side send 编排、metadata/payload 公共逻辑。
- `device_mapping.py`：`ASCEND_RT_VISIBLE_DEVICES`、logical/physical NPU、host-ip UUID。
- `hccl_engine.py`：只保留 HCCL communicator、broadcast、process group 初始化。
- `npu_ipc_engine.py`：只保留 NPU IPC handle encode/decode、receive rebuild。
- `packed/`：按 transport 拆分 packed tensor 工具。

示例目标结构：

```text
examples/rl/weight_transfer_http_utils.py
examples/rl/rlhf_http_hccl.py
examples/rl/rlhf_http_npu_ipc.py
examples/rl/rlhf_async_new_apis.py
```

测试目标结构：

```text
tests/e2e/pull_request/weight_transfer_utils.py
tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py
tests/ut/distributed/weight_transfer/test_registry.py
tests/ut/distributed/weight_transfer/test_device_mapping.py
tests/ut/distributed/weight_transfer/test_trainer_send.py
```

## 不建议的重构方式

### 不建议直接 fork upstream `base.py` / `factory.py`

vllm-ascend 当前复用 upstream `WeightTransferEngine` 和 `WeightTransferEngineFactory` 是正确方向。直接复制一份 Ascend base/factory 会让接口更容易漂移，违背“后续对齐 upstream 更轻”的目标。

### 不建议一次性把 HCCL/NPU IPC 合成一个大通用 engine

HCCL 和 NPU IPC 的底层传输模型不同：

- HCCL 是 broadcast 通信组。
- NPU IPC 是 handle 共享和 rebuild。

过度抽象会让公共层充满 backend if/else，反而降低可维护性。应抽 orchestration 和 lifecycle，保留 transport-specific engine。

### 不建议先动 packed_tensor 大拆分

packed tensor 和实际传输行为强相关，改错容易影响性能和正确性。应先抽低风险的 registry、示例 helper、测试 helper，再动 trainer_send 和 packed。

## 分阶段落地计划

### 第一轮：低风险收敛

目标：减少注册、示例、测试的同步成本，不改变核心传输行为。

改动：

1. 新增 `registry.py`，统一注册 `hccl/npu_ipc` 和 `nccl/ipc` alias。
2. 简化 `patch_weight_transfer_engine.py`，只调用 registry。
3. 抽 `examples/rl/weight_transfer_http_utils.py`。
4. 改造 `rlhf_http_hccl.py`、`rlhf_http_npu_ipc.py` 使用 helper。
5. 增加 registry UT。

验证：

```text
pytest tests/ut/distributed/weight_transfer
pytest tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
pytest tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

如果本地没有 NPU 环境，至少跑 registry/helper 的纯 Python UT，并保留 e2e 给 CI。

### 第二轮：生命周期和 device mapping

目标：把 upstream lifecycle 改动影响集中到 policy 层。

改动：

1. 新增 `lifecycle.py`。
2. HCCL 使用 layerwise reload policy。
3. NPU IPC 使用 noop policy。
4. 新增 `device_mapping.py`，迁移 `get_ip()`、`npu_generate_uuid()`、visible device 解析。
5. 增加 device mapping UT。

验证：

- NPU IPC UT。
- HCCL/NPU IPC e2e。
- import collection 测试，确保 lazy import 不回退。

### 第三轮：trainer_send 编排抽象

目标：减少 IPC/HCCL trainer send 对 upstream payload/send_mode 改动的重复同步。

改动：

1. 新增 `trainer_send.py`。
2. 抽 metadata 收集、rank0 send、HTTP/Ray/callable payload 分发。
3. NPU IPC `_send_unpacked` / `_send_packed` 改为调用公共 orchestration。
4. HCCL `trainer_send_weights` 改为只保留 HCCL broadcast producer hook。
5. 增加 trainer_send UT，mock HTTP/Ray/callable。

验证：

- UT 覆盖 callable/http payload。
- NPU IPC one-card e2e。
- HCCL two-card e2e。

### 第四轮：packed tensor 拆分

目标：让 packed common 和 transport-specific 操作分离，为后续 sparse/chunked 扩展做准备。

改动：

1. 创建 `packed/` 子包。
2. 迁移 common chunk/buffer 逻辑。
3. 拆 HCCL packed broadcast 和 NPU IPC packed producer/consumer。
4. 保留旧 `packed_tensor.py` 作为兼容 re-export，降低一次性迁移风险。
5. 后续再删除旧入口。

验证：

- packed tensor UT。
- packed=True 的 HCCL/NPU IPC 路径 e2e。

## 重构后的 upstream 对齐方式

重构后，当 upstream vLLM 改 weight_transfer API 时，优先检查这些集中点：

1. `registry.py`：backend name、factory register API、backend config 类型变化。
2. `lifecycle.py`：start/update/finish 生命周期语义变化。
3. `trainer_send.py`：trainer args、send_mode、payload schema、HTTP/Ray 调用变化。
4. `examples/rl/weight_transfer_http_utils.py`：endpoint、request/response schema 变化。
5. backend engine：只在通信行为变化时修改，例如 HCCL/NPU IPC handle 或 packed transport。

预期收益：

- upstream 改 API：主要改公共层。
- 新增 backend：复用 lifecycle/trainer_send/packed common。
- 示例更新：只改 helper。
- 测试更新：只改公共 fixture/helper。
- code review 更容易聚焦真实 NPU 差异。

## 最终判断

当前 vllm-ascend weight_transfer 代码有重构必要性。

它的主要问题不是“代码量大”，而是 porting 形成的重复边界不清：大量 upstream 生命周期、trainer send、示例 HTTP 流程被复制到 Ascend backend 和示例中。短期这能快速适配 NPU，但长期会让 upstream vLLM API 对齐成本持续增加。

推荐按低风险到高风险分阶段推进：

1. 先收敛 registry/alias。
2. 再抽示例和测试 helper。
3. 然后抽 lifecycle/device mapping。
4. 再抽 trainer_send orchestration。
5. 最后拆 packed tensor。

这个顺序能先拿到同步成本下降的收益，同时避免一开始就改动 HCCL/NPU IPC 的核心传输路径。