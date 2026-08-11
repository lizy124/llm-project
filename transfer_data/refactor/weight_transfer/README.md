# vLLM Ascend Weight Transfer 重构记录

这个目录记录 vllm-ascend `weight_transfer` 的核心流程、重构设计、Ascend PR 历史和 upstream vLLM API 演进。

```text
代码仓库: D:/lzy/project/kv_pool/code/vllm-ascend
分支: weight_transfer_refactor
HEAD: c5ed02f00
远端: origin/weight_transfer_refactor = c5ed02f00
```

远端原有的 5 个实验性重构提交已经被 PR1 替换，不应再把旧实验分支当作当前实现。

文档更新时间：2026-08-10。

## 文档索引

推荐阅读顺序：

1. `01-weight-transfer-core-flow.md`：当前 worker、trainer、control-plane 和 data-plane 流程。
2. `02-refactor-design.md`：重构边界、版本兼容、目标结构和验收标准。
3. `03-vllm-ascend-pr-history.md`：Ascend 侧功能引入、upstream 同步和实验提交历史。
4. `04-upstream-vllm-timeline.md`：upstream 从 NCCL/IPC 到 Stateful Trainer Send 的演进。
5. `05-pr-13049-validation-plan.md`：PR 13049 的真实改动范围、验证层级和验收模板。
6. `06-pr-13049-validation-record.md`：PR 13049 在当前 Ascend 服务器上的实际验证结果。
7. `pr-13049-validation/`：PR 13049 的过程目录，包含脚本、环境和结果摘要。

```text
01  当前协议如何运行
02  这次应该如何重构
03  Ascend 代码为什么演进成现在这样
04  upstream contract 为什么持续变化
05  PR 13049 应该如何验证
06  PR 13049 实际验证结果
07  PR 13049 过程目录
```

## 当前结论

可以继续进行第二阶段重构，但必须基于当前代码和 upstream 实际能力：

- upstream 已包含 `TrainerWeightTransferEngine`、`WeightTransferTrainerFactory` 和 `clients.py`。
- vLLM v0.26 虽有 trainer factory 类，但尚未注册 stateful trainer backend，不能只检查类是否存在。
- PR1 已集中 registry、compat、lazy-load 和 HTTP 示例/e2e 请求 helper。
- lifecycle、device identity、trainer orchestration 和 packed transport 重构尚未开始。

## PR1 已完成

PR1 只收敛注册和 control-plane helper，没有修改 HCCL/NPU IPC transport loop。

### 提交

```text
84b09f3e2 refactor(weight-transfer): centralize registration and HTTP helpers
c5ed02f00 fix(weight-transfer): detect stateful trainer capability
```

### 注册与兼容

新增：

```text
vllm_ascend/distributed/weight_transfer/compat.py
vllm_ascend/distributed/weight_transfer/registry.py
```

当前映射：

```text
hccl    -> HCCLWeightTransferEngine
npu_ipc -> NPUIPCWeightTransferEngine
nccl    -> hccl
ipc     -> npu_ipc
```

约束：

- engine loader 保持 lazy import。
- plugin register 和 platform patch 使用同一个幂等入口。
- `nccl`/`ipc` alias 由统一 registry 替换。
- 只有 upstream trainer registry 已存在 `ipc` 时，才注册 Ascend `npu_ipc` trainer engine。
- 注册阶段不 import Ray、`torch_npu`，也不初始化通信资源。

### HTTP 示例与 e2e helper

新增：

```text
examples/rl/weight_transfer_http_utils.py
tests/e2e/pull_request/weight_transfer_utils.py
```

已清理两个 RL HTTP 示例和两个 HCCL/NPU IPC e2e。helper 只负责 URL、HTTP 请求、timeout、状态码检查和 background POST 异常传播；backend payload、IPC 序列化和 data-plane 保持原位。

### UT

新增：

```text
tests/ut/distributed/weight_transfer/test_compat.py
tests/ut/distributed/weight_transfer/test_registry.py
tests/ut/distributed/weight_transfer/test_http_utils.py
```

覆盖 alias/幂等注册、lazy loader、trainer capability，以及 HTTP URL、payload、timeout 和错误传播。

## PR1 验证状态

已完成：

```text
Python AST parse: passed
Python compileall: passed
git diff --check: passed
registry stub smoke test: passed
HTTP helper stub smoke test: passed
```

当前执行环境没有安装 `pytest`、`ruff`、`torch` 和 `requests`，因此尚未执行真实 vLLM import、UT、lint 或 Ascend e2e。旧 README 中的“39 passed / 88 passed”不属于当前 PR1，不能继续引用。

完整环境应执行：

```text
pytest -q tests/ut/distributed/weight_transfer
ruff check vllm_ascend/distributed/weight_transfer examples/rl tests/ut/distributed/weight_transfer
pytest tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
pytest tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

## PR2 计划

PR2 处理高风险的 worker/trainer 编排和 data-plane 边界，必须单独 review。

### Compatibility adapter

- 对齐 `TrainerWeightTransferEngine` 与 legacy static trainer API。
- 按 trainer registry capability 判断，不依赖单一版本号。
- 复用 upstream `clients.py`，不复制通用 Ray/HTTP client。

### Lifecycle

- 明确 `init/start/update/finish/shutdown` 合法顺序。
- 保留 backend-specific layerwise reload/no-op 行为。
- 处理失败、cleanup、draft target 和 weight version。

### Device identity

- 分离 logical/physical NPU mapping 与 host/device IPC identity。
- 覆盖 `ASCEND_RT_VISIBLE_DEVICES`、多网卡、容器和非法映射。

### Trainer sender

- 公共层只负责 metadata 和 lifecycle 编排。
- HCCL 保留 communicator/broadcast。
- NPU IPC 保留 handle 创建、汇聚、序列化和 rebuild。
- legacy/static 与 stateful API 复用相同 wire contract。

### Packed contract

- 固定 names、shapes、dtypes、tensor_sizes 和 chunk ordering。
- 覆盖空输入、尾块、超大 tensor 和 producer/consumer 不一致。
- 不无理由修改现有 packed wire format。

## 非目标

- 不 fork upstream `base.py`、`factory.py` 或 `clients.py`。
- 不合并 HCCL 与 NPU IPC engine。
- 不修改 upstream HTTP schema 或 IPC pickle/base64 协议。
- 不新增跨主机 NPU IPC。
- 不在结构重构中顺带优化通信性能。

## 开始 PR2 前

- [ ] 在完整环境运行 PR1 UT 和 lint。
- [ ] 保存 HCCL/NPU IPC 当前 e2e 基线。
- [ ] 确认 v0.25、v0.26、v0.27 和 current main trainer capability。
- [ ] 重新核对 upstream `base.py`、`factory.py`、`clients.py` 和 backend engines。
- [ ] 保持 PR2 不包含无关模块改动。

每个重构提交都必须回答：

1. 这是 upstream contract，还是 Ascend transport 差异？
2. 是否改变 HTTP/Ray/callable 或 packed wire 行为？
3. 无 NPU 环境能否通过纯 Python/mock 验证？

```text
upstream API       -> compat / adapter / registry
control-plane      -> upstream clients + thin example helper
lifecycle          -> lifecycle policy + backend hooks
HCCL transport     -> hccl engine/sender
NPU IPC transport  -> npu_ipc engine/sender
packed protocol    -> explicit contract and joint tests
```

## 后续 UT 补充遗留

评审反馈：weight_transfer 代码本身已足够简洁，不需要加抽象层重构；应增加 UT 和使用用例。
因此 PR #13049 已回退 `compat.py`/`registry.py` 抽象层，只保留 HTTP/e2e helper 抽取和 UT。
后续 UT 补充分到后续 PR，按优先级排列如下（均可在 CPU 环境运行，无需 NPU 硬件）。

### 优先级 2: `hccl_engine.py`（当前覆盖率 0%）

源文件：`vllm_ascend/distributed/weight_transfer/hccl_engine.py`（340 行）

| 测试名 | 覆盖点 |
|---|---|
| `test_hccl_init_info_post_init_valid` | `HCCLWeightTransferUpdateInfo.__post_init__` 三 list 长度一致 |
| `test_hccl_init_info_post_init_mismatched_dtype` | `__post_init__` dtype_names 长度不一致 raise ValueError |
| `test_hccl_init_info_post_init_mismatched_shapes` | `__post_init__` shapes 长度不一致 raise ValueError |
| `test_hccl_receive_weights_packed` | `receive_weights` packed 路径，mock `packed_broadcast_consumer` |
| `test_hccl_receive_weights_unpacked` | `receive_weights` unpacked 路径，mock `group.broadcast` + `model.load_weights` |
| `test_hccl_receive_weights_not_initialized` | `receive_weights` 未初始化时 raise RuntimeError |
| `test_hccl_trainer_send_weights_packed` | `trainer_send_weights` packed 路径，mock `packed_broadcast_producer` |
| `test_hccl_trainer_send_weights_dict_args` | dict 入参自动转 `HCCLTrainerSendWeightsArgs` |
| `test_hccl_trainer_send_weights_default_post_iter_func` | post_iter_func=None 时取 x[1] |
| `test_hccl_shutdown_clears_group` | `shutdown` group 被置 None |

预期覆盖率：0% → ~70%

### 优先级 3: `npu_ipc_engine.py`（当前覆盖率 ~15%）

源文件：`vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py`（777 行）
现有 UT：`test_npu_ipc_engine.py`（4 个 test，覆盖 init/receive_unpacked/start/finish）

| 测试名 | 覆盖点 |
|---|---|
| `test_npu_generate_uuid_identity_mapping` | 无 `ASCEND_RT_VISIBLE_DEVICES`，logical=physical |
| `test_npu_generate_uuid_visible_devices_mapping` | `ASCEND_RT_VISIBLE_DEVICES=2,3`，logical 0→physical 2 |
| `test_npu_generate_uuid_cached` | `@lru_cache(1)` 幂等 |
| `test_get_ip_fallback_to_hostname` | socket 异常时 fallback |
| `test_receive_weights_packed` | `receive_weights` packed 路径，mock `packed_npu_ipc_consumer` |
| `test_receive_weights_missing_uuid` | `receive_weights` unpacked 路径 UUID 不匹配 raise ValueError |
| `test_parse_update_info_pickled_path` | `parse_update_info` `ipc_handles_pickled` 字段反序列化 |
| `test_parse_update_info_both_fields_raises` | 同时传 `ipc_handles` + `ipc_handles_pickled` raise |
| `test_parse_update_info_insecure_disabled` | `VLLM_ALLOW_INSECURE_SERIALIZATION=0` raise |
| `test_trainer_send_weights_packed` | `trainer_send_weights` packed，mock `_send_packed` |
| `test_trainer_send_weights_unpacked` | `trainer_send_weights` unpacked，mock `_send_unpacked` |
| `test_is_rank_zero_no_distributed` | `_is_rank_zero` 未初始化时返回 True |
| `test_all_gather_and_merge_handles_no_distributed` | `_all_gather_and_merge_handles` 未初始化时直接返回 |
| `test_do_send_http_transport` | `_do_send` http 路径，mock requests.post，验证 base64+pickle |
| `test_do_send_callable_send_mode` | `_do_send` callable 路径直接调用 |

预期覆盖率：~15% → ~60%

### 优先级 4: `__init__.py`（当前覆盖率 0%）

源文件：`vllm_ascend/distributed/weight_transfer/__init__.py`（47 行）

| 测试名 | 覆盖点 |
|---|---|
| `test_register_engine_registers_hccl` | mock factory，验证 hccl 注册 |
| `test_register_engine_registers_npu_ipc` | mock factory，验证 npu_ipc 注册 |
| `test_register_engine_v026_skips_trainer` | v0.26.0 时不注册 trainer |
| `test_register_engine_post_v026_registers_trainer` | 非 v0.26.0 时注册 trainer |

预期覆盖率：0% → ~95%（防 alias 回归）

### 优先级 5: `examples/rl/weight_transfer_http_utils.py`（当前覆盖率 ~50%）

源文件：`examples/rl/weight_transfer_http_utils.py`（84 行）
现有 UT：`test_http_utils.py`（3 个 test，覆盖 post_endpoint/start_weight_update/get_world_size）

| 测试名 | 覆盖点 |
|---|---|
| `test_pause_calls_post` | `pause` 验证 `/pause` 调用 |
| `test_resume_calls_post` | `resume` 验证 `/resume` 调用 |
| `test_init_weight_transfer_calls_post` | `init_weight_transfer` 验证 `/init_weight_transfer` 调用 |
| `test_update_weights_calls_post` | `update_weights` 验证 `/update_weights` 调用 |
| `test_finish_weight_update_calls_post` | `finish_weight_update` 验证 `/finish_weight_update` 调用 |
| `test_post_endpoint_raises_on_http_error` | `post_weight_transfer_endpoint` raise_for_status 抛异常时传播 |

预期覆盖率：~50% → ~95%

### 总览

| 文件 | 现有覆盖率 | 补完后预期 | 新增 UT 数 |
|---|---|---|---|
| `packed_tensor.py` | ✅ 已完成（18 个 UT） | ~85% | 0（已完成） |
| `hccl_engine.py` | 0% | ~70% | 10 |
| `npu_ipc_engine.py` | ~15% | ~60% | 15 |
| `__init__.py` | 0% | ~95% | 4 |
| `weight_transfer_http_utils.py` | ~50% | ~95% | 6 |
| **合计** | — | — | **35** |
