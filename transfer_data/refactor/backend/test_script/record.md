# PR 13354 backend 验证记录

详细步骤、启动命令、判断标准和阻塞项见：`validation_journal.md`。

## 当前状态

- vllm-ascend 分支：`pr-13354`。
- vllm-ascend commit：`e79a5aa9ac95941c72030286392c827de1a74ef0`。
- PR 声明的 vLLM verified commit：`d02df748bf9efd99022f1a062597dc3cb3808485`。
- 当前 vLLM commit：`568afb3a13806beb53bb2e6bd518269357b237c0`。
- verified vLLM commit 本地不存在；此前 GitHub fetch 超时，因此未切换。

## Run 记录

### Run 1：backend 结构、导入、目标 UT

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script/runs/20260804_034012_env`
- case：环境快照 + backend 结构检查 + import + py_compile + 目标 UT + 资源门禁
- 结果：部分通过

已通过项：

- `08_backend_structure_check.py` 复跑结果 `ok=true`。
- 新目录结构存在：`backend/`、`store_utils/`、`mooncake_backend/`、`memcache_backend/`、`yuanrong_backend/`、`ucm_connector/`、`lmcache_ascend_connector/`。
- 旧 flat 文件不存在：`ascend_store_connector.py`、`config_data.py`、`pool_worker.py` 等旧根路径文件均 absent。
- required imports 通过：`store_utils.*`、`mooncake_backend.*`、`memcache_backend.*`、`yuanrong_backend.*`。
- `backend_map` 指向新 backend package：`mooncake_backend.mooncake_backend`、`memcache_backend.memcache_backend`、`yuanrong_backend.yuanrong_backend`。
- `KVConnectorFactory` registry 指向新 connector package：
  - `AscendStoreConnector` -> `store_utils.ascend_store_connector`
  - `MooncakeConnectorStoreV1` -> `store_utils.ascend_store_connector`
  - `UCMConnector` -> `ucm_connector.ucm_connector`
  - `LMCacheAscendConnector` -> `lmcache_ascend_connector.lmcache_ascend_connector`
- `py_compile` 通过。
- `tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py`：`16 passed, 14 warnings in 0.06s`。
- `tests/ut/distributed/mooncake`：`11 passed, 14 warnings in 0.06s`。

失败/阻塞项：

- `python3 -m pytest tests/ut/distributed/ascend_store -q` 在 setup 阶段失败。
- 失败点：`tests/ut/conftest.py:161` 仍 patch `vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker`。
- PR 13354 重构后实际路径是 `vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.pool_worker`。
- 因此这项失败归类为测试 fixture 旧路径未同步，不是 backend import/registry 结构验证失败。
- `unittest discover` 也已尝试，失败于测试 mock 与真实 `vllm`/`zmq` 导入环境冲突：`ModuleNotFoundError: No module named 'zmq.asyncio'; 'zmq' is not a package`。
- 服务级 smoke 本轮未启动：`npu-smi` 显示所有卡有高内存占用，宿主侧 PID 在当前容器内不可见，不能安全清理或抢占。

## 当前结论

PR 13354 的 backend 重构在结构、required import、connector registry、backend_map 和两组相关目标 UT 上验证通过。全量 `tests/ut/distributed/ascend_store` 被 `tests/ut/conftest.py` 的旧 patch 路径阻塞，需要把 fixture 改到 `store_utils.pool_worker` 和 `store_utils.config_data` 后再重跑。服务级 KV Pool smoke 已准备脚本，但因当前 NPU 资源被外部进程占用，本轮没有启动。
