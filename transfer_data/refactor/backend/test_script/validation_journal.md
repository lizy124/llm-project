# PR 13354 backend 验证执行记录

这份记录保留本轮怎么验证、怎么判断、哪些结果通过、哪些结果阻塞，便于后续复跑和继续补服务级 smoke。

## 1. 验证目标

PR 13354 的核心变更是重组 AscendStore KV Pool backend 目录结构。验证重点不是做 vLLM 全量回归，而是确认：

1. 新 package layout 是否完整。
2. 旧 flat 路径是否已移除。
3. 新路径 required import 是否成功。
4. `backend_map` 是否指向新 backend package。
5. `KVConnectorFactory` registry 是否指向新 connector package。
6. 与 backend/kv_transfer 相关的目标 UT 是否能跑。
7. 如果资源允许，按 kvpool 验证方式补 baseline + KV Pool 服务 smoke。

## 2. 环境

本轮主 run：

`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script/runs/20260804_034012_env`

关键信息：

- vllm-ascend：`pr-13354`
- vllm-ascend commit：`e79a5aa9ac95941c72030286392c827de1a74ef0`
- PR verified vLLM commit：`d02df748bf9efd99022f1a062597dc3cb3808485`
- 当前 vLLM commit：`568afb3a13806beb53bb2e6bd518269357b237c0`
- `mooncake` 已安装。
- `memcache_hybrid` 已安装。
- `lmcache_ascend` 未安装。
- `ucm` 未安装。

vLLM verified commit 没有切换成功，因为本地没有 `d02df748bf9efd99022f1a062597dc3cb3808485` 对象，此前访问 GitHub fetch 超时。

## 3. 脚本

脚本都放在：

`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script`

脚本分工：

- `00_env_snapshot.sh`：创建 run 目录并记录 commit、包路径、NPU、关键环境变量。
- `01_check_import_ut.sh`：required import、optional import、connector registry、py_compile、目标 pytest。
- `02_start_mooncake_master.sh`：启动 Mooncake master。
- `03_start_server_baseline.sh`：启动不开 KV Pool 的 baseline OpenAI server。
- `04_start_server_kvpool_custom.sh`：启动 AscendStoreConnector KV Pool OpenAI server。
- `05_send_requests.py`：非流式请求并保存 `requests.jsonl` 和 `summary.json`。
- `06_grep_logs.sh`：提取 backend/KV Pool 关键日志。
- `07_stop_run.sh`：停止 run 中记录的后台进程。
- `08_backend_structure_check.py`：结构、import、backend_map、registry 的机器可读检查。
- `09_send_stream_requests.py`：流式 TTFT 请求。
- `10_health_check.py`：轮询 `/v1/models`。

## 4. 实际执行过程

### 4.1 环境快照

执行：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
```

输出 run dir：

`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test_script/runs/20260804_034012_env`

归档文件：

`env.txt`

### 4.2 backend 结构检查

执行：

```bash
python3 08_backend_structure_check.py > ${RUN_DIR}/backend_structure_check.json
```

后续修正检查脚本本身的 registry 判断方式后复跑：

```bash
python3 08_backend_structure_check.py > ${RUN_DIR}/backend_structure_check_rerun.json
```

最终复跑结果：`ok=true`。

检查覆盖：

- `backend/__init__.py`、`backend/backend.py` 存在。
- `store_utils/ascend_store_connector.py`、`config_data.py`、`coordinator.py`、`kv_transfer.py`、`pool_scheduler.py`、`pool_worker.py` 存在。
- `mooncake_backend/mooncake_backend.py`、`memcache_backend/memcache_backend.py`、`yuanrong_backend/yuanrong_backend.py` 存在。
- `ucm_connector/ucm_connector.py`、`lmcache_ascend_connector/lmcache_ascend_connector.py` 存在。
- 旧根路径文件不存在。
- required imports 均通过。
- `backend_map` 指向新 backend package。
- connector registry 闭包里记录的新 module path 正确。

### 4.3 import、py_compile、pytest

执行：

```bash
./01_check_import_ut.sh ${RUN_DIR}
```

结果分三段：

1. Required import：通过。
2. Optional import：`ucm`、`lmcache_ascend` 缺依赖，记录为可选缺失。
3. py_compile：通过。
4. `tests/ut/distributed/ascend_store`：失败。

失败锚点：

```text
tests/ut/conftest.py:161
patch(f"{_pfx}.pool_worker.get_attention_compute_start_gate")
AttributeError: module 'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store' has no attribute 'pool_worker'
```

判断：

- PR 13354 后 `pool_worker` 已移动到 `ascend_store.store_utils.pool_worker`。
- `tests/ut/conftest.py` 仍按旧路径 patch。
- 所以全量 ascend_store pytest 被测试 fixture 旧路径阻塞。

### 4.4 直接 unittest 尝试

执行：

```bash
cd /vllm-workspace/vllm-ascend
python3 -m unittest discover -s tests/ut/distributed/ascend_store -p 'test_*.py' -v > ${RUN_DIR}/unittest_ascend_store.log 2>&1 || true
```

结果：失败。

失败原因：绕开 pytest fixture 后，测试 mock 与真实 `vllm`/`zmq` 导入环境冲突：

```text
ModuleNotFoundError: No module named 'zmq.asyncio'; 'zmq' is not a package
```

判断：

- `unittest` 不能作为本轮全量 UT 的有效替代。
- 继续以结构检查、import、py_compile 和更细目标 pytest 作为可采信证据。

### 4.5 额外目标 pytest

执行：

```bash
cd /vllm-workspace/vllm-ascend
python3 -m pytest tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py -q > ${RUN_DIR}/pytest_kv_transfer_failures.log 2>&1 || true
python3 -m pytest tests/ut/distributed/mooncake -q > ${RUN_DIR}/pytest_mooncake.log 2>&1 || true
```

结果：

- `test_kv_transfer_failures.py`：`16 passed, 14 warnings in 0.06s`。
- `tests/ut/distributed/mooncake`：`11 passed, 14 warnings in 0.06s`。

判断：

- `store_utils.kv_transfer` 相关失败记录逻辑通过目标测试。
- Mooncake 相关测试通过。
- 这两组测试没有被 `tests/ut/conftest.py` 的旧 ascend_store patch 路径阻断。

### 4.6 路径引用扫描

执行：

```bash
cd /vllm-workspace/vllm-ascend
rg -n 'kv_pool\.ascend_store\.(ascend_store_connector|config_data|coordinator|kv_transfer|pool_scheduler|pool_worker|mooncake_backend|memcache_backend|yuanrong_backend|ucm_connector|lmcache_ascend_connector)' vllm_ascend tests docs .github
rg -n 'kv_pool\.ascend_store\.(store_utils|mooncake_backend|memcache_backend|yuanrong_backend|ucm_connector|lmcache_ascend_connector)' vllm_ascend tests docs .github
```

归档：

`path_reference_scan.txt`

读法：

- backend package 引用如 `ascend_store.mooncake_backend.mooncake_backend` 是新 package 路径，不是旧 flat 文件。
- `tests/ut/conftest.py` 的旧 patch 路径没有被这条正则扫出，因为它通过 `_pfx` 拼接 `.pool_worker`，但 pytest 日志已经明确给出失败锚点。
- 大部分测试源码已经使用 `store_utils.*`。

### 4.7 服务级 smoke 资源门禁

按 kvpool 方式，服务级 smoke 原本应执行：

1. baseline server。
2. `/v1/models` health check。
3. 两次固定 prompt 请求。
4. Mooncake master + AscendStoreConnector server。
5. 两次固定 prompt 请求。
6. 提取 AscendStore/Mooncake/backend hit/load/save 日志。

本轮没有启动服务，原因：

- `npu-smi info` 显示所有 NPU chip 均有约 57GB 级别内存占用。
- `npu-smi` 显示的是宿主侧 PID，当前容器内 `ps` 查不到这些 PID。
- 当前容器内未见可管理的 `vllm.entrypoints.openai.api_server` 或 `mooncake_master`。
- 8100、50088、50089、50090 端口未见占用。
- 由于不能确认占用进程归属，不能安全 kill 或抢卡启动。

归档：

`resource_gate.txt`

## 5. 判断标准与结果

通过项：

- 新目录结构完整。
- required imports 通过。
- py_compile 通过。
- `backend_map` 正确。
- connector registry 正确。
- `test_kv_transfer_failures.py` 通过。
- `tests/ut/distributed/mooncake` 通过。

阻塞项：

- `tests/ut/distributed/ascend_store` 全量 pytest 被 `tests/ut/conftest.py` 旧 patch 路径阻塞。
- 服务级 smoke 被 NPU 外部占用阻塞。

## 6. 后续复跑建议

先修测试 fixture 路径：

- `tests/ut/conftest.py:161` 从 `ascend_store.pool_worker` 改为 `ascend_store.store_utils.pool_worker`。
- `tests/ut/conftest.py:163` 从 `ascend_store.config_data` 改为 `ascend_store.store_utils.config_data`。

然后复跑：

```bash
cd /vllm-workspace/vllm-ascend
python3 -m pytest tests/ut/distributed/ascend_store -q
python3 -m pytest tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py -q
python3 -m pytest tests/ut/distributed/mooncake -q
```

NPU 空闲后再按 `README.md` 的服务级 smoke 命令复跑 baseline + KV Pool。
