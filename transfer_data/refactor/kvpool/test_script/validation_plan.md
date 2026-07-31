# PR 13160 验证方案

目标 PR：`https://github.com/vllm-project/vllm-ascend/pull/13160`

## 1. 主要验证什么

PR 标题：`Simplify AscendStore KV pool Implementation and Unit Tests`。

验证重点应放在 AscendStore KV Pool，不做泛化的 vLLM 全量回归。

必须回答三个问题：

1. 池化基本功能是否通。
   - 服务能启动。
   - AscendStoreConnector 能初始化。
   - backend 能初始化。
   - 首次请求能保存 KV。
   - 第二次相同 prefix 请求能查询并加载 KV。

2. 结果是否正确。
   - 开 KV Pool 和不开 KV Pool，在固定 prompt、固定采样参数下输出应一致或等价。
   - 同一个长 prefix 的第一次请求和第二次请求不能出现异常、截断、乱码、明显语义偏移。
   - `kv_load_failure_policy=recompute` 场景下，命中失败不能导致请求失败。

3. 性能是否有可解释数据。
   - 记录 TTFT、总耗时、completion tokens、prompt tokens。
   - 重点看第二次相同 prefix 请求相对第一次是否有 TTFT 下降趋势。
   - 不把性能作为严格门禁，只给数据和趋势，因为单机环境、模型大小、后端状态会影响波动。

## 2. 版本前置

当前已切换：

- `vllm-ascend`: `/vllm-workspace/vllm-ascend`
- PR 分支：`pr-13160`
- 当前提交：`3a0404d2d`

PR 分支声明的配套 vLLM commit：

- `/vllm-workspace/vllm-ascend/.github/vllm-main-verified.commit`
- commit：`d02df748bf9efd99022f1a062597dc3cb3808485`

当前本机 vLLM：

- 路径：`/vllm-workspace/vllm`
- 当前提交：`752a3a5`
- 状态：detached HEAD

建议正式验证前先把 `/vllm-workspace/vllm` 切到 `d02df748bf9efd99022f1a062597dc3cb3808485`，减少 API 不匹配干扰。

## 3. 模型选择

本机已发现的候选模型：

- `/home/data/Qwen3.5-4B`
- `/home/data/Qwen3.5-4B-lora`
- `/mnt/weight/Qwen3-0.6B`
- `/mnt/weight/Qwen3-8B`
- `/mnt/weight/Qwen3-8B-W8A8`
- `/mnt/weight/DeepSeek-V2-Lite-Chat`
- `/mnt/weight/DeepSeek-V2-Lite-W8A8`
- `/mnt/weight/Qwen3-30B-A3B-W8A8`
- `/mnt/weight/vllm-ascend/Qwen3-30B-A3B-W8A8`

推荐顺序：

1. 第一轮冒烟：`/mnt/weight/Qwen3-0.6B`
   - 启动快，适合验证脚本、服务启动、请求链路、日志保存。

2. 第一轮正式功能：`/home/data/Qwen3.5-4B` 或 `/mnt/weight/Qwen3-8B`
   - 比 0.6B 更接近实际使用，又不会像大模型那样启动成本太高。

3. MLA/layerwise 扩展：`/mnt/weight/DeepSeek-V2-Lite-Chat` 或 `/mnt/weight/DeepSeek-V2-Lite-W8A8`
   - 用于后续验证 layerwise、hybrid/cache family 等风险点。

本 PR 第一轮建议使用 `Qwen3.5-4B`；如果资源紧张，先用 `Qwen3-0.6B` 跑通脚本。

## 4. 验证路径

第一阶段：环境和导入检查。

- 记录 git commit。
- 记录 Python 包位置。
- 跑 AscendStore 相关 import/py_compile。
- 如果 pytest 可用，跑 AscendStore UT。

第二阶段：不开 KV Pool 的 baseline。

- 启动普通 vLLM 服务。
- 固定 prompt 和参数发请求。
- 保存响应、耗时、日志。

第三阶段：开启 AscendStore KV Pool 的 PD-Mixed。

- 启动 Mooncake master。
- 启动 vLLM 服务，配置 `AscendStoreConnector`、`kv_role=kv_both`、`backend=mooncake`。
- 同一个长 prefix 连续请求两次。
- 保存两次响应、耗时、服务日志。
- 日志中检查 lookup、put、get、load、save、hit 等关键字。

第四阶段：异常和回退。

- 保持 `kv_load_failure_policy=recompute`。
- 人工制造一次后端不可用或 miss 场景。
- 验证请求不崩，能 fallback recompute。

第五阶段：可选扩展。

- PD Disaggregation 双实例。
- Memcache backend。
- Memcache layerwise。
- `consumer_is_to_put`。
- DeepSeek/MLA/hybrid 场景。

## 5. 脚本规划

所有脚本和记录统一放在：

`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script`

建议文件：

- `00_env_snapshot.sh`
  - 记录 git commit、Python 包位置、NPU 状态、环境变量。

- `01_check_import_ut.sh`
  - 做 import、py_compile、可选 pytest。

- `02_start_mooncake_master.sh`
  - 启动 Mooncake master，日志写入本次 run 目录。

- `03_start_server_baseline.sh`
  - 启动不开 KV Pool 的 baseline 服务。

- `04_start_server_kvpool_mixed.sh`
  - 启动开启 AscendStore KV Pool 的 PD-Mixed 服务。

- `05_send_requests.py`
  - 发固定请求。
  - 支持 baseline 和 kvpool 两种模式。
  - 记录每次请求的 prompt、响应、HTTP 状态、耗时、TTFT 如可获取、token usage。

- `06_grep_logs.sh`
  - 从服务日志中提取 AscendStore、Mooncake、lookup、put、get、load、save、hit、failure、recompute 等关键行。

- `README.md`
  - 写清楚每一步怎么跑。

- `record.md`
  - 记录每次实际执行结果和结论。

- `validation_journal.md`
  - 记录每个阶段的实际执行过程，包括启动命令、请求命令、关键输出、判断标准、失败原因、处理方式和复盘要点。
  - 后续每推进一个验证阶段，必须同步更新该文件，不能只更新 checklist。

## 6. 日志和结果保存

每次验证使用独立 run 目录：

`runs/YYYYMMDD_HHMMSS_<case_name>/`

目录内容：

- `env.txt`
  - commit、包路径、NPU、关键环境变量。

- `mooncake_master.log`
  - Mooncake master 日志。

- `server.log`
  - vLLM 服务日志。

- `requests.jsonl`
  - 每次请求一行 JSON。

- `summary.json`
  - 本次请求数量、成功数量、失败数量、首轮耗时、二轮耗时、是否观察到命中日志。

- `log_extract.txt`
  - 关键日志提取结果。

- `conclusion.md`
  - 人工结论：通过/失败/阻塞，失败原因和下一步。

请求结果必须保存。否则无法判断输出正确性，也无法复盘性能数据。

## 7. 第一轮通过标准

第一轮只要求 Mooncake PD-Mixed 单实例通过：

- vLLM 服务启动成功。
- 首次请求成功返回。
- 第二次相同 prefix 请求成功返回。
- 日志可见 AscendStoreConnector 初始化，以及 AscendStore/Mooncake 的 hit、put/get 或 load/save 等任一关键路径证据。
- 第二次请求没有 load failure、block 提前释放、线程异常等错误。
- baseline 和 kvpool 输出在固定参数下无明显不一致。
- 记录 TTFT/耗时数据，能说明是否有命中后的趋势改善。

## 8. 不作为第一轮门禁

- 大规模性能压测。
- 多节点 PD disaggregation。
- Memcache layerwise。
- consumer 写回。
- 多 TP/DP/PP 组合。
- 所有模型全覆盖。

## 9. 执行进度

- [x] 2026-07-31：完成验证目标、模型选择、日志保存规则和脚本规划。
- [x] 创建环境快照脚本：`00_env_snapshot.sh`。
- [x] 创建导入检查/UT 脚本：`01_check_import_ut.sh`。
- [x] 创建 Mooncake master 启动脚本：`02_start_mooncake_master.sh`。
- [x] 创建 baseline 服务启动脚本：`03_start_server_baseline.sh`。
- [x] 创建 KV Pool PD-Mixed 服务启动脚本：`04_start_server_kvpool_mixed.sh`。
- [x] 创建请求发送和结果保存脚本：`05_send_requests.py`。
- [x] 创建日志提取脚本：`06_grep_logs.sh`。
- [x] 创建执行说明和结果记录模板：`README.md`、`record.md`。
- [x] 运行第一阶段环境和导入检查：切到 verified vLLM commit 后，import、py_compile、AscendStore UT 均通过，`146 passed, 14 warnings in 2.14s`。
- [x] 切换 `/vllm-workspace/vllm` 到 PR 声明的 verified commit：当前为 `d02df748bf9efd99022f1a062597dc3cb3808485`。
- [x] 启动服务并执行 baseline 请求：使用 `/mnt/weight/Qwen3-0.6B`、`ASCEND_RT_VISIBLE_DEVICES=8`，两次请求均返回 200，输出一致。
- [x] 启动 Mooncake + KV Pool 服务并执行二次命中请求：使用 `/mnt/weight/Qwen3-0.6B`、`ASCEND_RT_VISIBLE_DEVICES=8`，两次请求均返回 200，输出一致，日志可见 `AscendStoreConnector` 和 `kvpool hit tokens: 1024`。
- [x] 汇总 0.6B 冒烟验证结论：详见 `record.md` 和 `validation_journal.md`。
- [x] 尝试使用 `/home/data/Qwen3.5-4B` 执行第一轮正式功能验证：启动失败，该目录缺少 vLLM 可识别的 `config.json` 或 `params.json`。
- [x] 使用备选正式模型 `/mnt/weight/Qwen3-8B` 完成第一轮正式功能验证：baseline 与 KV Pool 请求均成功，输出和 usage 一致，日志可见 `AscendStoreConnector` 与 `kvpool hit tokens: 1024`。
- [x] 补充 `kv_load_failure_policy=recompute` miss/new-prefix 轻量回退验证：请求返回 200，未因 miss 或后端状态异常导致失败。
- [ ] 设计更强的后端 load failure 注入场景，明确观察 recompute 日志。
