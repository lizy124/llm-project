# PR 13160 验证记录

详细步骤、启动命令、请求方式、判断标准和学习记录见：`validation_journal.md`。

## 当前状态

- vllm-ascend 已切到 PR 13160 本地分支：`pr-13160`。
- 当前 vllm-ascend commit：`3a0404d2d`。
- PR 声明的 vLLM verified commit：`d02df748bf9efd99022f1a062597dc3cb3808485`。
- 当前 vLLM commit：`d02df748bf9efd99022f1a062597dc3cb3808485`。

## Run 记录

### Run 1

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_064826_env`
- case：环境检查 + 导入检查 + py_compile + AscendStore UT 尝试
- model：无
- vllm commit：`752a3a5`
- vllm-ascend commit：`3a0404d2d`
- 结果：部分通过
- 关键日志：
  - `vllm`、`vllm_ascend`、AscendStore 相关模块 import 通过。
  - AscendStore 相关文件 `py_compile` 通过。
  - pytest 启动失败：`ModuleNotFoundError: No module named 'vllm.third_party.flash_linear_attention'`。
- 结论：当前 vLLM commit 与 PR 期望环境可能不匹配；应先切 `/vllm-workspace/vllm` 到 PR 声明的 verified commit `d02df748bf9efd99022f1a062597dc3cb3808485` 后重跑 UT。

### vLLM verified commit 切换尝试

- 目标 commit：`d02df748bf9efd99022f1a062597dc3cb3808485`
- 执行结果：未切换成功。
- 原因：`git fetch origin d02df748bf9efd99022f1a062597dc3cb3808485` 访问 GitHub 时 TLS 连接中断；本地仓库没有该 commit 对象。
- 当前 vLLM 仍为：`752a3a504485790a2e8491cacbb35c137339ad34`
- 影响：UT 仍可能因 vLLM API/目录不匹配失败；服务验证前建议重试 fetch 或通过离线方式同步该 commit。

### 脚本语法检查

- shell：`common.sh`、`00_env_snapshot.sh`、`01_check_import_ut.sh`、`02_start_mooncake_master.sh`、`03_start_server_baseline.sh`、`04_start_server_kvpool_mixed.sh`、`06_grep_logs.sh`、`07_stop_run.sh` 均通过 `bash -n`。
- Python：`05_send_requests.py` 通过 `python3 -m py_compile`。

## 第一轮结论

截至 2026-07-31 08:05 左右，0.6B 冒烟验证已通过：

- vLLM 已切到 PR 声明的 verified commit：`d02df748bf9efd99022f1a062597dc3cb3808485`。
- 环境检查、AscendStore import、py_compile、UT 通过：`146 passed, 14 warnings in 2.14s`。
- baseline 服务在空闲设备 `ASCEND_RT_VISIBLE_DEVICES=8` 上启动成功，两次请求均返回 200。
- Mooncake + KV Pool PD-Mixed 服务在同一设备上启动成功，两次请求均返回 200。
- baseline 与 KV Pool 输出文本一致，usage 一致。
- KV Pool 日志中看到 `AscendStoreConnector` 创建，以及 `kvpool hit tokens: 1024, need to load: 1024`、`KV pool load spec created`。

需要继续的项：

- 更严格的性能数据采集，当前 `05_send_requests.py` 记录的是请求总耗时，不是严格 TTFT。

## 第一轮正式功能验证补充

截至 2026-07-31 09:10 左右，正式模型轮次已继续执行：

- 原计划模型 `/home/data/Qwen3.5-4B` 启动失败，原因是该目录不是 vLLM 可识别的模型目录，缺少 `config.json` 或 `params.json`。
- 按验证计划中的备选正式模型，改用 `/mnt/weight/Qwen3-8B` 继续验证。
- baseline run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_090026_env`。
- KV Pool run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_090307_env`。
- 设备：`ASCEND_RT_VISIBLE_DEVICES=8`。

baseline Qwen3-8B 结果：

- 两次请求均返回 200。
- `elapsed_sec=[5.843755770009011, 2.397889809915796]`。
- usage 一致：`prompt_tokens=1143`、`completion_tokens=64`、`total_tokens=1207`。
- `same_text_as_first=[true, true]`。

Mooncake + KV Pool Qwen3-8B 结果：

- 两次请求均返回 200。
- `requests.jsonl` 中两次正式请求耗时为 `2.3475408400408924` 和 `2.232989399926737`。
- usage 一致：`prompt_tokens=1143`、`completion_tokens=64`、`total_tokens=1207`。
- 两次输出文本一致。
- 日志可见 `kv_load_failure_policy='recompute'`、`Creating v1 connector with name: AscendStoreConnector`、`kvpool hit tokens: 1024, need to load: 1024`、`KV pool load spec created`。

recompute / miss 回退验证：

- 在同一 KV Pool 服务中继续执行。
- 先尝试停止记录的 Mooncake master PID，结果该 PID 已不存在；进程表中也未见实际 Mooncake master 进程。
- 对相同 prefix 继续发两次请求，均返回 200，耗时为 `2.2125850699376315` 和 `2.413814039900899`。
- 再使用一个全新 prefix 发单次请求，模拟外部 KV pool miss，返回 200，耗时为 `2.2373886699788272`，usage 为 `prompt_tokens=1304`、`completion_tokens=64`、`total_tokens=1368`。
- 日志未见 `traceback`、`exception`、请求失败或 load failure 导致请求失败的记录。
- 限制：这轮验证证明了 `kv_load_failure_policy=recompute` 配置存在且 miss/new-prefix 请求不失败；但没有制造出明确的后端 load failure 日志，因此不能算强故障注入，只能算轻量回退验证。

注意：`05_send_requests.py` 每次运行会重写 `summary.json`，所以 KV Pool run 的完整请求序列以 `requests.jsonl` 为准。
