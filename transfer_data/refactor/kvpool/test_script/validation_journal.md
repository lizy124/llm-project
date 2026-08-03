# PR 13160 验证执行记录

这份记录的目的不是只记“通过/失败”，而是把**怎么验证、怎么启动、怎么发请求、怎么判断正常、结果怎么读**都留下来，方便后续复跑和学习。

## 1. 这轮验证在验证什么

PR 目标是 `Simplify AscendStore KV pool Implementation and Unit Tests`，所以这轮只看 AscendStore KV Pool，不做泛化的全量 vLLM 回归。

本轮已经实际跑过的内容：

- 环境和导入检查
- baseline 服务验证
- Mooncake + KV Pool PD-Mixed 验证

本轮已继续补充的内容：

- `/home/data/Qwen3.5-4B` 启动可用性检查
- `/mnt/weight/Qwen3-8B` 正式模型轮次
- `kv_load_failure_policy=recompute` 的 miss/new-prefix 轻量回退验证

## 2. 现有脚本是怎么分工的

脚本都在：

`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script`

核心脚本如下：

- `00_env_snapshot.sh`
  - 创建 run 目录
  - 记录当前 git commit、Python 包位置、NPU 状态、环境变量

- `01_check_import_ut.sh`
  - 做 import 检查
  - 做 `py_compile`
  - 如果有 pytest，就跑 AscendStore 单测

- `02_start_mooncake_master.sh`
  - 后台启动 Mooncake master
  - 把 PID 写到 run 目录

- `03_start_server_baseline.sh`
  - 不开 KV Pool 的 baseline 服务

- `04_start_server_kvpool_mixed.sh`
  - 开启 `AscendStoreConnector`
  - 设置 `kv_role=kv_both`
  - 设置 `kv_load_failure_policy=recompute`
  - backend 走 `mooncake`

- `05_send_requests.py`
  - 向 `/v1/completions` 发请求
  - 默认发两次
  - 记录 `requests.jsonl` 和 `summary.json`

- `06_grep_logs.sh`
  - 从服务日志和 Mooncake 日志里提取关键字

- `07_stop_run.sh`
  - 停掉 run 里的后台服务

## 3. 怎么启动一轮验证

### 3.1 先建 run 目录并快照环境

每次验证都先生成一个独立 run 目录：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
echo "$RUN_DIR"
```

这一步会生成类似这样的目录：

- `runs/20260731_074619_env`
- `runs/20260731_075703_env`
- `runs/20260731_075952_env`

### 3.2 先跑环境和导入检查

命令：

```bash
./01_check_import_ut.sh "$RUN_DIR"
```

这一步会在 `check_import_ut.log` 里记录：

- `import` 是否成功
- `py_compile` 是否成功
- `pytest` 是否成功

### 3.3 baseline 怎么启动

README 里给的 baseline 启动方式是：

```bash
MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"
```

但这次实际验证时，发现默认设备 `0` 上显存不够，所以最终是这样跑通的：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"
```

这里的 `8` 对应空闲的 NPU 4 / chip 8。

### 3.4 怎么判断 baseline 服务起来了

脚本本身只是后台拉起服务，不自动等健康检查，所以我额外轮询：

```bash
python - <<'PY' "$RUN_DIR"
import sys, time, urllib.request
run_dir=sys.argv[1]
url='http://127.0.0.1:8100/v1/models'
last=''
for i in range(240):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            print(f'HEALTH_OK status={r.status} attempt={i+1}')
            print(f'RUN_DIR={run_dir}')
            raise SystemExit(0)
    except Exception as e:
        last=repr(e)
        time.sleep(2)
print(f'HEALTH_FAIL last={last}')
print(f'RUN_DIR={run_dir}')
raise SystemExit(1)
PY
```

判断标准很简单：

- `/v1/models` 返回 200
- 说明 OpenAI API server 已经起来了

### 3.5 怎么发请求

请求脚本是：

```bash
python3 05_send_requests.py --run-dir "$RUN_DIR" --case baseline --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
```

它做的事情是：

- 固定一个长 prompt
- 向 `http://127.0.0.1:8100/v1/completions` 发 POST
- 发两次，验证同 prompt 重复请求
- 把每次请求的状态、耗时、usage、返回文本写到 `requests.jsonl`
- 把汇总写到 `summary.json`

注意：

- 这个脚本记录的是**请求总耗时 `elapsed_sec`**
- 它**不是严格意义上的 TTFT**
- 现在如果要看 TTFT，只能另外加流式统计或更细粒度日志

### 3.6 baseline 怎么判断正常

看这几个点：

- `server_baseline.log` 里没有启动失败
- `/v1/models` 返回 200
- `requests.jsonl` 里两条请求都 `status=200`
- `summary.json` 里 `success_count=2`
- `summary.json` 里 `same_text_as_first=[true, true]`
- `06_grep_logs.sh` 能提取到正常服务日志

### 3.7 Mooncake + KV Pool 怎么启动

先起 Mooncake master：

```bash
./02_start_mooncake_master.sh "$RUN_DIR"
```

再起 KV Pool mixed 服务：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./04_start_server_kvpool_mixed.sh "$RUN_DIR"
```

这一步会把这些关键配置写进服务启动参数：

- `kv_connector=AscendStoreConnector`
- `kv_role=kv_both`
- `kv_load_failure_policy=recompute`
- `backend=mooncake`

### 3.8 KV Pool 服务怎么判断正常

仍然是三层判断：

1. 进程/日志层
   - Mooncake master 进程在跑
   - vLLM 服务没有报 EngineCore 初始化失败

2. 健康检查层
   - `/v1/models` 返回 200

3. 请求结果层
   - 两次请求都 200
   - `summary.json` 里两次文本一致
   - `06_grep_logs.sh` 能提取到 AscendStore / Mooncake / kvpool 相关关键日志

## 4. 实际跑了哪些 run

### Run A：环境和导入检查

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_074619_env`
- 内容：环境快照 + import + py_compile + pytest
- 结果：通过
- 关键结果：`146 passed, 14 warnings in 2.14s`

### Run B：baseline 冒烟

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_075703_env`
- 模型：`/mnt/weight/Qwen3-0.6B`
- 设备：`ASCEND_RT_VISIBLE_DEVICES=8`
- 结果：通过

`summary.json` 里的结果：

- `request_count=2`
- `success_count=2`
- `failure_count=0`
- `elapsed_sec=[10.210297629935667, 1.8054288099519908]`
- `same_text_as_first=[true, true]`

### Run C：Mooncake + KV Pool 冒烟

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_075952_env`
- 模型：`/mnt/weight/Qwen3-0.6B`
- 设备：`ASCEND_RT_VISIBLE_DEVICES=8`
- 结果：通过

`summary.json` 里的结果：

- `request_count=2`
- `success_count=2`
- `failure_count=0`
- `elapsed_sec=[1.8915102898608893, 1.8318590300623327]`
- `same_text_as_first=[true, true]`

### 失败尝试：默认设备启动 baseline

- 现象：服务没起来，请求连接拒绝
- 根因：默认设备 `0` 显存不足
- `server_baseline.log` 里的关键报错：
  - `Free memory on device (10.53/61.27 GiB) ... is less than desired GPU memory utilization (0.92, 56.37 GiB)`
- 处理：换到空闲设备 `ASCEND_RT_VISIBLE_DEVICES=8` 后重跑成功

这个失败很有价值，因为它说明：

- 验证脚本本身没问题
- 但**设备选择**会直接决定服务能不能起来
- 所以后面复跑时，先看 `npu-smi info`

## 5. 日志里看到了什么

### baseline 日志

baseline 启动成功后，日志里能看到：

- vLLM 服务路由注册完成
- 模型加载完成
- `/v1/models` 访问 200

### KV Pool 日志

`server_kvpool_mixed.log` 和 `log_extract.txt` 里能看到这些关键点：

- `Creating v1 connector with name: AscendStoreConnector`
- `kv_load_failure_policy='recompute'`
- `kvpool hit tokens: 1024, need to load: 1024`
- `KV pool load spec created`

Mooncake master 日志里能看到：

- `Master service started on port 50088`
- `service_ready=true`
- 后续有 `Ping`、`FetchTasks` 等正常指标更新

## 6. 怎么判断“这次验证是正常的”

我这次用的是下面这套判断：

- 版本对齐
  - `vllm` 已切到 `d02df748bf9efd99022f1a062597dc3cb3808485`

- 环境通过
  - `import` 成功
  - `py_compile` 成功
  - `pytest` 成功

- 服务能启动
  - baseline 能起
  - Mooncake master 能起
  - KV Pool 服务能起

- 请求能成功
  - 两次请求都返回 200
  - `summary.json` 里 `success_count=2`
  - 没有 `error`

- 输出正确性
  - 同一个 run 内两次文本一致
  - baseline 与 KV Pool 的输出文本也一致

- 日志能证明路径被走到了
  - baseline 有正常 vLLM 启动日志
  - KV Pool 有 `AscendStoreConnector`、`kvpool hit/load`、`recompute` 相关日志

## 7. 这轮学到的东西

1. `00_env_snapshot.sh` 先创建 run 目录，再统一把后续文件落进去，这样每次验证都可复盘。

2. `03_start_server_baseline.sh` 和 `04_start_server_kvpool_mixed.sh` 都是后台启动，不等健康；所以必须自己加 `/v1/models` 轮询。

3. `05_send_requests.py` 现在更像“功能验证脚本”，不是“性能剖面脚本”。
   - 它能告诉你请求是否成功、返回文本是否一致、总耗时是多少
   - 但不能直接给严格 TTFT

4. 设备显存很关键。
   - 默认设备未必能跑
   - 先看 `npu-smi info`，再选空闲卡，会少很多误判

5. baseline 和 KV Pool 的“快慢”不能只看一次两次。
   - baseline 第一次通常有冷启动/加载开销
   - 现在这两轮更适合看“链路是否通”和“输出是否一致”
   - 真要比较性能，要再做更严格的同条件重复

## 8. 现在到哪里了

当前已经完成：

- 验证目标和脚本框架
- 环境和导入检查
- baseline 0.6B 冒烟
- Mooncake + KV Pool 0.6B 冒烟
- `/home/data/Qwen3.5-4B` 模型目录可用性检查
- baseline Qwen3-8B 正式功能验证
- Mooncake + KV Pool Qwen3-8B 正式功能验证
- `recompute` miss/new-prefix 轻量回退验证
- 结果日志和汇总文件都已落盘

当前未完成：

- 更完整的性能对比
- 更强的后端故障注入验证，例如明确制造 load failure 并观察 recompute 日志

## 9. 相关文件和结果

- 计划：`validation_plan.md`
- 当前高层记录：`record.md`
- 环境检查 run：`runs/20260731_074619_env/`
- baseline 0.6B run：`runs/20260731_075703_env/`
- KV Pool 0.6B run：`runs/20260731_075952_env/`
- baseline Qwen3-8B run：`runs/20260731_090026_env/`
- KV Pool Qwen3-8B / recompute run：`runs/20260731_090307_env/`

## 10. 2026-07-31 09:10 补充验证

### `/home/data/Qwen3.5-4B` 启动尝试

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_084548_env`
- 命令：`ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/home/data/Qwen3.5-4B PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"`
- 结果：失败
- 关键报错：`Invalid repository ID or local directory specified: '/home/data/Qwen3.5-4B'`，并提示需要 `config.json` 或 `params.json`
- 结论：该目录当前不能作为 vLLM 模型目录使用，正式功能验证改用计划中的备选 `/mnt/weight/Qwen3-8B`

### Qwen3-8B baseline

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_090026_env`
- 命令：`ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"`
- 请求：`python3 05_send_requests.py --run-dir "$RUN_DIR" --case baseline_qwen3_8b --model /mnt/weight/Qwen3-8B --port 8100 --repeat 2`
- 结果：通过
- 两次请求均为 200
- 耗时：`[5.843755770009011, 2.397889809915796]`
- usage：两次均为 `prompt_tokens=1143`、`completion_tokens=64`、`total_tokens=1207`
- 输出：`same_text_as_first=[true, true]`

### Qwen3-8B Mooncake + KV Pool

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_090307_env`
- 启动 Mooncake：`./02_start_mooncake_master.sh "$RUN_DIR"`
- 启动服务：`ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B PORT=8100 ./04_start_server_kvpool_mixed.sh "$RUN_DIR"`
- 请求：`python3 05_send_requests.py --run-dir "$RUN_DIR" --case kvpool_mixed_qwen3_8b --model /mnt/weight/Qwen3-8B --port 8100 --repeat 2`
- 结果：通过
- 两次请求均为 200
- 耗时：`[2.3475408400408924, 2.232989399926737]`
- usage：两次均为 `prompt_tokens=1143`、`completion_tokens=64`、`total_tokens=1207`
- 输出：两次文本一致
- 关键日志：`Creating v1 connector with name: AscendStoreConnector`、`kv_load_failure_policy='recompute'`、`kvpool hit tokens: 1024, need to load: 1024`、`KV pool load spec created`
- 注意：Mooncake master 日志显示 metrics 端口 `9003` 被占用，但 RPC `50088` 有 `Master service started`，请求链路未受影响

### recompute / miss 轻量回退验证

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260731_090307_env`
- 配置证据：服务启动参数里有 `kv_load_failure_policy='recompute'`
- 相同 prefix 追加请求：`kvpool_recompute_after_master_stop_qwen3_8b`，两次均返回 200，耗时为 `[2.2125850699376315, 2.413814039900899]`
- 新 prefix 请求：`kvpool_recompute_new_prefix_qwen3_8b`，单次返回 200，耗时为 `2.2373886699788272`，usage 为 `prompt_tokens=1304`、`completion_tokens=64`、`total_tokens=1368`
- 结论：miss/new-prefix 场景不会导致请求失败，符合 recompute 回退的基本预期
- 限制：没有制造出明确的后端 load failure 日志，因此这不是强故障注入验证；后续如果要更严格，应单独设计能稳定触发 load failure 的场景
- 注意：`05_send_requests.py` 每次运行会重写 `summary.json`，所以该 run 的完整请求序列以 `requests.jsonl` 为准

## 11. 2026-08-03 补充验证计划：W8A8 流式 TTFT

### 资源检查

执行前检查结果：

- 16 个 Ascend 910 chip 均未显示 NPU 进程占用。
- 单 chip HBM 总量约 64GB，空闲约 61GB。
- `/vllm-workspace/vllm-ascend` 当前分支为 `pr-13160`，commit 为 `3a0404d2d7275ed3b753d4849d1b8ecdccdeaa83`。
- `/vllm-workspace/vllm` 当前为 detached HEAD，commit 为 `d02df748bf9efd99022f1a062597dc3cb3808485`。
- 发现一个历史残留 `mooncake_master` 进程占用 `50088`，因此本轮补充验证改用新的 Mooncake master 端口。
- `/mnt/weight/Qwen3-8B-W8A8` 模型目录存在，大小约 `10.50 GiB`，包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`。
- `mooncake`、`mooncake.store`、`memcache_hybrid` 均可 import。

### 新增脚本

新增 `09_send_stream_requests.py`：

- 向 `/v1/completions` 发送 `stream=true` 请求。
- 每次请求记录 `ttft_sec`、`total_sec`、SSE event 数、usage、输出文本和错误。
- 明细追加到 `stream_requests.jsonl`。
- 汇总写到 `stream_summary.json`。
- 这补齐了之前 `05_send_requests.py` 只能记录总耗时、不能记录严格 TTFT 的缺口。

新增 `10_start_server_kvpool_custom.sh`：

- 参数化 `KV_ROLE`、`KV_BACKEND`、`LOOKUP_RPC_PORT`、`LOAD_ASYNC`、`CONSUMER_IS_TO_PUT`、`CONSUMER_IS_TO_LOAD`、`USE_LAYERWISE`、`SAVE_DECODE_CACHE`、`PREFILL_TP_SIZE`、`DECODE_TP_SIZE`。
- 保存最终 `kv_transfer_config.json`。
- 保存 `kvpool_custom_env.txt`，便于复盘每次启动的关键参数。
- 默认仍等价于 Mooncake + `kv_both` + `kv_load_failure_policy=recompute`。

### 本轮验证目标

本轮只补 W8A8 模型的流式 TTFT 数据，不做大规模性能结论。

必须回答：

- baseline W8A8 流式请求是否稳定返回 200。
- KV Pool W8A8 流式请求是否稳定返回 200。
- KV Pool 日志是否仍能看到 `AscendStoreConnector`、`kv_load_failure_policy='recompute'`、hit/load/save 相关证据。
- `stream_summary.json` 是否能提供 TTFT 和 total latency 数据。
- baseline 与 KV Pool 输出在固定 prompt、`temperature=0` 下是否一致或无明显异常。

### 计划命令

baseline：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"
python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case baseline_qwen3_8b_w8a8_stream --model /mnt/weight/Qwen3-8B-W8A8 --port 8100 --repeat 4
./06_grep_logs.sh "$RUN_DIR"
./07_stop_run.sh "$RUN_DIR"
```

KV Pool：

```bash
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
MOONCAKE_MASTER_PORT=50089 ./02_start_mooncake_master.sh "$RUN_DIR"
MOONCAKE_MASTER_ADDRESS=127.0.0.1:50089 ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 KV_ROLE=kv_both KV_BACKEND=mooncake ./10_start_server_kvpool_custom.sh "$RUN_DIR"
python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case kvpool_qwen3_8b_w8a8_stream --model /mnt/weight/Qwen3-8B-W8A8 --port 8100 --repeat 4
./06_grep_logs.sh "$RUN_DIR"
./07_stop_run.sh "$RUN_DIR"
```

### 通过标准

- 两组服务健康检查 `/v1/models` 返回 200。
- `stream_summary.json` 中 `failure_count=0`。
- 每次请求都有非空 `ttft_sec` 和 `total_sec`。
- 同组内 `same_text_as_first` 全为 `true`，或只存在格式上可解释的流式差异。
- KV Pool `log_extract.txt` 可见 AscendStore/KV Pool 关键日志，且无 `traceback`、致命 `exception`、请求失败。

### 实际执行记录

#### baseline W8A8 stream

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260803_034125_env`
- 启动命令：`ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 ./03_start_server_baseline.sh "$RUN_DIR"`
- 服务 PID：`4094221`
- 健康检查：`/v1/models` 在第 13 次轮询返回 200。
- 请求命令：`python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case baseline_qwen3_8b_w8a8_stream --model /mnt/weight/Qwen3-8B-W8A8 --port 8100 --repeat 4`
- 结果：4 次请求均返回 200，均有 usage，均有 TTFT。
- 单次结果：
  - 第 1 次：`ttft_sec=0.2651`，`total_sec=2.6867`，`prompt_tokens=1143`，`completion_tokens=64`，`total_tokens=1207`
  - 第 2 次：`ttft_sec=0.0933`，`total_sec=2.5485`，usage 同上
  - 第 3 次：`ttft_sec=0.0873`，`total_sec=2.5284`，usage 同上
  - 第 4 次：`ttft_sec=0.0868`，`total_sec=2.5120`，usage 同上
- 输出文件：
  - `stream_requests.jsonl`
  - `stream_summary.json`
  - `log_extract.txt`
- 停止：`./07_stop_run.sh "$RUN_DIR"` 已停止 `server_baseline: 4094221`。

#### KV Pool custom 首次启动失败

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260803_043010_env`
- Mooncake master 启动命令：`MOONCAKE_MASTER_PORT=50089 ./02_start_mooncake_master.sh "$RUN_DIR"`
- KV Pool 启动命令：`MOONCAKE_MASTER_ADDRESS=127.0.0.1:50089 ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 KV_ROLE=kv_both KV_BACKEND=mooncake ./10_start_server_kvpool_custom.sh "$RUN_DIR"`
- 失败原因：`10_start_server_kvpool_custom.sh` 中生成 JSON 的 Python 子进程读取 `LOOKUP_RPC_PORT` 时出现 `KeyError: 'LOOKUP_RPC_PORT'`。
- 根因：脚本中 `LOOKUP_RPC_PORT`、`KV_BACKEND`、`KV_ROLE` 等是 shell 本地变量，没有 export，Python 子进程无法通过 `os.environ` 读取。
- 处理：已将这些变量改为 `export`，并执行 `bash -n 10_start_server_kvpool_custom.sh` 通过。
- 清理：执行 `./07_stop_run.sh "$RUN_DIR"`，pid 文件里的 Mooncake master 已不在运行，没有残留服务进程。

#### KV Pool custom 第二次启动失败

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260803_060014_env`
- Mooncake master 启动命令：`MOONCAKE_MASTER_PORT=50089 ./02_start_mooncake_master.sh "$RUN_DIR"`
- KV Pool 启动命令：`MOONCAKE_MASTER_ADDRESS=127.0.0.1:50089 ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 KV_ROLE=kv_both KV_BACKEND=mooncake ./10_start_server_kvpool_custom.sh "$RUN_DIR"`
- 健康检查结果：等待 10 分钟未返回 200。
- 关键日志：`mooncake_master.log` 显示 `Failed to start master admin server on port 9003`，`Address already in use`，随后 task cleanup thread stopped。
- KV Pool 服务日志根因：`RuntimeError: Initialize mooncake failed.`，API server 报 `Engine core initialization failed`。
- 判断：Mooncake master 的 RPC 端口改到了 `50089`，但 metrics/admin 端口仍默认使用 `9003`，被历史残留的 50088 master 占用，导致新 master 启动失败。
- 处理：执行 `./07_stop_run.sh "$RUN_DIR"`，确认该 run 没有残留 `mooncake_master` 或 `server_kvpool_custom` 进程。
- 下一步：复用历史残留且仍在运行的 `127.0.0.1:50088` Mooncake master，不再为本轮启动新 master。

#### KV Pool W8A8 stream：复用 50088 Mooncake master 后通过

- run dir：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script/runs/20260803_061258_env`
- 启动命令：`MOONCAKE_MASTER_ADDRESS=127.0.0.1:50088 ASCEND_RT_VISIBLE_DEVICES=8 MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 PORT=8100 KV_ROLE=kv_both KV_BACKEND=mooncake ./10_start_server_kvpool_custom.sh "$RUN_DIR"`
- 服务 PID：`4187680`
- 健康检查：`/v1/models` 在第 33 次轮询返回 200。
- 实际 KV transfer 配置文件：`kv_transfer_config.json`
- 实际启动参数记录：`kvpool_custom_env.txt`
- 请求命令：`python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case kvpool_qwen3_8b_w8a8_stream --model /mnt/weight/Qwen3-8B-W8A8 --port 8100 --repeat 4`
- 结果：4 次请求均返回 200，均有 usage，均有 TTFT。
- 单次结果：
  - 第 1 次：`ttft_sec=0.1890`，`total_sec=2.5146`，`prompt_tokens=1143`，`completion_tokens=64`，`total_tokens=1207`
  - 第 2 次：`ttft_sec=0.0921`，`total_sec=2.4334`，usage 同上
  - 第 3 次：`ttft_sec=0.0881`，`total_sec=2.4481`，usage 同上
  - 第 4 次：`ttft_sec=0.0904`，`total_sec=2.4524`，usage 同上
- `stream_summary.json` 汇总：
  - `request_count=4`
  - `success_count=4`
  - `failure_count=0`
  - `ttft_avg_sec=0.11492577742319554`
  - `ttft_p50_sec=0.09213291993364692`
  - `total_avg_sec=2.4621055024908856`
  - `same_text_as_first=[true, true, true, true]`
- 日志证据：
  - `kv_transfer_config` 中 `kv_connector='AscendStoreConnector'`、`kv_role='kv_both'`、`kv_connector_extra_config={'lookup_rpc_port': '1', 'backend': 'mooncake'}`、`kv_load_failure_policy='recompute'`
  - `Creating v1 connector with name: AscendStoreConnector`
  - `kvpool hit tokens: 1024, need to load: 1024`
  - `KV pool load spec created`
  - `External prefix cache hit rate: 67.2%`
- 停止：`./07_stop_run.sh "$RUN_DIR"` 已停止 `server_kvpool_custom: 4187680`。
- 收尾检查：没有 vLLM server 残留，NPU 无运行进程；系统上仍保留进入本轮前就存在的历史 `mooncake_master --port 50088`。

### W8A8 stream 小结

- baseline run：`runs/20260803_034125_env`
- KV Pool run：`runs/20260803_061258_env`
- 两组流式请求均 `4/4` 成功。
- 两组 usage 完全一致：`prompt_tokens=1143`、`completion_tokens=64`、`total_tokens=1207`。
- 两组同组内输出完全一致。
- baseline 平均 TTFT：`0.13312208757270128s`，平均总耗时：`2.5689232900040224s`。
- KV Pool 平均 TTFT：`0.11492577742319554s`，平均总耗时：`2.4621055024908856s`。
- 这轮数据证明 W8A8 + KV Pool mixed 流式路径可用，并补上真实 TTFT 记录；样本数只有 4，不作为严格性能结论。

