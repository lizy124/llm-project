# PR 13160 验证执行记录

这份记录的目的不是只记“通过/失败”，而是把**怎么验证、怎么启动、怎么发请求、怎么判断正常、结果怎么读**都留下来，方便后续复跑和学习。

## 1. 这轮验证在验证什么

PR 目标是 `Simplify AscendStore KV pool Implementation and Unit Tests`，所以这轮只看 AscendStore KV Pool，不做泛化的全量 vLLM 回归。

本轮已经实际跑过的内容：

- 环境和导入检查
- baseline 服务验证
- Mooncake + KV Pool PD-Mixed 验证

本轮还没继续的内容：

- `/home/data/Qwen3.5-4B` 的正式模型轮次
- `kv_load_failure_policy=recompute` 的人工异常注入回退验证

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
- 结果日志和汇总文件都已落盘

当前未完成：

- `/home/data/Qwen3.5-4B` 正式模型轮次
- `recompute` 异常回退场景
- 更完整的性能对比

## 9. 相关文件和结果

- 计划：`validation_plan.md`
- 当前高层记录：`record.md`
- 环境检查 run：`runs/20260731_074619_env/`
- baseline run：`runs/20260731_075703_env/`
- KV Pool run：`runs/20260731_075952_env/`


