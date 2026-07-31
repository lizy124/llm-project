# PR 13160 验证脚本

脚本目录：`/home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script`

## 推荐执行顺序

先同步 vLLM 到 PR 声明的 verified commit：

```bash
cd /home/lizhongyang/refactor/llm-project/transfer_data/refactor/kvpool/test_script
./08_sync_vllm_verified_commit.sh
```

如果 GitHub 网络不可用，可先跳过该步，但 UT 可能因为 vLLM API/目录不匹配失败。

创建 run 目录并记录环境：

```bash
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
echo ${RUN_DIR}
```

导入检查和 UT：

```bash
./01_check_import_ut.sh ${RUN_DIR}
```

baseline 服务：

```bash
MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./03_start_server_baseline.sh ${RUN_DIR}
python3 05_send_requests.py --run-dir ${RUN_DIR} --case baseline --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./06_grep_logs.sh ${RUN_DIR}
./07_stop_run.sh ${RUN_DIR}
```

KV Pool PD-Mixed：

```bash
RUN_DIR=$(./00_env_snapshot.sh | awk -F= '/RUN_DIR=/{print $2}')
./02_start_mooncake_master.sh ${RUN_DIR}
MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./04_start_server_kvpool_mixed.sh ${RUN_DIR}
python3 05_send_requests.py --run-dir ${RUN_DIR} --case kvpool_mixed --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./06_grep_logs.sh ${RUN_DIR}
./07_stop_run.sh ${RUN_DIR}
```

正式模型可替换为：

```bash
MODEL_PATH=/home/data/Qwen3.5-4B
```

## 结果文件

每次 run 保存到：`runs/YYYYMMDD_HHMMSS_<case>/`

关键文件：

- `env.txt`：环境、commit、包路径、NPU 信息。
- `check_import_ut.log`：导入检查、py_compile、pytest 结果。
- `server_baseline.log`：baseline 服务日志。
- `server_kvpool_mixed.log`：KV Pool 服务日志。
- `mooncake_master.log`：Mooncake master 日志。
- `requests.jsonl`：请求与响应明细。
- `summary.json`：请求汇总。
- `log_extract.txt`：关键日志摘取。
- `conclusion.md`：人工结论。

## 注意

PR 分支声明的 vLLM verified commit 是：

`d02df748bf9efd99022f1a062597dc3cb3808485`

正式验证前建议切 `/vllm-workspace/vllm` 到该 commit。
