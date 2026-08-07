# PR 13354 验证流程

这份文档记录这次验证的执行顺序，方便复查和复用。

## 1. 先做环境快照

```bash
./00_env_snapshot.sh
```

目的：记录当前仓库状态、Python 包位置、NPU 和关键环境变量。后续所有 run 目录都会基于这份快照。

## 2. 先跑导入和 UT

```bash
./01_check_import_ut.sh <run_dir>
```

重点看三件事：
- 新路径是否能正常 import
- `ascend_store` 相关 UT 是否通过
- `mooncake_kv_transfer` 是否还能工作

这一步主要证明 PR 13354 的路径迁移没有把基础模块打坏。

## 3. 跑 baseline 端到端

```bash
MODEL_PATH=/mnt/weight/Qwen3-0.6B PORT=8100 ./03_start_server_baseline.sh <run_dir>
python3 ./05_send_requests.py --run-dir <run_dir> --case baseline --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./06_grep_logs.sh <run_dir>
./07_stop_run.sh <run_dir>
```

目的：证明普通推理链路没受影响。

## 4. 跑 KV Pool 端到端

```bash
./02_start_mooncake_master.sh <run_dir>
MOONCAKE_MASTER_ADDRESS=127.0.0.1:50088 \
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL_PATH=/mnt/weight/Qwen3-0.6B \
PORT=8100 \
KV_ROLE=kv_both \
KV_BACKEND=mooncake \
./10_start_server_kvpool_custom.sh <run_dir>
python3 ./05_send_requests.py --run-dir <run_dir> --case kvpool_both --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 2
./06_grep_logs.sh <run_dir>
./07_stop_run.sh <run_dir>
```

目的：证明 `attention_fence` 迁移到 `ascend_store` 后，真实 KV Pool 链路还能正常工作。

## 5. 补流式请求

```bash
python3 ./09_send_stream_requests.py --run-dir <run_dir> --case kvpool_both_stream --model /mnt/weight/Qwen3-0.6B --port 8100 --repeat 4
```

目的：确认流式输出和 TTFT 没有异常。

## 6. 看记录文件

每次 run 都会保存在：

```text
runs/<时间戳>_<case>/
```

常见文件：
- `env.txt`
- `check_import_ut.log`
- `server_baseline.log`
- `server_kvpool_custom.log`
- `mooncake_master.log`
- `requests.jsonl`
- `summary.json`
- `stream_requests.jsonl`
- `stream_summary.json`
- `log_extract.txt`

## 7. 复查标准

- UT 全部通过
- baseline 能启动并返回请求结果
- KV Pool 能启动并返回请求结果
- 流式请求成功
- 日志里没有 `ImportError`、`ModuleNotFoundError`、旧路径残留、循环依赖、fence/store 初始化异常
