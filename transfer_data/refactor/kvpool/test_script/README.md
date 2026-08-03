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

## 补充验证脚本

流式 TTFT 请求：

```bash
python3 09_send_stream_requests.py --run-dir ${RUN_DIR} --case baseline_stream --model /mnt/weight/Qwen3-8B-W8A8 --port 8100 --repeat 4
```

输出文件：

- `stream_requests.jsonl`：每次流式请求的 TTFT、总耗时、usage、输出文本和错误信息。
- `stream_summary.json`：TTFT/总耗时汇总、成功失败数量、输出一致性。

参数化 KV Pool 启动：

```bash
MOONCAKE_MASTER_ADDRESS=127.0.0.1:50089 \
ASCEND_RT_VISIBLE_DEVICES=8 \
MODEL_PATH=/mnt/weight/Qwen3-8B-W8A8 \
PORT=8100 \
KV_ROLE=kv_both \
KV_BACKEND=mooncake \
./10_start_server_kvpool_custom.sh ${RUN_DIR}
```

常用可调环境变量：

- `KV_ROLE`：`kv_both`、`kv_producer`、`kv_consumer`。
- `KV_BACKEND`：默认 `mooncake`，可用于后续尝试 `memcache`。
- `LOAD_ASYNC`、`CONSUMER_IS_TO_PUT`、`CONSUMER_IS_TO_LOAD`、`USE_LAYERWISE`、`SAVE_DECODE_CACHE`：设为 `true`/`1` 时写入 `kv_connector_extra_config`。
- `LOOKUP_RPC_PORT`：默认 `1`。
- `EXTRA_CONFIG_JSON`：额外 JSON 覆盖，例如 `'{"consumer_is_to_put":true}'`。

脚本会保存：

- `kv_transfer_config.json`：最终传给 vLLM 的 KV transfer 配置。
- `kvpool_custom_env.txt`：本次启动的关键环境变量。
- `server_kvpool_custom.log`：服务日志。

## 注意

PR 分支声明的 vLLM verified commit 是：

`d02df748bf9efd99022f1a062597dc3cb3808485`

正式验证前建议切 `/vllm-workspace/vllm` 到该 commit。

本机曾残留一个旧的 `mooncake_master` 占用 `50088`，补充验证建议用 `MOONCAKE_MASTER_PORT=50089`、`50090` 等新端口。
