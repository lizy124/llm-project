# launch scripts

## 协议级 benchmark

脚本路径:

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

运行方式:

```bash
cd /tmp/vllm-ascend-pr
PYTHONPATH=/tmp/vllm-ascend-pr:$PYTHONPATH \
python benchmarks/scripts/benchmark_zmq_lookup_payload.py \
  --case 65536:32768 \
  --iterations 1000 \
  --warmup 100
```

这个脚本会：

- 计算 `FULL` / `SUFFIX` payload bytes；
- 启动真实 `LookupKeyServer` / `LookupKeyClient`；
- 测本机 ZMQ REQ/REP RTT；
- 用 stub worker 记录收到的 hash 数和返回值。

## e2e 探针

脚本路径:

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

运行方式:

```bash
cd /tmp/vllm-ascend-pr
PYTHONPATH=/tmp/vllm-ascend-pr:$PYTHONPATH \
python benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

这个脚本会：

- 启动 `mooncake_master`；
- 启动 `vllm.entrypoints.openai.api_server`；
- 配置 `AscendStoreConnector` 和 `lookup_hash_mode=suffix`；
- 发送三次请求：
  - populate HBM；
  - 同 prefix 长 suffix；
  - 重复长 suffix；
- 在临时 `sitecustomize.py` 里 monkeypatch `KVPoolScheduler.get_num_new_matched_tokens()`，打印 `computed`、`prompt`、`block_hashes`、`mode`、`result`。

## 本次验证使用的关键参数

- `--max-model-len 2048`
- `--block-size 16`
- `VLLM_BATCH_INVARIANT=1`
- `lookup_hash_mode=suffix`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
