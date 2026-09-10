# key logs

## 协议级 benchmark

```text
tokens=65536 block_size=16 hbm_hit_tokens=32768 full_hashes=4096 suffix_hashes=2048
  server_received: full_hashes=4096 suffix_hashes=2048 full_hit_tokens=65536 suffix_hit_tokens=65536
  payload_bytes: full=270354 suffix=135188 saved=135166 saved_pct=50.00
  rtt_us_full: mean=758.40 p50=765.17 p95=787.38 p99=822.95
  rtt_us_suffix: mean=427.94 p50=421.87 p95=446.56 p99=458.84
```

## e2e 关键日志

### 短 suffix 场景

```text
PATCHED_LOOKUP req=cmpl-b5db3712206d4adc-0-97ad8e90 computed=0 prompt=496 block_hashes=3 mode=LookupHashMode.SUFFIX result=0
```

### 真实 HBM prefix 命中

```text
PATCHED_LOOKUP req=cmpl-9ef715a3ed1b0aae-0-b0861c45 computed=384 prompt=977 block_hashes=7 mode=LookupHashMode.SUFFIX result=0
```

### 长 suffix 场景的 lookup

```text
KV pool lookup request mode=suffix token_len=896 group=0 keys=4 multi_tp_keys=4 exists_count=0/4 exists_sample=[0, 0, 0, 0]
KV pool lookup response token_len=896 groups=[0] hit_tokens=384
```

## e2e 输出

```text
RESULT populate_hbm elapsed_s=0.272 prompt_tokens=496 text=' 1. '
RESULT same_prefix_long_suffix elapsed_s=1.682 prompt_tokens=977 text=' epsilon zeta eta'
RESULT repeat_long_suffix elapsed_s=1.629 prompt_tokens=977 text=' epsilon zeta eta'
```
