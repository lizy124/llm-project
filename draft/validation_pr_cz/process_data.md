# process data

## 1. 协议级 benchmark 数据

### 64K tokens / 50% HBM hit

```text
tokens=65536 block_size=16 hbm_hit_tokens=32768 full_hashes=4096 suffix_hashes=2048
  server_received: full_hashes=4096 suffix_hashes=2048 full_hit_tokens=65536 suffix_hit_tokens=65536
  payload_bytes: full=270354 suffix=135188 saved=135166 saved_pct=50.00
  rtt_us_full: mean=758.40 p50=765.17 p95=787.38 p99=822.95
  rtt_us_suffix: mean=427.94 p50=421.87 p95=446.56 p99=458.84
```

### 计算结果

- payload: 270354 -> 135188 bytes
- payload reduction: 135166 bytes, 50.00%
- RTT mean: 758.40 -> 427.94 us
- RTT reduction: 330.46 us, 43.55%

## 2. e2e 请求数据

### 三次请求

- `populate_hbm`: 496 prompt tokens
- `same_prefix_long_suffix`: 977 prompt tokens
- `repeat_long_suffix`: 977 prompt tokens

### e2e 观测结果

- 第二次请求在日志里出现 `computed=384`
- 真实 lookup 里出现 `hit_tokens=384`
- 长 suffix 场景里 `mode=suffix`

### 请求耗时

```text
RESULT populate_hbm elapsed_s=0.272 prompt_tokens=496 text=' 1. '
RESULT same_prefix_long_suffix elapsed_s=1.682 prompt_tokens=977 text=' epsilon zeta eta'
RESULT repeat_long_suffix elapsed_s=1.629 prompt_tokens=977 text=' epsilon zeta eta'
```

### 端到端收益

- 1.682 s -> 1.629 s
- 收益约 0.053 s
- 约 3.2%

## 3. 关键验证点

- `lookup_hash_mode=suffix` 已进入真实 lookup 请求
- `token_len=896` 的长 suffix 场景里，worker 返回 `hit_tokens=384`
- `computed=384` 表明 HBM prefix 命中已发生
