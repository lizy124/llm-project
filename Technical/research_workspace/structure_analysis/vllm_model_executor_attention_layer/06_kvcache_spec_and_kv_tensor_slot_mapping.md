# 06 KVCacheSpec 与 KV Tensor、SlotMapping

本篇梳理模型层如何声明 KV cache 需求，运行时如何分配 KV cache tensor，Scheduler/GPUModelRunner 如何通过 block table 和 slot mapping 把 token 映射到具体 KV cache 位置。

## 1. 总体链路

```text
Attention layer
  ↓ get_kv_cache_spec()
KVCacheSpec / AttentionSpec
  ↓ EngineCore 收集与合并
KVCacheConfig / KVCacheGroupSpec
  ↓ Worker/GPUModelRunner 分配 KV cache tensor
KV cache tensor
  ↑
slot mapping / block table
  ↑
SchedulerOutput 中的 block ids
  ↑
Scheduler + KVCacheManager 分配逻辑 block
```

## 2. KVCacheSpec

基础类在 `code/vllm/vllm/v1/kv_cache_interface.py:96`。

它表示某类 cache 的规格，提供：

- `page_size_bytes()`：每页/block 占多少字节；
- `storage_block_size()`：实际存储 block size；
- `max_memory_usage_bytes()`：最大内存占用估算；
- `copy_with_new_block_size()`：复制并替换 block size；
- `merge()`：同类 spec 合并；
- `is_uniform_with_collection()`：判断是否可归为 uniform group。

## 3. KVQuantMode

`KVQuantMode` 定义在 `code/vllm/vllm/v1/kv_cache_interface.py:33`。

常见模式：

- `NONE`：不量化；
- `FP8_PER_TENSOR`：FP8 per tensor；
- `INT8_PER_TOKEN_HEAD`：INT8 per token/head；
- `FP8_PER_TOKEN_HEAD`：FP8 per token/head；
- `NVFP4`：NVFP4 cache。

辅助函数：

- `get_kv_quant_mode()`；
- `is_quantized_kv_cache()`；
- `kv_cache_uses_per_token_head_scales()`。

KV quant mode 会影响：

- KV cache dtype；
- page size；
- scale tensor；
- kernel 选择；
- cache update 方式。

## 4. AttentionSpec

`AttentionSpec` 定义在 `code/vllm/vllm/v1/kv_cache_interface.py:160`。

它是 attention 层 KV cache spec 的基类，包含：

- block size；
- num kv heads；
- head size；
- dtype；
- kv quant mode；
- attention type；
- non-causal 标记；
- per-layer 特殊配置。

它的 `page_size_bytes()` 会根据 head 数、head size、dtype、block size、量化模式计算一页 KV cache 占用。

## 5. 常见 Attention KVCacheSpec 类型

| 类型 | 位置 | 含义 |
|---|---|---|
| `FullAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:204` | 普通 full causal attention |
| `TQFullAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:339` | TurboQuant-aware page size |
| `MLAAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:365` | MLA attention |
| `HiddenStateCacheSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:428` | hidden state cache 变体 |
| `ChunkedLocalAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:435` | chunked local attention |
| `SlidingWindowSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:472` | sliding window attention |
| `SlidingWindowMLASpec` | `code/vllm/vllm/v1/kv_cache_interface.py:544` | sliding window MLA |
| `MambaSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:620` | Mamba/SSM state cache |
| `EncoderOnlyAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:661` | encoder-only attention |
| `CrossAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:668` | encoder-decoder cross attention |
| `SinkFullAttentionSpec` | `code/vllm/vllm/v1/kv_cache_interface.py:681` | attention sink |

## 6. Attention.get_kv_cache_spec

模型层 `Attention` 通过：

```text
Attention.get_kv_cache_spec(vllm_config)
```

声明自己的 cache 需求。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:567`。

它会根据：

- attention type；
- sliding window；
- head size；
- num kv heads；
- KV cache dtype；
- backend layout；
- MLA/sink/prefix-lm；
- quant mode；
- block size；
- non-causal；

生成对应 spec。

## 7. spec 合并与 cache group

多个 layer 的 KVCacheSpec 会被收集后合并/分组。

相关结构：

- `KVCacheTensor`：`code/vllm/vllm/v1/kv_cache_interface.py:843`
- `KVCacheGroupSpec`：`code/vllm/vllm/v1/kv_cache_interface.py:853`
- `KVCacheConfig`：`code/vllm/vllm/v1/kv_cache_interface.py:868`

`KVCacheGroupSpec` 描述一个 cache group。

`KVCacheConfig` 描述最终运行时 cache 配置，包括：

- cache groups；
- num blocks；
- tensor specs；
- 是否有 Mamba layers；
- 是否需要 zeroing。

## 8. KV cache tensor 分配

worker 侧分配入口：

```text
Worker.initialize_from_config()
  ↓
GPUModelRunner.initialize_kv_cache(kv_cache_config)
```

代码位置：

- `Worker.initialize_from_config()`：`code/vllm/vllm/v1/worker/gpu_worker.py:562`
- `GPUModelRunner.initialize_kv_cache()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

GPUModelRunner 还会使用：

- `_allocate_kv_cache_tensors()`：分配底层 tensor；
- `_reshape_kv_cache_tensors()`：按 backend layout reshape；
- `_attn_group_iterator()`：遍历 attention group；
- `initialize_attn_backend()`：初始化 backend；
- `initialize_metadata_builders()`：初始化 metadata builder。

## 9. block id 与 physical tensor 的关系

Scheduler 和 KVCacheManager 管的是逻辑 block 分配：

```text
request -> block ids
```

KV cache tensor 是一大块预分配显存：

```text
[num_blocks, ... cache layout ...]
```

block id 就是访问这个大 tensor 的 page/block 索引。

但是具体 token 写到哪里，还需要 slot mapping。

## 10. slot mapping

slot mapping 由 GPUModelRunner 生成。

方法：

```text
GPUModelRunner._get_slot_mappings()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`。

调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`。

它建立：

```text
token index in current batch -> KV cache slot
```

如果用二维理解：

```text
slot = physical_block_id * block_size + offset_in_block
```

实际实现还要考虑：

- 多 cache group；
- padding；
- CUDA graph capture size；
- ubatching；
- separate KV cache update backend；
- Mamba/hybrid layout；
- sliding window；
- spec decode lookahead。

## 11. block table

block table 告诉 attention kernel：某个 request 的逻辑第几个 block 对应哪个物理 block id。

路径：

```text
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu/block_table.py
```

attention kernel 在计算历史 attention 时，需要根据 block table 找到历史 K/V 所在的 pages。

## 12. ForwardContext 如何拿到 KV cache

在 forward 前，GPUModelRunner 调：

```text
set_forward_context(..., slot_mapping=slot_mappings, attn_metadata=...)
```

Attention.forward 内调用：

```text
get_attention_context(layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`。

它会返回：

- 当前 layer；
- 当前 layer 的 KV cache；
- 当前 layer 的 metadata；
- 当前 layer 的 slot mapping。

## 13. KV cache update 与 attention read

### 写入

当前 step 新 token 的 key/value 通过 slot mapping 写入 KV cache：

```text
unified_kv_cache_update
  -> backend.do_rope_and_kv_cache_update
  -> cache kernel
```

### 读取

attention forward 通过 metadata/block table 读取历史 K/V：

```text
unified_attention_with_output
  -> self.impl.forward
  -> paged attention kernel
```

## 14. spec 对内存估算的影响

`page_size_bytes()` 会直接影响 KV cache 可容纳 block 数。

例如 full attention 的 page size 主要由以下决定：

```text
block_size
num_kv_heads
head_size
key + value 两份 cache
dtype bytes
quant scale overhead
```

MLA、NVFP4、sliding window、Mamba 等会改变计算方式。

## 15. 一句话总结

`KVCacheSpec` 是模型层对 cache 需求的声明，`KVCacheConfig` 是运行时合并后的全局配置，KV cache tensor 是 worker 侧实际分配的显存，block table 和 slot mapping 则把 scheduler 的逻辑 block 分配映射到 attention kernel 可读写的物理 KV cache slot。
