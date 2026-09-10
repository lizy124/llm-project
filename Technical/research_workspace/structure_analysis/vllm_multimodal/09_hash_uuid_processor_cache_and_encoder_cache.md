# 09 哈希、UUID、Processor Cache 与 Encoder Cache

本篇梳理 vLLM 多模态缓存体系。多模态缓存至少有两层：processor/receiver cache 缓存预处理结果，encoder cache 缓存多模态 encoder 输出。它们的 key、容量单位、生命周期都不同。

## 1. `MultiModalHasher`

文件：`code/vllm/vllm/multimodal/hasher.py`。

关键位置：

- hasher 定义与算法：`code/vllm/vllm/multimodal/hasher.py:22`
- 对象序列化：`code/vllm/vllm/multimodal/hasher.py:50`
- `hash_kwargs()`：`code/vllm/vllm/multimodal/hasher.py:154`

`MultiModalHasher` 的职责是把多模态输入稳定序列化并生成内容 hash。

支持算法来自环境变量 `VLLM_MM_HASHER_ALGORITHM`，常见包括：

- `blake3`
- `sha256`
- `sha512`

## 2. 哈希不是简单 pickle

hasher 会按对象类型定制序列化：

| 类型 | 哈希内容 |
|---|---|
| str/int/float/bytes | 直接转 bytes。 |
| PIL Image | 优先 EXIF ImageID UUID，否则 mode + 像素数组 + palette/rawmode。 |
| MediaWithBytes(Image) | 优先 EXIF UUID，否则原始 bytes。 |
| torch.Tensor | dtype + shape + data，bfloat16 有专门分支。 |
| np.ndarray | dtype + shape + data，非连续数组先拷贝。 |
| dict kwargs | key 排序后逐项 hash。 |

这保证：

- key 顺序不影响 hash；
- shape/dtype 变化会影响 hash；
- 图像 mode/palette 变化会影响 hash；
- EXIF UUID 可避免大图重复内容 hash。

测试参考：`code/vllm/tests/multimodal/test_hasher.py:19`。

## 3. UUID 的作用

UUID 有两层来源：

1. 用户显式传 `multi_modal_uuids`；
2. 图像 EXIF 中的 UUID。

UUID 解析入口：`code/vllm/vllm/multimodal/parse.py:714`。

renderer 侧 UUID 校验：`code/vllm/vllm/renderers/base.py:617`。

作用：

- 稳定标识跨请求相同媒体；
- 避免大媒体重复 hash；
- 支持 processor cache / prefix cache / encoder cache 的复用；
- 在 cache 关闭场景下也可退化为 request-local 标识。

## 4. `mm_hash` 与 `identifier`

定义在 `MultiModalFeatureSpec`：`code/vllm/vllm/multimodal/inputs.py:301`。

两个字段必须区分：

| 字段 | 语义 | 是否可能带 LoRA 语义 | 主要用途 |
|---|---|---|---|
| `mm_hash` | 多模态内容或 processor 输出身份 | 通常不带 | processor cache / receiver cache。 |
| `identifier` | encoder output 身份 | 可能带 | encoder cache。 |

为什么要区分？

同一张图在不同 LoRA 下：

- processor 输出可能完全一样，可以共享；
- encoder/tower 输出可能受 LoRA 影响，不能共享。

所以 processor cache 更偏向 `mm_hash`，encoder cache 更偏向 `identifier`。

## 5. `MultiModalCache`

基础缓存实现：`code/vllm/vllm/multimodal/cache.py:98`。

它是按字节大小计量的 LRU cache。

特点：

- tensor 按 `nbytes` 计算；
- 其他对象递归估算大小；
- 主要缓存 processor 输出或进程间传输对象；
- 容量单位是 bytes。

## 6. P0/P1 processor cache 协议

`BaseMultiModalCache` 说明了 P0 frontend / P1 core 的镜像缓存协议。

位置：`code/vllm/vllm/multimodal/cache.py:175`。

核心思想：

- P0 可以多次调用 `is_cached()`；
- P0 和 P1 的 `get_and_update()` 必须严格按顺序调用；
- 通过镜像 eviction 顺序推断 P1 是否命中；
- 尽量减少额外通信。

## 7. processor-only / sender / receiver cache

`cache.py` 中有几类缓存角色：

| 缓存 | 位置 | 职责 |
|---|---|---|
| processor-only cache | `code/vllm/vllm/multimodal/cache.py:326` | 纯前处理结果缓存。 |
| sender cache | `code/vllm/vllm/multimodal/cache.py:379` | P0 侧发送处理结果或共享内存地址。 |
| receiver cache | `code/vllm/vllm/multimodal/cache.py:589` | P1 侧接收并复用处理结果。 |
| SHM receiver cache | `code/vllm/vllm/multimodal/cache.py:678` | 通过共享内存传递大对象。 |

receiver cache 命中时返回缓存的 `MultiModalKwargsItem`，未命中时把输入 item 放入 LRU。

## 8. receiver cache key

receiver cache 中有关键逻辑：

```text
cache_key = feature.mm_hash or feature.identifier
```

位置：`code/vllm/vllm/multimodal/cache.py:589`。

这意味着：

- 如果存在 `mm_hash`，优先按内容/processor 输出复用；
- 否则退回 `identifier`。

测试 `test_processor_cache_shared_across_loras` 覆盖了 LoRA 下 processor cache 共享：`code/vllm/tests/multimodal/test_cache.py:517`。

## 9. EncoderCacheManager

scheduler 侧 encoder cache 管理器：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`。

它管理的是多模态 encoder output 的生命周期，不是 processor 结果。

关键方法：

- `check_and_update_cache()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:94`
- `can_allocate()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:123`
- `allocate()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:184`
- `free_encoder_input()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:216`
- `get_freed_mm_hashes()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:255`

容量单位是 encoder token/embed 数，而不是 bytes。

## 10. Processor cache 与 Encoder cache 对比

| 维度 | Processor/Receiver Cache | Encoder Cache |
|---|---|---|
| 缓存内容 | `MultiModalKwargsItem`、processor 输出、IPC 对象 | image/audio/video encoder output embeddings |
| key | `mm_hash` 优先 | `identifier` |
| 容量单位 | bytes | encoder tokens/embeds |
| 生命周期 | 输入处理 / P0-P1 通信 | scheduler / worker 执行阶段 |
| 是否持有 GPU tensor | 通常不是 | GPU 侧 cache 持有 tensor |
| 是否区分 LoRA | 通常不区分 | 需要区分 |

## 11. GPU encoder cache

GPU 侧真实 tensor cache：`code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py:8`。

它保存 encoder outputs，供 `_gather_mm_embeddings()` 在后续 step 中按 placeholder window 取用。

scheduler 侧 `EncoderCacheManager` 释放引用后，会通过 `free_encoder_mm_hashes` 通知 worker 删除 GPU cache 中对应 tensor。

## 12. 测试覆盖

关键测试：

- hasher：`code/vllm/tests/multimodal/test_hasher.py:19`
- processor/receiver cache：`code/vllm/tests/multimodal/test_cache.py:198`
- LoRA 下 processor cache 共享：`code/vllm/tests/multimodal/test_cache.py:517`
- SHM cache：`code/vllm/tests/multimodal/test_cache.py:336`
- encoder cache manager：`code/vllm/tests/v1/core/test_encoder_cache_manager.py:35`
- cudagraph encoder budget：`code/vllm/tests/v1/cudagraph/test_encoder_cudagraph.py:124`

## 13. 一句话总结

vLLM 把多模态缓存拆成两层：processor cache 以 `mm_hash`、字节大小、LRU 为核心，复用预处理结果；encoder cache 以 `identifier`、encoder token 数、引用生命周期为核心，复用真正的 encoder output。UUID 和内容 hash 是这两层复用的身份基础。
