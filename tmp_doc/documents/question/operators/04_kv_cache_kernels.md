# 04. KV Cache 相关算子负责什么？

源码位置：

- `vllm/vllm/attention/`
- `vllm/vllm/worker/`
- `vllm/vllm/_custom_ops.py`
- `vllm/csrc/cache/`

这个问题关注：KV cache 写入、复制、reshape、swap、block 操作、slot mapping、KV cache quantization 等底层算子如何把逻辑 token 位置映射到物理 cache 张量。

---

## 1. 一句话回答

KV cache 算子负责把模型产生的 K/V 状态写入、搬移或读取到正确的 cache block 位置。

最小链路是：

```text
Scheduler 分配 block
  → ModelRunner 准备 slot mapping / block table
  → model forward 产生 K/V
  → KV cache kernel 写入或读取 cache
  → attention kernel 使用 cache
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. KV cache tensor layout；
2. slot mapping 如何定位写入位置；
3. block table 如何支持 paged attention 读取；
4. copy / swap / reshape / gather 等 cache 操作；
5. KV cache quantization 如何改变读写路径；
6. prefix cache / external KV / connector 场景下的 cache 操作；
7. 常见 shape mismatch 和 illegal memory access 如何排查。
```

---

## 3. 需要串起来的主线

```text
KVCacheManager / Scheduler
  → block allocation
  → SchedulerOutput
  → ModelRunner slot mapping
  → cache write / attention read kernels
```

---

## 4. 后续补充重点

```text
- block size、num layers、num heads、head dim；
- cache dtype 与 quantized cache；
- cache block 生命周期；
- prefix hit 和 recompute；
- CUDA Graph 下 cache 地址稳定性。
```
