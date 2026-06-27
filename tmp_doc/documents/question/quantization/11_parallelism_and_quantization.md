# 11. 并行场景下量化权重如何切分？

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/distributed/`
- `code/vllm/vllm/model_executor/model_loader/`

本问题关注：tensor parallel、pipeline parallel、expert parallel 下量化权重、scale、zero point 和 packed layout 如何处理。

---

## 1. 一句话回答

并行会让量化权重加载更复杂，因为不仅 qweight 要切分，scale、zero point、group metadata 和 packed layout 也必须按同样规则对齐。

---

## 2. Tensor parallel 占位

需要梳理：

```text
ColumnParallelLinear：
  output dimension 切分。

RowParallelLinear：
  input dimension 切分。

QKVParallelLinear：
  Q/K/V fused 权重和 head 维度切分。

MergedColumnParallelLinear：
  gate_up_proj 等 fused 权重切分。
```

量化额外需要处理：

```text
- qweight shard；
- scales shard；
- zero points shard；
- group size 边界；
- packed int4/int8 layout 是否能直接切。
```

---

## 3. Pipeline parallel 占位

```text
PP 主要影响 layer 分布：

- 不同 rank 只加载自己负责的 layers；
- 每个 PP rank 上仍可能有 TP 切分；
- 量化权重只在对应 rank 加载；
- lm_head / embedding 所在 rank 要特殊处理。
```

---

## 4. Expert parallel 占位

```text
EP 影响 expert 权重：

- 每个 rank 持有部分 experts；
- per-expert scale / qweight 也按 expert 切分；
- routing 结果需要和本地 expert 分布对齐；
- fused MoE 量化 kernel 要支持这种布局。
```

---

## 5. 一句话总结

```text
并行量化的核心，是让权重切分、scale 切分和 kernel 期望的 packed layout 三者一致。
```
