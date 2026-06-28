# 11. 算子如何适配 TP / PP / DP / EP？

源码位置：

- `vllm/vllm/distributed/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/attention/`

这个问题关注：tensor parallel、pipeline parallel、data parallel、expert parallel、context parallel 如何改变算子的输入输出布局、通信边界和 backend 选择。

---

## 1. 一句话回答

并行策略会改变算子的局部输入规模和通信方式，因此同一个 layer 在不同 parallel config 下可能对应不同的算子组合。

最小链路是：

```text
ParallelConfig
  → model layer partition
  → rank-local operator execution
  → collective communication
  → next operator / next stage
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. TP 如何影响 column / row parallel linear；
2. attention head 切分如何影响 attention kernel；
3. PP 如何改变 forward 边界和 intermediate tensors；
4. DP 如何复制模型并影响 batch 分配；
5. EP 如何改变 MoE dispatch、expert GEMM、combine；
6. CP 如何影响 attention / KV cache 相关算子；
7. 通信算子和计算算子如何交错优化。
```

---

## 3. 需要串起来的主线

```text
parallel topology
  → layer partition
  → local compute kernel
  → collective op
  → global semantic output
```

---

## 4. 后续补充重点

```text
- all-reduce / all-gather / reduce-scatter；
- vocab parallel logits；
- expert parallel routing；
- attention head partition；
- CUDA Graph 下通信与计算组合。
```
