# 01. vLLM 里的算子专题应该回答什么？

源码位置：

- `vllm/vllm/_custom_ops.py`
- `vllm/vllm/model_executor/layers/`
- `vllm/vllm/attention/`
- `vllm/csrc/`

这个问题关注：vLLM 语境里的 operator / kernel 到底指什么、处在系统哪一层、和 ModelRunner / model layer / backend 的边界是什么，以及为什么需要把它作为独立专题梳理。

---

## 1. 一句话回答

vLLM 的算子层是 **把模型层抽象计算落到具体硬件 backend 执行的底层执行面**。

它回答的问题不是“请求怎么调度”，而是：

```text
这一层计算最后由哪个 kernel 跑？
它需要什么 tensor / metadata？
它为什么走 CUDA / Triton / FlashAttention / FlashInfer / torch fallback？
它如何影响吞吐、延迟、显存和 CUDA Graph？
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. operator / kernel / layer / backend 的概念边界；
2. vLLM 中算子分布在哪些目录；
3. 哪些主链路会进入算子层；
4. 算子层和已有 attention / quantization / MoE / compile 专题如何分工；
5. 阅读算子源码时应该先抓哪些入口。
```

---

## 3. 最小心智模型

```text
Scheduler 决定跑哪些 token；
ModelRunner 准备输入和 metadata；
model layer 表达数学计算；
operator / kernel 负责真正执行这段计算。
```

---

## 4. 后续补充重点

```text
- _custom_ops.py 的职责边界；
- Python wrapper 和 native extension 的关系；
- attention / quantization / MoE / norm / sampling 算子族地图；
- backend selection 与 fallback；
- debug 和 profiling 方法。
```
