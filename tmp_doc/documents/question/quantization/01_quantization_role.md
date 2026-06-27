# 01. Quantization 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/config.py`

本问题关注：量化在 vLLM 中的边界，区分权重量化、激活量化、KV cache 量化和 kernel 支持。

---

## 1. 一句话回答

量化负责用更低 bit 或特殊数值格式表示权重、激活或 KV cache，并通过专门的参数加载和 kernel 执行来降低显存或提升吞吐。

```text
不是所有量化都作用在同一层：

权重量化：
  影响 Linear / MoE 权重和 GEMM。

激活量化：
  影响 forward 中输入 activation 的数值格式。

KV cache 量化：
  影响 attention 读取历史 K/V 的 cache dtype / layout / scale。

kernel 量化：
  影响运行时调用哪个 matmul / attention / MoE kernel。
```

---

## 2. 量化不负责什么

占位：

```text
- 不直接改变请求调度策略；
- 不直接改变 sampling 逻辑；
- 不直接改变 tokenizer / detokenizer；
- 不直接决定模型结构；
- 不一定改变 attention 语义，但会限制 backend 选择。
```

---

## 3. 量化与其他模块的关系占位

```text
config：
  决定启用哪种量化。

model loader：
  负责读入量化 checkpoint。

layer：
  创建量化参数并调用 quant_method。

kernel：
  真正执行低 bit 计算。

attention：
  处理 KV cache quantization 和 backend 支持。

parallelism：
  处理量化权重和 scale 在 TP / PP / EP 下的切分。
```

---

## 4. 一句话总结

```text
量化是配置、权重、layer、kernel 和 runtime 的协作机制，不是单一开关。
```
