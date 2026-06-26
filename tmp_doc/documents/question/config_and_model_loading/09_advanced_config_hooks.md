# 09. 高级能力如何挂到配置和模型加载？

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/config/multimodal.py`
- `code/vllm/vllm/config/speculative.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/config/kv_transfer.py`
- `code/vllm/vllm/lora/`
- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/v1/spec_decode/`
- `code/vllm/vllm/compilation/`

本问题关注：LoRA、多模态、Speculative Decoding、KV Transfer、Compilation 等高级能力，如何通过配置进入模型加载或执行链路。

---

## 1. 一句话回答占位

占位：后续补充高级能力通常先进入对应 Config，再在 Engine / Worker / ModelRunner / Model layer 中按需挂接。

```text
用户参数
  → 对应子配置
  → VllmConfig
  → Engine / Scheduler / Worker / ModelRunner
  → 模型加载或执行路径中的 hook
```

---

## 2. LoRA 占位

```text
LoRAConfig
  → 是否启用 LoRA；
  → 替换 / 包装支持 LoRA 的 layer；
  → WorkerLoRAManager 管理 adapter；
  → 请求侧 LoRARequest 决定当前 batch active adapters。
```

---

## 3. Multimodal 占位

```text
MultiModalConfig
  → 决定 processor、mm limits、encoder cache；
  → ModelConfig / registry 判断模型是否多模态；
  → ModelRunner 执行 mm encoder；
  → inputs_embeds 合并进模型 forward。
```

---

## 4. Spec Decode 占位

```text
SpeculativeConfig
  → 创建 draft model / ngram / eagle proposer；
  → 影响 ModelRunner 输入准备和 logits indices；
  → 影响 KV connector finalize 时机；
  → Scheduler update 阶段处理 accepted / rejected tokens。
```

---

## 5. Compilation / CUDA graph 占位

```text
CompilationConfig
  → 控制 torch.compile；
  → 控制 CUDA graph capture sizes；
  → 控制 piecewise graph；
  → 影响 Worker warmup / capture；
  → 影响 attention backend metadata 构造。
```

---

## 6. 后续待补源码证据

占位：补充各 Config 字段、对应初始化位置、模型加载阶段和执行阶段的 hook。
