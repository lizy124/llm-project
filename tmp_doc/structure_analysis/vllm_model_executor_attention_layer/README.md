# vLLM 模型执行与 Attention 层完整梳理

本目录用于系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的模型执行层与 Attention 层，包括模型注册、模型加载、权重加载、GPUModelRunner 到模型 forward、ForwardContext、Attention 层、AttentionBackend、KVCacheSpec、KV cache tensor、slot mapping、底层 CUDA/C++ kernel、量化、MoE、LoRA、多模态等内容。

## 文档导航

建议按以下顺序阅读：

1. [01_模型执行与Attention总览.md](01_模型执行与Attention总览.md)
   - 总体说明模型执行层、Attention 层在 vLLM 推理链路中的位置和职责。

2. [02_模型注册与加载链路.md](02_模型注册与加载链路.md)
   - 梳理 architecture registry、model loader、权重加载、模型实例化、模型能力接口。

3. [03_GPUModelRunner到ForwardContext.md](03_GPUModelRunner到ForwardContext.md)
   - 梳理 GPUModelRunner 如何准备 batch、slot mapping、attention metadata，并通过 ForwardContext 调用模型。

4. [04_Attention层核心实现.md](04_Attention层核心实现.md)
   - 梳理 `model_executor/layers/attention/attention.py` 中 `Attention` 类、KV cache、forward、自定义 op 注册。

5. [05_AttentionBackend选择与MetadataBuilder.md](05_AttentionBackend选择与MetadataBuilder.md)
   - 梳理 `v1/attention/backend.py`、`selector.py`、backend registry、metadata builder、impl 的关系。

6. [06_KVCacheSpec与KV_Tensor_SlotMapping.md](06_KVCacheSpec与KV_Tensor_SlotMapping.md)
   - 梳理 KVCacheSpec、AttentionSpec、KV cache tensor、cache group、slot mapping、block table 如何衔接。

7. [07_量化_MoE_LoRA_多模态.md](07_量化_MoE_LoRA_多模态.md)
   - 梳理量化、KV cache quant、MoE、LoRA、多模态与模型执行/Attention 的关系。

8. [08_CUDA_csrc_kernel调用链与调试地图.md](08_CUDA_csrc_kernel调用链与调试地图.md)
   - 梳理 Python Attention 到 custom op、torch binding、CUDA/C++ kernel 的调用链，并给出调试地图。

## 一条核心主链

```text
模型选择/加载
  vllm/model_executor/models/registry.py
  vllm/model_executor/model_loader/*
        ↓
Worker.load_model()
        ↓
GPUModelRunner.load_model()
        ↓
具体模型 forward
  vllm/model_executor/models/*
        ↓
Attention 层
  vllm/model_executor/layers/attention/attention.py
        ↓
ForwardContext 取运行时上下文
  vllm/forward_context.py
        ↓
AttentionBackend / AttentionImpl / MetadataBuilder
  vllm/v1/attention/*
        ↓
custom op / torch.ops.vllm / backend op
        ↓
CUDA/C++/CPU kernel
  csrc/*
```

## 关键代码锚点

- `code/vllm/vllm/model_executor/model_loader/__init__.py:122`：`get_model_loader()`，根据 load format 选择 loader。
- `code/vllm/vllm/model_executor/model_loader/__init__.py:130`：`get_model()`，模型加载入口。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:178`：模型层 `Attention`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:397`：Attention layer 注册到 `static_forward_context`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:438`：`Attention.forward()`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:567`：`Attention.get_kv_cache_spec()`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:649`：`get_attention_context()`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:692`：`unified_kv_cache_update()`。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:736`：`unified_attention_with_output()`。
- `code/vllm/vllm/v1/attention/backend.py:55`：`AttentionBackend` 抽象。
- `code/vllm/vllm/v1/attention/backend.py:362`：`CommonAttentionMetadata`。
- `code/vllm/vllm/v1/attention/backend.py:533`：`AttentionMetadataBuilder`。
- `code/vllm/vllm/v1/attention/backend.py:780`：`AttentionImpl`。
- `code/vllm/vllm/v1/attention/selector.py:54`：`get_attn_backend()`。
- `code/vllm/vllm/v1/kv_cache_interface.py:33`：`KVQuantMode`。
- `code/vllm/vllm/v1/kv_cache_interface.py:96`：`KVCacheSpec`。
- `code/vllm/vllm/v1/kv_cache_interface.py:160`：`AttentionSpec`。
- `code/vllm/vllm/v1/kv_cache_interface.py:204`：`FullAttentionSpec`。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`：`GPUModelRunner.execute_model()`。

## 核心结论

vLLM 的模型执行层不是简单的“加载 HF 模型然后 forward”。它把模型层和运行时调度强绑定：

- 模型层 Attention 在初始化时声明 backend、KV cache spec，并注册进静态 forward context；
- Scheduler 决定每一步 request/token/block；
- GPUModelRunner 把调度结果转换成 token tensors、slot mapping、attention metadata；
- ForwardContext 把运行时 metadata 暴露给模型层；
- Attention.forward 通过 context 找到 KV cache、metadata 和 slot mapping；
- AttentionBackend/Impl 决定实际 kernel 调用；
- csrc 执行 paged attention、cache update、量化 cache 写入等底层操作。
