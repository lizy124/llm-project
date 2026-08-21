# pull_request e2e 测试详情

## 一、拉起机制：进程内启动，无需外部脚本

测试不依赖外部启动脚本，而是通过 `vllm.LLM` 在进程内直接构造引擎：

```python
# tests/e2e/conftest.py:951
class VllmRunner:
    def __init__(self, model_name, ...):
        self.model = LLM(           # 进程内启动 vLLM 引擎
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            ...
        )
```

测试代码直接调 `model.generate()` 拿到输出，不走 HTTP server 模式。这是典型的 offline 测试。

---

## 二、每个特性一个测试函数，不混合

每个测试函数有独立的 `@pytest.mark.e2e_coverage` 标记，声明自己测什么特性：

```python
# test_batch_invariant.py:101 — 只测 batch_invariant
@pytest.mark.e2e_coverage(feature="batch_invariant", ...)

# test_qwen3_multi_loras.py:43 — 测 lora 相关的三个小特性
@pytest.mark.e2e_coverage(feature="lora,multi_lora,runtime_lora", ...)

# test_qwen3_0_6b.py:30 — 空特性，纯基础功能
@pytest.mark.e2e_coverage(feature="", ...)
```

同一个文件内可能有多个测试函数（通过 parametrize 不同参数组合），各自独立标记。

---

## 三、每个测试的模型是硬编码的

### 方式一：`@pytest.mark.e2e_model` 装饰器

```python
# test_qwen3_0_6b.py:25
@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
```

### 方式二：文件内常量/字典

```python
# test_qwen3_multi_loras.py:16
MODEL_PATH = "Qwen/Qwen3-0.6B"

# test_prefix_caching.py:12
MODELS = ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V2-Lite-Chat"]

# spec_decode/utils.py:11
MODELS = {
    "eagle3": {"main": "Qwen/Qwen3-8B", "spec": "RedHatAI/Qwen3-8B-speculator.eagle3"},
    "draft_parallel": {"main": "Meta-Llama-3.1-8B-Instruct", "spec": "amd/PARD-Llama-3.2-1B"},
    "dflash": {"main": "Qwen/Qwen3-8B", "spec": "z-lab/Qwen3-8B-DFlash-b16"},
    "dspark": {"main": "Qwen/Qwen3-8B", "spec": "deepseek-ai/dspark_qwen3_8b_block7"},
}
```

---

## 四、一卡（one_card）测试详情

| 文件 | 特性 | 模型 | 平台 |
|---|---|---|---|
| `test_qwen3_0_6b.py` | 基础功能 | `Qwen/Qwen3-0.6B` | A2 |
| `test_qwen3_5_0_8b.py` | 基础功能 | `Qwen3.5-0.8B` | - |
| `test_qwen3_8b_w8a8.py` | W8A8 量化 | `Qwen3-8B-W8A8` | - |
| `test_qwen3_embedding_0_6b.py` | embedding | `Qwen3-Embedding-0.6B` | - |
| `test_batch_invariant.py` | batch_invariant | `Qwen/Qwen3-0.6B` | A2, A3 |
| `test_attention_fa3.py` | fa3 | - | - |
| `test_cpu_offloading.py` | cpu_offloading | - | - |
| `test_cpu_weight_offload.py` | cpu_weight_offload | - | - |
| `test_simple_cpu_offload.py` | cpu_offloading | - | - |
| `test_guided_decoding.py` | guided_decoding | - | - |
| `test_multi_instance.py` | multi_instance | - | - |
| `test_vlm.py` | multimodal | - | - |
| `test_xlite.py` | xlite | - | - |
| `test_sampler.py` | logprobs | - | - |
| `test_npu_ipc_weight_transfer.py` | weight_transfer | - | - |
| `test_completion_with_prompt_embeds.py` | prompt_embeds | - | - |
| `test_multistream_overlap_shared_expert.py` | multistream_moe | - | - |
| `test_camem.py` | - | CAMEM | - |
| `test_minicpm.py` | - | MiniCPM | - |

### 子目录

| 子目录 | 测试文件 | 特性 | 模型 |
|---|---|---|---|
| `spec_decode/` | test_eagle.py | spec_decode, eagle3 | `Qwen3-8B` + EAGLE3 speculator |
| | test_ngram.py | spec_decode | - |
| | test_mtp_eagle_correctness.py | mtp | - |
| | test_dflash.py | spec_decode | `Qwen3-8B` + DFlash |
| | test_dspark.py | spec_decode | `Qwen3-8B` + DSPark |
| | test_draft_parallel.py | spec_decode | `Llama-3.1-8B` + PARD-1B |
| | test_suffix.py | spec_decode | - |
| | test_extract_hidden_states.py | - | - |
| `lora/` | test_qwen3_multi_loras.py | lora, multi_lora, runtime_lora | `Qwen/Qwen3-0.6B` (A2) |
| | test_llama32_lora.py | lora | `Llama-3.2-3B` |
| | test_ilama_lora.py | lora | - |
| | test_olmoe_lora.py | lora | `OLMoE-1B-7B` |
| | test_lora_with_spec_decode.py | lora, spec_decode | - |
| | test_qwen35_densemodel_lora.py | lora | `Qwen3.5` |
| | test_qwen3_reranker_lora.py | lora, reranker | - |
| `pooling/` | test_embedding.py | embedding | - |
| | test_classification.py | classification | - |
| | test_scoring.py | scoring | - |
| `compile/` | test_norm_quant_fusion.py | compile_fusion | - |
| | test_graphex_norm_quant_fusion.py | compile_fusion | - |
| | test_graphex_qknorm_rope_fusion.py | compile_fusion | - |
| `model_runner_v2/` | test_basic.py | - | - |
| | test_uva.py | - | - |
| `_310p/` | test_dense_model_310p.py | - | 310P 平台 |
| | test_vl_model_310p.py | multimodal | 310P 平台 |
| | test_embedding_310p.py | embedding | 310P 平台 |
| | test_classification_310p.py | classification | 310P 平台 |
| | test_scoring_310p.py | scoring | 310P 平台 |
| | test_spec_decode_mtp_310p.py | spec_decode, mtp | 310P 平台 |

---

## 五、两卡（two_card）测试详情

| 文件 | 特性 | 模型 |
|---|---|---|
| `test_qwen3_30b_a3b.py` | - | `Qwen3-30B-A3B` (A3) |
| `test_qwen3_5_35b_a3b_w8a8.py` | W8A8 | `Qwen3.5-35B-A3B` |
| `test_qwen3_6_27b_fia.py` | fia_comparison | `Qwen3.6-27B` |
| `test_qwen3_vl_30b_a3b_instruct.py` | multimodal | `Qwen3-VL-30B-A3B` |
| `test_qwen3_performance.py` | profiling | `Qwen3-30B-A3B` |
| `test_qwen3_moe_eplb.py` | eplb | `Qwen3-MoE` |
| `test_deepseek_multistream_moe.py` | multistream_moe | `DeepSeek` |
| `test_moe_routing_replay.py` | mo_routing_replay | - |
| `test_prefix_caching.py` | prefix_caching | `Qwen3-8B`, `DeepSeek-V2-Lite` |
| `test_data_parallel.py` | DP | - |
| `test_disaggregated_encoder.py` | - | - |
| `test_flashcomm_distributed.py` | flashcomm1 | - |
| `test_external_launcher.py` | - | - |
| `test_sequence_parallelism_moe.py` | SP | - |
| `test_shared_expert_dp.py` | DP | - |
| `test_sp_pass.py` | SP | - |
| `test_hccl_weight_transfer.py` | weight_transfer | - |
| `test_offline_weight_load.py` | - | - |
| `test_gpt_oss_distributed.py` | - | GPT-OSS |

### 子目录

| 子目录 | 测试文件 | 特性 | 模型 |
|---|---|---|---|
| `spec_decode/` | test_spec_decode.py | spec_decode | - |
| `lora/` | test_llama32_lora_tp2.py | lora | `Llama-3.2-3B` |
| | test_ilama_lora_tp2.py | lora | - |
| | test_qwen3moe_lora_tp.py | lora | `Qwen3-MoE` |
| `aclgraph/` | test_aclgraph_capture_replay.py | aclgraph | - |
| `model_runner_v2/` | test_data_parallel.py | DP | - |

---

## 六、四卡（four_card）测试详情

| 文件 | 特性 | 模型 |
|---|---|---|
| `test_deepseek_v4.py` | - | `DeepSeek-V4` (A3) |
| `test_deepseek_v3_2_w8a8_pruning.py` | W8A8 | `DeepSeek-V3.2` |
| `test_qwen3_5.py` | - | `Qwen3.5` |
| `test_qwen3_next.py` | - | `Qwen3-Next` |
| `test_graph_mode.py` | graph_mode | - |
| `test_pipeline_parallel.py` | PP | - |
| `test_data_parallel_tp2.py` | DP, TP | - |
| `test_profiling_chunk_performance.py` | profiling | - |

### 子目录

| 子目录 | 测试文件 | 特性 | 模型 |
|---|---|---|---|
| `spec_decode/` | test_mtp_qwen3_next.py | mtp | `Qwen3-Next` |
| | test_mtp_step3p5.py | mtp | `Step3.5` |
| `context_parallel/` | test_accuracy.py | CP | - |
| | test_prefix_caching_cp.py | prefix_caching, CP | - |
| `_310p/` | test_dense_model_310p.py | - | 310P 平台 |
| | test_moe_model_310p.py | moe | 310P 平台 |
| | test_vl_model_310p.py | multimodal | 310P 平台 |

---

## 七、总结

| 维度 | 详情 |
|---|---|
| **拉起方式** | 进程内 `vllm.LLM()`，不走 HTTP server |
| **模型指定** | 每个测试文件硬编码，通过 `@pytest.mark.e2e_model` 或文件常量 |
| **特性隔离** | 一个测试函数 = 一个 `e2e_coverage` 标记 = 一组特性 |
| **一卡主力模型** | `Qwen3-0.6B`、`Qwen3-8B` |
| **两卡主力模型** | `Qwen3-30B-A3B`、`Qwen3-8B`、`DeepSeek-V2-Lite` |
| **四卡主力模型** | `DeepSeek-V4`、`DeepSeek-V3.2`、`Qwen3-Next` |
| **平台分布** | 一卡：A2 为主；两卡：A3 为主；四卡：A3 为主 |