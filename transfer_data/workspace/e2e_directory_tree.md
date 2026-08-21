# tests/e2e 目录结构树状图

## 完整目录树

```
tests/e2e/
│
├── doctests/                         # 文档/快速入门测试
│   ├── 001-quickstart-test.sh
│   └── 002-pip-binary-installation-test.sh
│
├── models/                           # 模型精度评测（独立于频率）
│   ├── configs/                      # 各种模型的精度 YAML
│   │   ├── Qwen3-8B.yaml
│   │   ├── Qwen3-30B-A3B.yaml
│   │   ├── Qwen3-VL-8B-Instruct.yaml
│   │   ├── Llama-3.2-3B-Instruct.yaml
│   │   ├── Mixtral-8x7B-Instruct-v0.1.yaml
│   │   └── ...（共 22 个模型）
│   ├── conftest.py
│   ├── test_lm_eval_correctness.py
│   ├── test_asr_eval_correctness.py
│   └── test_rm_eval_correctness.py
│
├── pull_request/                     # ★ PR 级：按卡数分
│   ├── one_card/                     #   单卡
│   │   ├── spec_decode/              #     投机解码
│   │   │   ├── conftest.py
│   │   │   ├── test_eagle.py
│   │   │   ├── test_ngram.py
│   │   │   ├── test_mtp_eagle_correctness.py
│   │   │   └── ...
│   │   ├── lora/                     #     LoRA
│   │   │   ├── test_qwen3_multi_loras.py
│   │   │   ├── test_llama32_lora.py
│   │   │   └── ...
│   │   ├── pooling/                  #     embedding/classification/scoring
│   │   │   ├── test_embedding.py
│   │   │   ├── test_classification.py
│   │   │   └── test_scoring.py
│   │   ├── compile/                  #     图编译
│   │   │   ├── test_graphex_norm_quant_fusion.py
│   │   │   └── test_norm_quant_fusion.py
│   │   ├── model_runner_v2/          #     新 runner
│   │   │   ├── test_basic.py
│   │   │   └── test_uva.py
│   │   ├── _310p/                    #     310P 平台
│   │   │   ├── test_dense_model_310p.py
│   │   │   ├── test_vl_model_310p.py
│   │   │   └── ...
│   │   ├── aclgraph/                 #     ACL 图模式
│   │   ├── test_qwen3_0_6b.py
│   │   ├── test_batch_invariant.py
│   │   ├── test_cpu_offloading.py
│   │   ├── test_vlm.py
│   │   └── ...
│   ├── two_card/                     #   双卡
│   │   ├── spec_decode/
│   │   │   └── test_spec_decode.py
│   │   ├── lora/
│   │   │   ├── test_llama32_lora_tp2.py
│   │   │   └── test_qwen3moe_lora_tp.py
│   │   ├── model_runner_v2/
│   │   │   └── test_data_parallel.py
│   │   ├── aclgraph/
│   │   │   └── test_aclgraph_capture_replay.py
│   │   ├── test_qwen3_30b_a3b.py
│   │   ├── test_data_parallel.py
│   │   ├── test_prefix_caching.py
│   │   ├── test_disaggregated_encoder.py
│   │   └── ...
│   └── four_card/                    #   四卡
│       ├── spec_decode/
│       │   ├── test_mtp_qwen3_next.py
│       │   └── test_mtp_step3p5.py
│       ├── context_parallel/
│       │   ├── test_accuracy.py
│       │   └── test_prefix_caching_cp.py
│       ├── _310p/
│       │   ├── test_dense_model_310p.py
│       │   ├── test_moe_model_310p.py
│       │   └── test_vl_model_310p.py
│       ├── test_deepseek_v4.py
│       ├── test_graph_mode.py
│       ├── test_pipeline_parallel.py
│       └── ...
│
├── nightly/                          # ★ 每日：按单节点/多节点分
│   ├── single_node/                  #   单节点（但可能多卡，卡数在 YAML 里配置）
│   │   ├── models/                   #     模型级测试（YAML 驱动）
│   │   │   ├── configs/
│   │   │   │   ├── DeepSeek-V3.2-W8A8.yaml
│   │   │   │   ├── DeepSeek-V4-Flash-W8A8-A3.yaml
│   │   │   │   ├── GLM-5.1-W8A8-PrefillMC2.yaml
│   │   │   │   ├── Kimi-K2.5.yaml
│   │   │   │   ├── Qwen3-235B-A22B-W8A8.yaml
│   │   │   │   ├── MiniMax-M2.5-w8a8-QuaRot-A2.yaml
│   │   │   │   ├── MiniMax-M2.5-w8a8-QuaRot-A3.yaml
│   │   │   │   ├── Qwen3.5-27B-w8a8-A2.yaml
│   │   │   │   ├── Qwen3.5-27B-w8a8-A3.yaml
│   │   │   │   ├── Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml
│   │   │   │   └── ...（共 30+ 个模型）
│   │   │   └── scripts/
│   │   │       ├── test_single_node.py
│   │   │       └── single_node_config.py
│   │   └── ops/                      #     算子级测试
│   │       ├── singlecard_ops/       #       单卡算子
│   │       │   ├── triton/           #         Triton 算子
│   │       │   │   ├── test_rope.py
│   │       │   │   ├── test_apply_penalties_triton.py
│   │       │   │   └── ...（30+ 个）
│   │       │   ├── test_compressor_metadata.py
│   │       │   ├── test_fused_moe.py
│   │       │   ├── test_kv_quant_sparse_flash_attention.py
│   │       │   └── ...（20+ 个）
│   │       ├── multicard_ops_a2/     #       多卡算子（A2 平台）
│   │       │   └── test_matmul_allreduce_add_rmsnorm.py
│   │       └── multicard_ops_a3/     #       多卡算子（A3 平台）
│   │           ├── test_dispatch_ffn_combine.py
│   │           ├── test_dispatch_ffn_combine_bf16.py
│   │           └── test_dispatch_ffn_combine_w4a8.py
│   ├── multi_node/                   #   多节点
│   │   ├── scripts/                  #     通用脚本
│   │   │   ├── run.sh
│   │   │   └── utils.py
│   │   ├── internal_dp/              #     内部 DP（disaggregated_prefill 字段）
│   │   │   ├── config/
│   │   │   │   ├── DeepSeek-V3_2-W8A8-EP.yaml          # A3 + PD 分离
│   │   │   │   ├── GLM5_1-W8A8-EP.yaml                 # A3 + PD 分离
│   │   │   │   ├── GLM5_1-W8A8-A2-dual-nodes.yaml      # A2 普通多节点
│   │   │   │   ├── Qwen3-235B-A22B-A2.yaml             # A2 普通多节点
│   │   │   │   ├── Qwen3-235B-disagg-pd.yaml           # PD 分离（无平台标记）
│   │   │   │   ├── DeepSeek-R1-W8A8-EPLB.yaml          # PD 分离（无平台标记）
│   │   │   │   └── ...（共 17 个）
│   │   │   └── scripts/
│   │   │       ├── test_multi_node.py
│   │   │       └── multi_node_config.py
│   │   └── external_dp/              #     外部 DP（routing.type: "disaggregated_prefill"）
│   │       ├── config/
│   │       │   ├── GLM5_1-W8A8-EP-external.yaml        # A3 + PD 分离
│   │       │   └── template.md
│   │       └── scripts/
│   │           ├── test_external_dp.py
│   │           └── runtime.py
│   └── 310p/                         #   310P 平台
│       └── single_node/ops/
│           └── singlecard_ops/
│               ├── test_chunk_fwd_o_310.py
│               └── test_recurrent_gated_delta_rule_v310.py
│
└── weekly/                           # ★ 每周：同样按单节点/多节点分
    ├── single_node/
    │   ├── configs/                  #   模型 YAML
    │   │   ├── Qwen3.5-122B-A10B-W8A8-A2.yaml
    │   │   ├── Qwen3.5-122B-A10B-W8A8-A3.yaml
    │   │   ├── Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml
    │   │   ├── Kimi-K2.5.yaml
    │   │   └── ...（共 17 个）
    │   ├── models/
    │   │   └── test_qwen3_30b_acc.py
    │   └── engine_func_test_robot/   #   引擎功能自动化测试
    │       └── tests/
    │           ├── test_temperature.py
    │           ├── test_top_p.py
    │           ├── test_stop.py
    │           └── ...（共 17 个）
    └── multi_node/
        ├── internal_dp/config/
        │   ├── DeepSeek-V3.yaml
        │   ├── DeepSeek-V3_2-W8A8-EP_weekly.yaml
        │   └── GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
        └── external_dp/config/
            ├── DeepSeek-V4-flash-w8a8-PD.yaml
            ├── QWEN3_235B_PD.yaml
            ├── Kimi-K2.5-W4A8-128k-1k-TPOT50.yaml
            └── ...（共 16 个）
```

## 组织方式对比

| 维度 | `pull_request` | `nightly` / `weekly` |
|---|---|---|
| **组织方式** | 按**卡数**：`one_card/` `two_card/` `four_card/` | 按**拓扑**：`single_node/` `multi_node/` |
| **原因** | PR 测试规模小、卡数固定，卡数直接决定测试能跑什么 | 规模大、卡数灵活，YAML 里配 `npu_per_node` 控制，不需要目录区分 |
| **测试入口** | 每个 `test_*.py` 是独立的 pytest 文件 | 通过 YAML 驱动，`test_single_node.py` / `test_multi_node.py` 统一入口读取配置 |
| **平台标记方式** | pytest `@pytest.mark.e2e_coverage(hardware="A2/A3")` | YAML 环境变量 `ASCEND_A3_ENABLE` 或文件名后缀 `-A2`/`-A3` |

## 关键目录说明

- **`pull_request/one_card/pooling/`**：embedding/classification/scoring 的 pooling runner 测试，**不是** KV 传输语义的"池化"
- **`nightly/multi_node/internal_dp/`**：PD 分离（disaggregated prefill）的主要测试区
- **`nightly/multi_node/external_dp/`**：外部 PD 分离（routing.type）的测试区
- **`nightly/310p/`**：310P 硬件平台专属测试
- **`weekly/engine_func_test_robot/`**：引擎功能参数自动化测试（temperature、top_p 等）