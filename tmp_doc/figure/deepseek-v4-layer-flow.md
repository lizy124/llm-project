# DeepSeek-V4 单层流程图

> 说明：本图基于 `vllm-ascend` 中 `DeepseekV4Attention` / `DeepseekV2DecoderLayer` 的实现，以及 DeepSeek-V4-Pro `config.json` 中的关键配置整理。由于 Hugging Face 页面在线抓取受限，配置字段来自用户贴出的 `deepseek-ai/DeepSeek-V4-Pro/config.json` 内容，并结合本地源码分析。

## 配置要点

```text
model_type: deepseek_v4
num_hidden_layers: 61
num_nextn_predict_layers: 1
hidden_size H: 7168
hc_mult C: 4
num_attention_heads Nh: 128
num_key_value_heads Nkv: 1
head_dim Dh: 512
q_lora_rank Rq: 1536
o_lora_rank Ro: 1024
qk_rope_head_dim Dr: 64
qk_nope_head_dim Dn: 448
o_groups G: 16
sliding_window W: 128
compress_ratios: 62 项 = 31 个 c128 + 30 个 c4 + 1 个 0
MoE: 384 routed experts, 每 token 选 6 个 expert, 1 个 shared expert
mHC: hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6
```

维度记号：

```text
T: 当前 forward 中展平后的 token 数，约等于 batch 内所有 scheduled tokens 总数
H: hidden_size = 7168
C: hc_mult = 4
Rq: q_lora_rank = 1536
Ro: o_lora_rank = 1024
Nh: num_attention_heads = 128
Dh: head_dim = 512
Dr: qk_rope_head_dim = 64
Dn: head_dim - qk_rope_head_dim = 448
G: o_groups = 16
E: n_routed_experts = 384
TopK: num_experts_per_tok = 6
```

主模型 61 层压缩配置：

```text
layer 0  -> c128
layer 1  -> c128
layer 2  -> c4
layer 3  -> c128
layer 4  -> c4
...
layer 59 -> c128
layer 60 -> c4
MTP layer -> 0 / no compression
```

## Mermaid 流程图

```mermaid
flowchart TD
    %% DeepSeek-V4 single decoder layer with approximate feature dimensions

    A[输入 hidden_states<br/>约 T x C x H<br/>H=7168, C=4] --> B[保存 residual_attn<br/>约 T x C x H]

    B --> C[mHC Pre for Attention<br/>npu_hc_pre<br/>输入 T x C x H<br/>输出 T x C x H]
    C --> C1[得到 hidden_states<br/>T x C x H<br/>以及 post_attn 和 comb_attn]
    C1 --> D[Input RMSNorm<br/>T x C x H]

    D --> E[DeepseekV4Attention<br/>输入 T x C x H]

    subgraph ATTN[DeepseekV4Attention 内部]
        direction TB

        E --> Q1[Q 低秩路径 wq_a<br/>T x C x H -> T x C x Rq<br/>Rq=1536]
        Q1 --> Q2[q_norm<br/>T x C x Rq]
        Q2 --> Q3[wq_b<br/>T x C x Rq -> T x C x Nh*Dh<br/>Nh=128, Dh=512]
        Q3 --> Q4[Query<br/>逻辑形状 T x C x Nh x Dh<br/>Dh=512]

        E --> K1[KV 路径 wkv<br/>T x C x H -> T x C x Dh<br/>Dh=512]
        K1 --> K2[kv_norm<br/>T x C x Dh]
        K2 --> K3[基础 KV 或 latent KV<br/>逻辑 Nkv=1, Dh=512]

        Q4 --> R[Complex RoPE<br/>旋转维度 Dr=64<br/>非旋转维度 Dn=448]
        K3 --> R

        E --> CR{本层 compress_ratio<br/>来自 compress_ratios 的 layer_idx 项}

        CR -->|0 或 <=1| C1PATH[c1 或 no compression<br/>不创建 Compressor<br/>普通或默认 cache family]
        CR -->|4| C4PATH[c4: Compress-4-Attention<br/>约每 4 token 一组压缩状态]
        CR -->|128| C128PATH[c128: Compress-128-Attention<br/>约每 128 token 一组压缩状态]

        C4PATH --> C4A[Compressor 4<br/>wkv: H -> 2*Dh<br/>wgate: H -> 2*Dh]
        C4PATH --> C4B[Indexer<br/>index_n_heads=64<br/>index_head_dim=128<br/>index_topk=1024]
        C4B --> C4C[Indexer Cache 和 top-k 路径<br/>用于稀疏选择远端压缩块]
        C4A --> C4D[Compressor State Cache c4<br/>state_dim=2048<br/>kv_state + score_state]

        C128PATH --> C128A[Compressor 128<br/>wkv: H -> Dh<br/>wgate: H -> Dh]
        C128A --> C128B[Compressor State Cache c128<br/>state_dim=1024<br/>更粗粒度压缩状态]
        C128PATH --> C128C[无 c4 专用 Indexer<br/>直接进入压缩注意力路径]

        E --> SWA[Sliding Window Cache<br/>window W=128<br/>近期 token 细粒度窗口]

        R --> DSA[AscendDeepseekSparseAttention / DSA<br/>融合 SWA + compressed attention]
        C1PATH --> DSA
        C4C --> DSA
        C4D --> DSA
        C128B --> DSA
        SWA --> DSA

        DSA --> O1[Attention 输出<br/>逻辑按 groups 汇聚]
        O1 --> O2[输出低秩路径 wo_a -> wo_b<br/>中间约 T x C x G*Ro<br/>G=16, Ro=1024]
        O2 --> O3[attention result<br/>约 T x C x H]
    end

    E --> O3
    O3 --> F[mHC Post for Attention<br/>融合 attention result 和 residual_attn<br/>输出 T x C x H]

    F --> G[保存 residual_ffn<br/>约 T x C x H]
    G --> H[mHC Pre for FFN / MoE<br/>T x C x H -> T x C x H]
    H --> H1[得到 hidden_states<br/>T x C x H<br/>以及 post_ffn 和 comb_ffn]
    H1 --> I[Post-Attention RMSNorm<br/>T x C x H]

    I --> J[DeepseekV4MoE<br/>输入约 T x C x H]

    subgraph MOE[DeepseekV4MoE 内部]
        direction TB
        J --> M0[展平专家输入<br/>约 T*C x H]
        M0 --> M1[Router / Gate<br/>T*C x H -> T*C x E<br/>E=384]
        M1 --> M2[选择 top-k experts<br/>TopK=6]
        M2 --> M3[Routed Experts<br/>每个 expert FFN<br/>H -> moe_intermediate_size 3072 -> H]
        M0 --> M4[Shared Expert<br/>n_shared_experts=1]
        M3 --> M5[专家输出聚合<br/>约 T*C x H<br/>routed_scaling_factor=2.5]
        M4 --> M5
        M5 --> M6[恢复形状<br/>约 T x C x H]
    end

    J --> M6
    M6 --> K[mHC Post for FFN<br/>融合 MoE 输出和 residual_ffn<br/>输出 T x C x H]
    K --> L[输出本层 hidden_states, residual<br/>约 T x C x H]
```

## 压缩层选择图

```mermaid
flowchart LR
    A[当前 layer_idx] --> B[读取 compress_ratios 的 layer_idx 项]
    B --> C{compress_ratio}
    C -->|128| D[c128 层<br/>创建 Compressor 128<br/>state_dim=1024<br/>Compress-128-Attention]
    C -->|4| E[c4 层<br/>创建 Compressor 4<br/>state_dim=2048<br/>创建 Indexer<br/>Compress-4-Attention]
    C -->|0 或 <=1| F[c1 或 no compression 层<br/>不创建 Compressor]

    D --> G[进入 DSA attention]
    E --> G
    F --> G
```

## 单层结构简化版

```mermaid
flowchart TD
    X[hidden_states<br/>T x C x H] --> A1[mHC Pre Attention<br/>T x C x H]
    A1 --> A2[RMSNorm<br/>T x C x H]
    A2 --> A3[Hybrid Attention<br/>SWA W=128<br/>c4 or c128 compressor<br/>输出 T x C x H]
    A3 --> A4[mHC Post Attention<br/>T x C x H]
    A4 --> F1[mHC Pre FFN<br/>T x C x H]
    F1 --> F2[RMSNorm<br/>T x C x H]
    F2 --> F3[MoE<br/>输入 T*C x H<br/>384 routed experts, top-6<br/>输出 T*C x H]
    F3 --> F4[mHC Post FFN<br/>T x C x H]
    F4 --> Y[layer output<br/>T x C x H]
```

## 关键理解

1. DeepSeek-V4 的每一层不是同时拥有 c1/c4/c128 三套完整 attention，而是通过 `compress_ratios[layer_idx]` 决定这一层的压缩类型。
2. DeepSeek-V4-Pro 主模型 61 层里，c128 和 c4 基本交替：31 个 c128 层，30 个 c4 层。
3. c4 层会创建 `Compressor 4` 和 `Indexer`，c128 层会创建 `Compressor 128`，但没有 c4 那种专用 Indexer。
4. `sliding_window=128` 表示近期上下文还有局部窗口机制；压缩 attention 主要服务于长上下文效率。
5. `mHC` 包在 attention 和 MoE 前后，替代普通残差连接的简单加法，形成更复杂的 residual/hyper-connection 混合。
6. 图中的维度是根据配置和源码整理的逻辑维度，底层 Ascend kernel 可能会为了并行、量化、图编译和内存布局做进一步重排。

## 本地源码依据

- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:701`：`DeepseekV4Attention` 定义。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:780`：每层读取 `compress_ratio`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:805`：`compress_ratio > 1` 时创建 `Compressor`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:816`：`compress_ratio == 4` 时创建 `Indexer`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:847`：创建 `AscendDeepseekV4SWACache`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:903`：单个 decoder layer 定义。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:974`：单层 forward 流程。
