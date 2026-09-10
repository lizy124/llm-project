# DeepSeek-V4 第 10 层详细流程图

> 第 10 层按 0 起始编号，即 `layer_idx = 10`。根据 DeepSeek-V4-Pro 的 `compress_ratios` 配置，第 10 层的 `compress_ratio = 4`，所以它是一个 **c4 / Compress-4-Attention 层**，会创建 `Compressor 4` 和 `Indexer`。

## 第 10 层配置

```text
layer_idx: 10
compress_ratio: 4
cache family: c4
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
index_n_heads: 64
index_head_dim: 128
index_topk: 1024
MoE routed experts E: 384
num_experts_per_tok TopK: 6
n_shared_experts: 1
moe_intermediate_size: 3072
```

维度记号：

```text
T: 当前 forward 中的 token 数
C: hc_mult = 4
H: hidden_size = 7168
Rq: q_lora_rank = 1536
Ro: o_lora_rank = 1024
Nh: attention heads = 128
Dh: head_dim = 512
Dr: RoPE dim = 64
Dn: no-RoPE dim = 448
G: output groups = 16
E: routed experts = 384
TopK: 每 token 选择 6 个 expert
```

## Mermaid 流程图

```mermaid
flowchart TD
    %% DeepSeek-V4 layer 10, c4 attention layer

    A[Layer 10 输入 hidden_states<br/>T x C x H<br/>C=4, H=7168] --> B[保存 attention residual<br/>residual_attn = hidden_states<br/>T x C x H]

    B --> C[mHC Pre Attention<br/>npu_hc_pre<br/>输入 T x C x H]
    C --> C1[输出混合 hidden_states<br/>T x C x H<br/>同时产生 post_attn 和 comb_attn]
    C1 --> D[Input RMSNorm<br/>T x C x H]

    D --> ATTN_IN[进入 DeepseekV4Attention<br/>layer_idx=10<br/>compress_ratio=4]

    subgraph ATTN[第 10 层 Attention 详细流程]
        direction TB

        ATTN_IN --> Q1[Q 路径 wq_a<br/>T x C x H -> T x C x Rq<br/>Rq=1536]
        Q1 --> Q2[q_norm<br/>T x C x Rq]
        Q2 --> Q3[wq_b<br/>T x C x Rq -> T x C x Nh*Dh<br/>Nh=128, Dh=512]
        Q3 --> Q4[Query 张量<br/>逻辑 T x C x Nh x Dh]

        ATTN_IN --> KV1[KV 路径 wkv<br/>T x C x H -> T x C x Dh<br/>Dh=512]
        KV1 --> KV2[kv_norm<br/>T x C x Dh]
        KV2 --> KV3[基础 latent KV 表示<br/>逻辑 Nkv=1, Dh=512]

        Q4 --> R1[Complex RoPE<br/>第 10 层是 c4<br/>使用 compress_rope_theta]
        KV3 --> R1
        R1 --> R2[RoPE 后 Q/K 表示<br/>Dr=64 旋转维度<br/>Dn=448 非旋转维度]

        ATTN_IN --> SWA[Sliding Window Cache<br/>W=128<br/>保留近期上下文细粒度访问]

        ATTN_IN --> COMP[Compressor 4<br/>Compress-4-Attention]
        COMP --> COMP1[wkv in compressor<br/>T x C x H -> T x C x 2*Dh<br/>因为 c4 overlap=true]
        COMP --> COMP2[wgate in compressor<br/>T x C x H -> T x C x 2*Dh]
        COMP1 --> COMP3[压缩 KV 状态<br/>约每 4 token 形成压缩单元]
        COMP2 --> COMP3
        COMP3 --> COMP4[Compressor State Cache c4<br/>state_dim=2048<br/>kv_state + score_state]

        ATTN_IN --> IDX[Indexer for c4<br/>c128 层没有这个专用 Indexer]
        IDX --> IDX1[wq_b in indexer<br/>q_lora_rank -> index_n_heads*index_head_dim]
        IDX --> IDX2[weights_proj<br/>H -> index_n_heads<br/>index_n_heads=64]
        IDX1 --> IDX3[Indexer K Cache<br/>index_head_dim=128]
        IDX2 --> IDX4[top-k 选择<br/>index_topk=1024]
        IDX3 --> IDX4

        R2 --> DSA[AscendDeepseekSparseAttention<br/>融合 local window 和 c4 compressed attention]
        SWA --> DSA
        COMP4 --> DSA
        IDX4 --> DSA

        DSA --> O1[Attention 聚合输出<br/>逻辑仍回到 hidden 通道]
        O1 --> O2[输出投影 wo_a<br/>约 T x C x Nh*Dh/G -> T x C x G*Ro<br/>G=16, Ro=1024]
        O2 --> O3[输出投影 wo_b<br/>T x C x G*Ro -> T x C x H]
        O3 --> ATTN_OUT[Attention 输出<br/>T x C x H]
    end

    ATTN_IN --> ATTN_OUT
    ATTN_OUT --> E[mHC Post Attention<br/>融合 attention 输出和 residual_attn<br/>输出 T x C x H]

    E --> F[保存 FFN residual<br/>residual_ffn = hidden_states<br/>T x C x H]
    F --> G[mHC Pre FFN<br/>npu_hc_pre<br/>T x C x H]
    G --> G1[输出混合 hidden_states<br/>T x C x H<br/>同时产生 post_ffn 和 comb_ffn]
    G1 --> H[Post Attention RMSNorm<br/>T x C x H]

    H --> MOE_IN[进入 DeepseekV4MoE<br/>输入 T x C x H]

    subgraph MOE[第 10 层 MoE 详细流程]
        direction TB

        MOE_IN --> M0[展平 token 和 hc 维度<br/>T x C x H -> T*C x H]
        M0 --> M1[Router Gate<br/>T*C x H -> T*C x E<br/>E=384]
        M1 --> M2[选择 TopK experts<br/>TopK=6]
        M2 --> M3[Routed Expert FFN<br/>每个 expert: H -> 3072 -> H]
        M0 --> M4[Shared Expert<br/>n_shared_experts=1<br/>H -> 3072 -> H]
        M3 --> M5[按 gate 权重聚合 routed experts<br/>T*C x H]
        M4 --> M5
        M5 --> M6[乘 routed_scaling_factor=2.5<br/>并恢复形状 T x C x H]
    end

    MOE_IN --> M6
    M6 --> N[mHC Post FFN<br/>融合 MoE 输出和 residual_ffn<br/>输出 T x C x H]
    N --> Z[Layer 10 输出<br/>hidden_states, residual<br/>T x C x H]
```

## 第 10 层的关键点

1. 第 10 层是 `c4` 层，不是 `c128` 层。
2. 因为 `compress_ratio = 4`，所以这一层会创建：
   - `Compressor 4`
   - `Indexer`
   - `Compressor State Cache c4`
3. c4 层的 Indexer 用于辅助稀疏选择远端压缩块；c128 层没有这个 c4 专用 Indexer。
4. 近期上下文仍有 `sliding_window = 128` 的局部窗口路径。
5. 第 10 层整体仍然是 decoder layer 结构：

```text
mHC Pre Attention
-> RMSNorm
-> c4 Hybrid Attention
-> mHC Post Attention
-> mHC Pre FFN
-> RMSNorm
-> MoE
-> mHC Post FFN
```

## 本地源码依据

- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:701`：`DeepseekV4Attention` 定义。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:780`：每层读取 `compress_ratio`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:805`：`compress_ratio > 1` 时创建 `Compressor`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:816`：`compress_ratio == 4` 时创建 `Indexer`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:847`：创建 `AscendDeepseekV4SWACache`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:903`：单个 decoder layer 定义。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:974`：单层 forward 流程。
