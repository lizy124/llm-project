# DeepSeek-V4 的 Compressed Sparse Attention（CSA）详解

本文基于 `D:\lzy\project\kv_pool\DeepSeek_V4.pdf` 中第 2.3 节，尤其是第 2.3.1 节 **Compressed Sparse Attention** 整理。

先说明一个关键点：DeepSeek-V4 PDF 里确实明确提出了 **Compressed Sparse Attention, CSA**。它不是泛泛意义上的“长上下文稀疏注意力”，而是 DeepSeek-V4 hybrid attention 架构中的一个具体组件。

DeepSeek-V4 的 CSA 可以概括为：

> 先把 KV cache 沿序列维度压缩，把每 `τ` 个 token 的 KV cache 压成一个 compressed KV entry；然后用 Lightning Indexer 在 compressed KV entries 上做 top-k 稀疏选择；最后核心 attention 只访问这些 selected compressed KV entries，同时额外拼接 sliding window 中最近 token 的未压缩 KV，以保留局部细节。

---

## 1. DeepSeek-V4 为什么要引入 CSA

随着上下文长度扩展到极长规模，尤其是 DeepSeek-V4 支持 **1M token context**，attention 会成为主要计算瓶颈。

普通 dense attention 中，每个 query token 都需要 attend 到大量历史 KV：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

如果序列长度是 `L`，标准 attention 的训练复杂度大致是：

$$
O(L^2)
$$

推理时，单个新 token 也要和历史全部 KV cache 交互，单 token attention 成本大致随上下文长度线性增长：

$$
O(L)
$$

当 `L = 1,000,000` 时，这个成本非常高。

DeepSeek-V4 因此设计了 hybrid attention：

```text
Hybrid Attention = CSA + HCA
```

其中：

- **CSA：Compressed Sparse Attention**
  - 先压缩 KV；
  - 再稀疏选择 top-k compressed KV；
  - 更偏向在效率和远程信息选择能力之间取得平衡。

- **HCA：Heavily Compressed Attention**
  - 更激进压缩 KV；
  - 不做 sparse top-k；
  - 在 heavily compressed KV 上做 dense attention；
  - 更偏向极致降低 KV cache 和计算。

论文摘要中给出的效率结果是，在 1M 上下文下：

- DeepSeek-V4-Pro 只需要 DeepSeek-V3.2 约 **27%** 的 single-token inference FLOPs；
- DeepSeek-V4-Pro 的 KV cache 约为 DeepSeek-V3.2 的 **10%**；
- DeepSeek-V4-Flash 更低，约为 DeepSeek-V3.2 的 **10% FLOPs** 和 **7% KV cache**。

---

## 2. CSA 的一句话定义

论文中的核心描述是：

> CSA first compresses the KV cache of each τ tokens into one entry, and then applies DeepSeek Sparse Attention for further acceleration.

翻译成中文就是：

> CSA 先把每 `τ` 个 token 的 KV cache 压缩成一个 entry，然后再应用 DeepSeek Sparse Attention，让每个 query token 只 attend 到少量 selected compressed KV entries。

所以 CSA 的核心不是简单的“压缩后 dense attend 所有压缩块”，也不是“先压缩再回到原始远程 token 精读”。

DeepSeek-V4 的 CSA 更准确的流程是：

```text
原始 hidden states / KV tokens
    ↓
生成 KV entries 和 compression weights
    ↓
每 τ 个 token 压缩成 1 个 compressed KV entry
    ↓
生成 compressed indexer keys
    ↓
Lightning Indexer 对 compressed entries 打分
    ↓
Top-k selector 选出最相关的 compressed KV entries
    ↓
selected compressed KV + sliding window uncompressed KV
    ↓
Shared Key-Value Multi-Query Attention
    ↓
Grouped Output Projection
    ↓
Attention output
```

---

## 3. CSA 的总体结构

论文 Figure 3 展示了 CSA 的核心架构。可以简化成下面的结构：

```text
Hidden States of KV Tokens
        │
        ▼
Token-Level Compressor
        │
        ▼
Compressed KV Entries
        │
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
Lightning Indexer              Core Attention
打分、Top-k 选择                使用 selected compressed KV
        │                         ▲
        ▼                         │
Selected Compressed KV Entries ───┘
        │
        ▼
Concatenation with Sliding Window KV
        │
        ▼
Shared Key-Value Multi-Query Attention
        │
        ▼
Grouped Output Projection
        │
        ▼
Attention Output
```

CSA 主要包含以下几个部分：

1. **Compressed Key-Value Entries**
2. **Lightning Indexer for Sparse Selection**
3. **Shared Key-Value Multi-Query Attention, MQA**
4. **Grouped Output Projection**
5. **Sliding Window KV Entries**
6. 其他稳定性和效率技巧：RMSNorm、Partial RoPE、Attention Sink、低精度计算与存储等。

---

## 4. 第一步：Compressed Key-Value Entries

假设输入 hidden states 为：

$$
H \in \mathbb{R}^{L \times d}
$$

其中：

- `L` 是序列长度；
- `d` 是 hidden size。

普通 attention 一般会为每个 token 生成 key 和 value：

$$
K,V \in \mathbb{R}^{L \times d_h}
$$

而 CSA 不直接让长程 attention 使用完整 per-token KV，而是先生成两组待压缩 KV-like entries，以及对应的 compression weights。

为了便于理解，可以用简化符号表示论文中的公式：

$$
U = H W_U
$$

$$
V = H W_V
$$

$$
A = H W_A
$$

$$
B = H W_B
$$

其中：

- `U` 和 `V` 是两组待压缩的 KV entries；
- `A` 和 `B` 是对应的 compression weights；
- `W_U, W_V, W_A, W_B` 是可训练参数；
- 每个 entry 的维度大致对应 attention head dimension。

接下来，CSA 每 `τ` 个 token 压缩成一个 compressed KV entry。

如果原始长度是 `L`，压缩后长度大约是：

$$
\frac{L}{\tau}
$$

例如：

| 原始长度 | 压缩率 `τ` | compressed KV 数量 |
|---:|---:|---:|
| 1,000,000 | 64 | 15,625 |
| 1,000,000 | 128 | 7,812 |
| 1,000,000 | 256 | 3,906 |

这一步直接把 KV cache 的序列维度缩小到原来的 `1/τ`。

---

## 5. CSA 的压缩不是简单平均池化

CSA 不是简单地做：

$$
Comp_i = mean(K_{i\tau:(i+1)\tau})
$$

它使用的是 **带可学习权重的 token-level compression**。

论文中提到，每个 compressed entry 来自 `2τ` 个 KV entries，但相邻 compressed entries 使用的索引存在重叠，所以整体压缩率仍然是 `1/τ`。

可以直观理解为：

```text
Comp_0 由一段局部 token 加权得到
Comp_1 由下一段局部 token 加权得到，但和 Comp_0 的来源有部分重叠
Comp_2 继续向后滑动，同样存在重叠
...
```

这种 overlapping compression 有几个好处：

1. **减少块边界损失**

   如果重要信息刚好跨越两个 block，简单分块平均容易丢失。重叠压缩可以缓解这种边界问题。

2. **让 compressed entry 更平滑**

   每个 compressed KV entry 不只是孤立 block 的摘要，而是能包含邻近上下文的一些信息。

3. **更适合 causal attention**

   压缩过程中可以严格控制只使用当前位置之前的信息，避免未来信息泄漏。

论文里的计算大致可以理解为：

$$
[\alpha;\beta] = Softmax([A + bias_A; B + bias_B])
$$

然后：

$$
Comp_i = \sum_j \alpha_j \odot U_j + \sum_j \beta_j \odot V_j
$$

其中：

- `⊙` 是 Hadamard product，也就是逐元素乘法；
- softmax 在 `2τ` 个候选位置上归一化；
- bias 是可学习的位置偏置；
- `α` 和 `β` 决定不同 token 对 compressed entry 的贡献。

直观理解：

> 每个 compressed KV entry 是模型学出来的局部摘要，而不是固定平均值。

这很关键。因为如果只做平均池化，代码变量名、数字、特殊符号、异常 token 都可能被冲淡。CSA 通过可学习 compression weights 让模型自己决定哪些 token 更应该进入压缩表示。

---

## 6. 第二步：Lightning Indexer 做稀疏选择

压缩后，CSA 得到：

$$
Comp \in \mathbb{R}^{\frac{L}{\tau} \times d_h}
$$

如果上下文是 1M，压缩率 `τ = 64`，compressed entries 仍然有 15,625 个。

这比 1M 少很多，但如果每个 query 都 dense attend 到 15K 个 compressed entries，在很多层、很多 head 上仍然有不小开销。

所以 CSA 继续引入一个轻量索引器，论文称为：

```text
Lightning Indexer
```

它的作用是：

> 给每个 query token 和每个 preceding compressed block 计算 index score，然后只选择 top-k 个 compressed KV entries 进入核心 attention。

---

## 7. Compressed Indexer Keys

CSA 会用和 compressed KV 类似的压缩方式，额外生成一组 compressed indexer keys：

$$
IComp \in \mathbb{R}^{\frac{L}{\tau} \times d_i}
$$

其中 `d_i` 是 indexer head dimension，通常比核心 attention 的维度更小。

这组 `IComp` 不是直接给最终 attention 用，而是给 selector 打分用。

可以理解为：

```text
Comp  ：真正给 core attention 当 key/value 的压缩记忆
IComp ：给 Lightning Indexer 检索和打分用的轻量索引
```

---

## 8. Query 端用低秩方式生成 indexer queries

对于当前 query token 的 hidden state：

$$
h_t \in \mathbb{R}^{d}
$$

CSA 先把它投影成一个低维 latent vector：

$$
c_t = h_t W_D
$$

然后再从这个 latent vector 生成多个 indexer query heads：

$$
[q^I_{t,1}, q^I_{t,2}, ..., q^I_{t,n_I}] = c_t W_U^I
$$

这里有两个重点。

### 8.1 低秩生成

不是直接从 `h_t` 生成所有 indexer queries，而是：

```text
h_t → low-rank latent c_t → multiple indexer query heads
```

这样可以降低计算量。

### 8.2 多个 indexer heads

多个 indexer heads 可以捕捉不同类型的相关性。例如：

- 有的 head 更关注主题相关；
- 有的 head 更关注实体名称；
- 有的 head 更关注代码结构；
- 有的 head 更关注位置或模式相关。

这比单个 dot product selector 更灵活。

---

## 9. Index score 的计算

对于 query token `t` 和某个 preceding compressed block `j`，CSA 会计算一个 index score：

$$
s_{t,j}
$$

论文里的形式大致可以理解为：

$$
s_{t,j} = \sum_{r=1}^{n_I} w_{t,r} \cdot ReLU(q^I_{t,r} \cdot IComp_j)
$$

其中：

- `q^I_{t,r}` 是第 `r` 个 indexer query head；
- `IComp_j` 是第 `j` 个 compressed indexer key；
- `w_{t,r}` 是 query token 对不同 indexer head 的权重；
- `ReLU` 会过滤掉负相关，让负相关不贡献正分数。

这说明 CSA 的 selector 不是简单的单头 dot product，而是：

```text
多个 indexer query heads
+ query-dependent head weights
+ ReLU 过滤负相关
+ 汇总成 compressed block score
```

直观上，Lightning Indexer 就像一个轻量检索器：

> 当前 token 先用低成本 indexer 在 compressed memory 上搜索，看哪些压缩块最可能有用。

---

## 10. Top-k selector

有了所有 index scores 后，CSA 对 compressed KV entries 做 top-k：

$$
Selected_t = TopK(s_{t,:})
$$

然后：

$$
CSprsComp_t = \{Comp_j \mid j \in TopK(s_{t,:})\}
$$

也就是说，每个 query token 只会选择少量 `k` 个 compressed KV entries 进入核心 attention。

这一步把远程 attention 的 key/value 数量从：

$$
\frac{L}{\tau}
$$

进一步降到：

$$
k
$$

例如：

```text
L = 1,000,000
τ = 64
compressed entries ≈ 15,625
top-k = 256
```

那么每个 query 的远程 core attention 只需要看 256 个 compressed entries，而不是 1,000,000 个原始 token，也不是全部 15,625 个 compressed entries。

---

## 11. 第三步：Shared Key-Value Multi-Query Attention

选出 sparse compressed KV entries 后，CSA 进行真正的核心 attention。

论文说 CSA 使用：

```text
Shared Key-Value Multi-Query Attention, MQA
```

并且每个 compressed KV entry 同时作为 attention key 和 attention value。

也就是说：

$$
key = CSprsComp_t
$$

$$
value = CSprsComp_t
$$

这和普通 attention 不一样。

普通 attention 通常有两套不同向量：

$$
K = H W_K
$$

$$
V = H W_V
$$

而 CSA 的核心 attention 中，compressed entry 同时承担 key 和 value：

```text
一个 compressed KV entry 既用于匹配，也用于输出聚合
```

这样可以进一步减少 KV cache 存储和访存压力。

---

## 12. Query 仍然是多头的

对于 query token `t`，CSA 会从前面的 latent vector `c_t` 生成多个 attention query heads：

$$
[q_{t,1}, q_{t,2}, ..., q_{t,n_H}] = c_t W_Q
$$

然后每个 query head 都对相同的 selected compressed KV entries 做 attention：

$$
o_{t,r}=CoreAttn(query=q_{t,r}, key=CSprsComp_t, value=CSprsComp_t)
$$

这就是 MQA 的思想：

- query 有多个 head；
- KV 是共享的；
- 不需要为每个 query head 存独立 KV；
- 显著减少 KV cache 和访存成本。

---

## 13. 第四步：Grouped Output Projection

CSA 的 query heads 数量较大。如果直接把所有 attention head 的输出拼接后投影回 hidden size，计算量会很高。

普通做法类似：

$$
O_t = [o_{t,1}; o_{t,2}; ...; o_{t,n_H}]
$$

$$
\hat{o}_t = O_t W_O
$$

如果 `n_H` 很大，那么输出投影矩阵 `W_O` 的成本也会很高。

DeepSeek-V4 因此设计了 **grouped output projection**：

1. 把 attention heads 分成若干组；
2. 每组先投影到较小的 intermediate dimension；
3. 再把所有 intermediate outputs 拼接；
4. 最后投影到最终 hidden size。

可以理解为：

```text
很多 attention head outputs
        ↓
按组分块
        ↓
每组先压缩投影
        ↓
拼接 intermediate outputs
        ↓
最终输出投影
```

这样可以降低输出投影的计算量。

---

## 14. 第五步：Sliding Window Attention 分支

CSA 有一个重要问题：

> 每个 query 只 attend 到 preceding compressed KV blocks，因此它不能访问自己所在 compressed block 内的其他 token。

举个例子，假设 `τ = 64`，当前 token 在 block 100 里面。按照严格 causal compressed block 规则，它只能 attend 到 block 99 及之前的 compressed entries，不能看 block 100 内自己前面的那些 token。

但语言建模中最近 token 通常非常重要。例如：

```text
def calculate_total(price, tax):
    return price +
```

当前 token 极大依赖最近几个 token。

如果只使用 compressed block，局部细粒度信息会受到损失。

所以 DeepSeek-V4 在 CSA 里额外引入：

```text
Additional Branch of Sliding Window Attention
```

每个 query token 额外生成最近 `w_win` 个 token 的未压缩 KV entries：

$$
KV_{window} = KV_{t-w_{win}:t}
$$

然后在 core attention 中，把 sliding window KV 和 selected compressed KV 一起使用：

```text
final KV = selected compressed KV + recent uncompressed sliding window KV
```

直观理解：

```text
远程信息：靠 compressed sparse attention
近程信息：靠 sliding window uncompressed KV
```

这样 CSA 既降低了长程 attention 成本，又不会明显损害局部建模能力。

---

## 15. CSA 的核心 attention 实际看哪些内容？

对于某个 query token，CSA 最终 attend 的对象大致是：

```text
1. top-k selected compressed KV entries
2. 最近 sliding window 里的 uncompressed KV entries
3. attention sink
```

不是看全部历史 token。

也不是看全部 compressed KV entries。

也不是回到远程原始 token 做 full attention。

可以写成：

$$
AttentionSet_t = TopKCompressedKV_t \cup WindowKV_t \cup Sink
$$

最终 attention 是：

$$
o_t = Attention(q_t, AttentionSet_t)
$$

这是 DeepSeek-V4 CSA 的核心设计取舍：

> 远程信息使用 compressed KV 表示，近程信息使用 uncompressed sliding window KV 表示。

---

## 16. CSA 和 DSA 的关系

论文中说：

> CSA applies the DSA strategy to select top-k compressed KV entries for core attention.

也就是说：

- **DSA** 是 DeepSeek Sparse Attention，是稀疏选择策略；
- **CSA** 是在 compressed KV entries 上使用 DSA；
- CSA = Compression + DSA + MQA + Sliding Window。

可以简单理解为：

```text
DSA：从很多 KV 里稀疏选 top-k
CSA：先把 KV 压缩，再在压缩 KV 上做 DSA
```

为什么先压缩再 DSA？

因为如果直接在 1M 个原始 token 上做 sparse selection，indexer 本身也会很贵。

先压缩到 `L/τ` 后，Lightning Indexer 的候选空间小很多。

---

## 17. 和普通 Dense Attention 的复杂度对比

普通 dense attention 中，每个 query token 要看所有历史 token：

$$
O(L)
$$

整个序列训练时大致是：

$$
O(L^2)
$$

CSA 中，每个 query 主要包含三部分成本：

1. Lightning Indexer 扫描 compressed indexer keys：

$$
O\left(\frac{L}{\tau}\right)
$$

2. Core attention 看 top-k selected compressed KV：

$$
O(k)
$$

3. Sliding window 看最近未压缩 KV：

$$
O(w_{win})
$$

所以单 query 的粗略成本可以理解为：

$$
O\left(\frac{L}{\tau} + k + w_{win}\right)
$$

相比 dense attention 的：

$$
O(L)
$$

当 `τ` 较大、`k` 和 `w_win` 固定时，成本会显著下降。

例如：

```text
L = 1,000,000
τ = 64
L / τ = 15,625
k = 256
w_win = 512
```

普通 dense attention 单 query 要看接近 1,000,000 个位置。

CSA 的 core attention 只看：

```text
256 selected compressed entries
+ 512 recent window entries
```

虽然 indexer 还要扫描 15,625 个 compressed indexer keys，但 indexer 是轻量、低维、低精度路径，比核心 attention 扫 1M 原始 KV 便宜很多。

---

## 18. 为什么 CSA 可以显著减少 KV cache？

普通 attention 的 KV cache 需要为每个 token 存 key 和 value：

```text
每个 token:
  K: d_h
  V: d_h
```

CSA 的长程 KV cache 变成：

```text
每 τ 个 token:
  一个 compressed KV entry
```

而且这个 compressed KV entry 在核心 attention 中同时作为 key 和 value。

所以长程部分的 KV cache 序列长度变成原来的：

$$
\frac{1}{\tau}
$$

同时 key/value 共享表示，进一步减少存储。

论文第 2.3.4 节还提到以下效率优化：

1. RoPE 维度使用 BF16；
2. 非 RoPE 维度使用 FP8；
3. Lightning Indexer 的 attention computation 使用 FP4；
4. DeepSeek-V4 选用了比 DeepSeek-V3.2 更小的 attention top-k；
5. compressed attention 和 hybrid attention 是降低 FLOPs 与 KV cache 的主要原因。

论文还说，以 BF16 GQA8、head dimension 128 作为常见 baseline，DeepSeek-V4 在 1M context 下 KV cache size 可以降到约 baseline 的 **2%**。

注意：这里的 **2%** 是相对 BF16 GQA8 baseline；摘要中的 **10%** 是相对 DeepSeek-V3.2。二者参照物不同。

---

## 19. CSA 里的额外稳定性技巧

### 19.1 Query 和 KV Entry Normalization

论文说，CSA 和 HCA 都会在 core attention 前，对以下对象做额外 RMSNorm：

- 每个 query head；
- compressed KV entry 的唯一 head。

目的：

```text
避免 attention logits 爆炸，提高训练稳定性
```

因为 compressed KV entry 同时承载 key 和 value 信息，如果尺度不稳定，attention score 容易失控。

---

### 19.2 Partial Rotary Positional Embedding

DeepSeek-V4 不是对全部维度使用 RoPE，而是：

> 对 query vector 和 KV entry vector 的最后 64 维应用 RoPE。

因为 CSA 中 compressed KV entry 同时作为 key 和 value，所以 attention 输出会携带绝对位置嵌入。

论文说，为了处理这个问题，会对 core attention outputs 的最后 64 维再应用一个位置为负的 RoPE，使输出携带相对位置信息。

直观理解：

```text
输入端加入位置
attention 输出端再抵消/转换位置
让输出更像相对位置表示
```

---

### 19.3 Attention Sink

CSA/HCA 中还使用了 attention sink。

普通 softmax attention 的权重和为 1：

$$
\sum_j a_j = 1
$$

attention sink 会在 softmax 分母中加入一个可学习的 sink logit：

$$
a_{i,j} = \frac{e^{l_{i,j}}}{\sum_j e^{l_{i,j}} + e^{s}}
$$

这样真正分配给 token/KV entries 的 attention mass 可以小于 1，甚至接近 0。

作用是：

> 如果某个 query head 当前不需要从上下文中取信息，可以把注意力“漏”到 sink，而不是被迫分配给某些无关 token。

这对 sparse attention 很有用，因为 top-k 中不一定总有非常相关的 entry。Attention sink 给模型一个“不看任何上下文”的出口。

---

## 20. CSA 与 HCA 的区别

DeepSeek-V4 不只使用 CSA，还交错使用 HCA。

二者区别如下：

| 项目 | CSA | HCA |
|---|---|---|
| 压缩强度 | 中等 | 更强 |
| 是否 sparse top-k | 是 | 否 |
| 是否 dense attend 压缩后 KV | 否，只看 top-k | 是，看全部 heavily compressed KV |
| 是否有 sliding window | 有 | 有 |
| 主要目标 | 效率与动态远程检索能力之间平衡 | 极致压缩 KV cache 和计算 |
| 适合 | 需要动态选择远程信息 | 更粗粒度的全局记忆 |

HCA 相当于：

```text
把 KV 压得更狠
然后在 heavily compressed KV 上做 dense attention
```

CSA 相当于：

```text
先适度压缩 KV
再从 compressed entries 里稀疏选 top-k
```

DeepSeek-V4 交错使用二者，是为了兼顾：

- 长程信息可检索性；
- KV cache 极小；
- inference FLOPs 低；
- 1M context 可用。

---

## 21. CSA 的直观类比

假设上下文是一本 1000 页的书，当前问题是：

> 前面某一处提到的 API 限流规则是什么？

普通 dense attention 类似于：

```text
每生成一个 token，都把 1000 页全部扫一遍。
```

CSA 类似于：

```text
1. 每 τ 个 token 压缩成一张摘要卡片。
2. Lightning Indexer 先在摘要卡片里快速找相关卡片。
3. 选 top-k 张卡片。
4. 核心 attention 只读这 top-k 张卡片。
5. 同时保留最近几页的原文窗口。
```

它不是完整读所有原文页，而是：

```text
远处读压缩摘要
近处读原文窗口
```

这就是它能支持 1M context 的原因。

---

## 22. CSA 的优点

### 22.1 显著减少长上下文 attention 成本

CSA 通过以下路径缩小长程 attention 的候选空间：

$$
L \rightarrow \frac{L}{\tau} \rightarrow k
$$

先压缩，再 top-k 选择。

---

### 22.2 大幅降低 KV cache

每 `τ` 个 token 只存一个 compressed KV entry，而且 key/value 共享表示，再配合 FP8/FP4 等低精度策略，KV cache 显著降低。

---

### 22.3 保留动态远程检索能力

固定 sliding window 只能看最近内容。

CSA 通过 Lightning Indexer 可以动态选择远程 compressed entries。

即使关键信息在几十万 token 之前，也有机会通过 top-k 被选中。

---

### 22.4 保留局部细节

因为有 sliding window uncompressed KV，最近 token 不会被压缩损失。

这对代码、数学、对话、函数调用、工具调用等场景都很重要。

---

### 22.5 更适合 GPU 工程优化

CSA 选择的是 compressed KV entries，而不是任意原始 token。

相比 token-level 随机稀疏访问，compressed block/entry 级别的稀疏访问更规整，更容易做 kernel 优化。

论文还提到 DeepSeek-V4 使用了 TileLang、自研 kernel、batch-invariant deterministic kernel 等工程优化。

---

## 23. CSA 的潜在缺点

### 23.1 压缩会丢细节

远程 token 被压缩成 compressed entries 后，细粒度信息可能丢失。

例如：

```text
某个很长文件里唯一出现的一串 ID
某个变量名
某个特殊边界条件
某个精确数字
某段代码里的罕见符号
```

如果 compression weights 没有保留这些细节，即使后续 top-k 选中了该 compressed entry，核心 attention 看到的也只是压缩后的表示。

---

### 23.2 Top-k 可能漏选

如果 Lightning Indexer 没有把真正相关的 compressed block 排进 top-k，核心 attention 就看不到它。

这类错误可以称为：

```text
retrieval miss
```

对长上下文问答、代码检索、长文档定位尤其关键。

---

### 23.3 Indexer 本身也有成本

CSA 并不是完全 `O(1)` 的长上下文访问。

每个 query 仍然需要在 `L/τ` 个 compressed indexer keys 上打分。

只是这个路径是轻量、低维、低精度的，因此比 dense attention 直接扫 `L` 个原始 KV 便宜得多。

---

### 23.4 工程实现复杂

CSA 涉及很多组件：

- token-level compression；
- overlapping compression；
- compressed indexer keys；
- low-rank indexer queries；
- top-k sparse selection；
- shared key-value MQA；
- grouped output projection；
- sliding window 拼接；
- partial RoPE；
- attention sink；
- FP8/FP4 混合精度；
- 长上下文 KV cache 管理。

工程复杂度远高于普通 attention。

---

## 24. 和泛化理解的关键差异

一个容易误解的地方是：

```text
压缩 → 找相关块 → 回到原始 token 精读
```

这是一种泛化的 compressed sparse attention 思路，但不是 DeepSeek-V4 PDF 中 CSA 的精确定义。

DeepSeek-V4 的 CSA 更准确是：

```text
压缩原始 KV
    ↓
在 compressed KV 上 sparse selection
    ↓
对 selected compressed KV 做 core attention
```

也就是说，DeepSeek-V4 CSA 的远程分支不会回到远程原始 token 做 full attention。

只有最近窗口部分保留 uncompressed KV：

```text
远程：compressed sparse KV
近程：uncompressed sliding window KV
```

这是 DeepSeek-V4 CSA 的核心设计取舍：

> 远程信息用压缩表示换效率，近程信息用原始 KV 保细节。

---

## 25. 最终总结

DeepSeek-V4 的 **Compressed Sparse Attention** 可以浓缩成下面这句话：

> CSA 是一种面向 1M 长上下文的高效 attention 结构，它先把每 `τ` 个 token 的 KV cache 压缩成一个 shared key-value entry，再用 Lightning Indexer 对 compressed entries 做 top-k 稀疏选择，核心 attention 只访问这些 selected compressed entries，同时拼接最近 sliding window 的未压缩 KV 来保留局部细节。

最关键流程是：

```text
Hidden States
   ↓
生成 KV entries + compression weights
   ↓
每 τ 个 token 压成 1 个 compressed KV entry
   ↓
生成 compressed indexer keys
   ↓
query 通过 low-rank Lightning Indexer 打分
   ↓
top-k 选择 compressed KV entries
   ↓
selected compressed KV + sliding window KV
   ↓
shared key-value MQA
   ↓
grouped output projection
   ↓
attention output
```

它的本质是：

```text
用压缩减少 KV cache
用稀疏选择减少 attention FLOPs
用 sliding window 保留局部精度
用 MQA 和 grouped projection 进一步降低计算和存储
```

参考来源：

- [DeepSeek-V4 PDF on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- [DeepSeek-V4-Pro inference implementation referenced by the paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)
