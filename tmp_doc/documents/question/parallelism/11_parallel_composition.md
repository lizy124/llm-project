# 11. TP / PP / DP / EP / CP 如何组合？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/model_executor/`
- `vllm/vllm/v1/attention/`

本问题关注：多种并行策略同时开启时如何理解 rank mesh，TP / PP / DP / EP / CP 之间哪些是正交维度，哪些会复用或约束彼此，单个请求的一次 forward 会经过哪些 group，以及组合并行下最容易混淆的状态归属问题。

---

## 1. 一句话回答

组合并行要按两个问题拆开看：

```text
1. 单个请求的一次 forward 由哪些 rank 合作完成？
   主要看 TP / PP / CP / EP。

2. 不同请求如何分给不同 replica？
   主要看 DP。
```

最小心智模型：

```text
DP replica 内部可以有一套 TP x PP x EP x CP 的模型并行结构。
```

---

## 2. 本文要回答的问题

```text
TP + PP 如何组合？
DP + TP + PP 的 rank mesh 如何理解？
EP 是独立维度还是依附于 TP / DP？
CP 和 attention backend 如何组合？
KV cache 在组合并行下属于哪个 rank / group？
logits / sampling 在组合并行下在哪里发生？
哪些配置组合有整除或 backend 支持约束？
```

---

## 3. 典型组合占位

### 3.1 TP + PP

```text
TP：
  同一 layer 内多个 rank 一起算。

PP：
  不同 layers 放在不同 stage。

组合后：
  每个 pipeline stage 内部有一个 TP group；
  stage 内用 TP 通信；
  stage 间用 PP send / recv。
```

### 3.2 DP + TP + PP

```text
DP：
  多个 replica 处理不同请求。

每个 DP replica 内部：
  可以是一组 TP x PP rank。

理解方式：
  请求先被分配到某个 DP replica；
  然后在该 replica 内由 TP / PP 合作完成 forward。
```

### 3.3 TP + EP

```text
TP：
  dense tensor 切分。

EP：
  MoE expert 切分。

组合后：
  dense 层走 TP 通信；
  MoE 层先 router，再 token all-to-all 到 expert rank；
  expert 输出 combine 后继续后续层。
```

### 3.4 CP + Attention Backend

```text
CP：
  切 context / KV / attention 计算范围。

组合后：
  backend 必须支持 partial attention state；
  partial output 需要 LSE merge；
  不是所有 attention backend 都能支持。
```

---

## 4. 组合并行下的状态归属占位

```text
模型权重：
  受 TP / PP / EP 影响。

请求状态：
  主要受 DP / Scheduler / Worker 归属影响。

KV cache：
  受 TP head 分片、PP layer 归属、DP replica 隔离、CP context 切分影响。

attention metadata：
  受 TP head 数、CP 本地 seq lens、KV block table、backend 能力影响。

logits / sampling：
  受 PP last stage、TP vocab 分片、DP replica 输出回收影响。
```

---

## 5. 待梳理源码点

```text
parallel_config 中 size 组合校验
parallel_state.py rank mesh 构造
model layer partition + TP group 构造
MoE expert group 与 TP / DP 的关系
DCP / PCP group 与 TP / PP 的关系
is_first_pp_rank / is_last_pp_rank 与 TP rank 关系
logits processor / sampler 在组合并行下的 rank 条件
KV cache group 与 attention group 的组合逻辑
```

---

## 6. 常见误区占位

```text
误区 1：world_size = TP * PP 就够了。
  实际还可能包含 DP / EP / CP 等维度或变体。

误区 2：DP 和 TP 都是“多卡并行”，所以类似。
  DP 切请求，TP 切 layer 内 tensor。

误区 3：EP 就是 TP 的一种。
  EP 切 expert，核心通信是 token all-to-all，不是普通 tensor all-reduce。

误区 4：CP 只是 KV cache 分片。
  CP 直接改变 attention 计算和 softmax merge 语义。
```
