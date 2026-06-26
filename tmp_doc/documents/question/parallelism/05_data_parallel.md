# 05. Data Parallel 如何做请求级并行？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/v1/engine/`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/distributed/parallel_state.py`

本问题关注：Data Parallel 在 vLLM 中如何把不同请求分给不同 replica，每个 DP replica 内部如何拥有自己的 TP / PP / EP / CP 组合，EngineCore / Executor / Worker 如何协同，以及 DP 下 KV cache、SchedulerOutput、ModelRunnerOutput 和负载均衡如何处理。

---

## 1. 一句话回答

Data Parallel 是请求级并行：

```text
多个模型 replica 同时存在；
不同请求 / batch 被分发到不同 replica；
每个 replica 内部可以再使用 TP / PP / EP / CP 完成单个请求的模型计算。
```

一句话记忆：

```text
DP 是“不同请求给不同模型副本跑”。
```

---

## 2. 本文要回答的问题

```text
DP replica 如何定义？
DP 和 TP / PP 的边界是什么？
Scheduler 是全局调度还是每个 DP rank 独立调度？
Executor 如何把请求分发给 DP rank？
DP 下 KV cache 是否 replica-local？
DP 输出如何回到 EngineCore？
DP load balancing 在哪里发生？
```

---

## 3. 最小主链路占位

```text
外层 Engine / EngineCore
  → 请求进入系统
  → 根据 DP 策略选择 replica
  → 对应 DP replica 内部 Scheduler / Worker 执行
  → replica-local KV cache / batch state
  → ModelRunnerOutput 回到对应 EngineCore / 汇总层
  → 输出给用户
```

---

## 4. DP 和模型并行的区别

```text
TP / PP：
  一个请求的一次 forward 由多个 rank 合作完成。

DP：
  不同请求分给不同 replica；每个 replica 负责自己的请求。

所以：
  DP 是请求级扩展；
  TP / PP 是单个模型副本内部的计算切分。
```

---

## 5. 待梳理源码点

```text
data_parallel_size 配置
DP rank / local DP rank 计算
Executor 与 DP worker 分发关系
EngineCore 是否按 DP 拆分
SchedulerOutput 在 DP 下的归属
DP load balancing 相关逻辑
DP 下 KV cache manager / block pool 是否隔离
DP 输出聚合路径
```

---

## 6. 和其他并行的关系

```text
DP + TP：
  每个 DP replica 内部有一个 TP group。

DP + PP：
  每个 DP replica 内部可以是一条 PP pipeline。

DP + EP：
  expert placement 可以在 replica 内部组织，也可能和 DP 维度存在组合约束。

DP + KV cache：
  默认理解为每个 replica 持有自己的 KV cache 状态，跨 replica 共享需要 KV transfer / connector 等额外机制。
```
