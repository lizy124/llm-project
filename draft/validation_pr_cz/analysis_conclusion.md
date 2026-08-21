# analysis and conclusion

## 结果分析

这次验证把两件事分开了：

1. 协议层是否真的少发了 lookup 数据；
2. 真实服务里是否真的触发了 HBM prefix 命中后的 suffix lookup。

### 协议层

协议级 benchmark 结果很清楚：

- `SUFFIX` 发送的 hashes 数量是 `FULL` 的一半；
- payload 从 270354 bytes 降到 135188 bytes；
- ZMQ RTT mean 从 758.40 us 降到 427.94 us。

这说明 PR 的核心改动有效，而且收益和 HBM hit 比例正相关。

### 真实 e2e

真实 e2e 里，前两轮短 suffix 请求没有进入最终的 suffix lookup load 路径，但这不是协议问题，而是 `token_len` 被 cache granularity 截断后没有留下可加载的 suffix。

把 suffix 拉长后，日志里出现了：

- `computed=384`
- `mode=LookupHashMode.SUFFIX`
- `hit_tokens=384`

这说明真实服务里已经出现了 HBM prefix 命中，并且 suffix lookup 路径被实际执行。

### 端到端时间

端到端时间从 1.682 s 降到 1.629 s，只下降了约 53 ms，约 3.2%。

原因是：

- lookup 协议只占整个请求路径的一部分；
- 模型 forward、调度和 NPU 执行成本更大；
- 所以协议优化在 e2e 上会被明显稀释。

## 结论

- PR 的协议修改是正确的。
- `suffix` 真的减少了 scheduler -> worker 的 lookup 负载。
- 真实服务里也已经看到 HBM prefix 命中后的 suffix lookup。
- 但端到端收益不大，当前观测约 53 ms，约 3.2%。
- 如果要继续做性能报告，应该把重点放在协议层和 lookup RTT，而不是只看 TTFT。
