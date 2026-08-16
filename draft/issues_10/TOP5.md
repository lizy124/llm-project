# AscendStore 池化优化 Top 5

> 代码基线：vllm-ascend `main@d5e9816065ede613327d93908f87fee9f5c47128`
>
> 选取标准：对 AscendStore 池化正确性、传输生命周期、后端契约、控制面可靠性和通用数据面效率的综合价值。
>
> 完整排序与理由见 [VALUE_RANKING.md](VALUE_RANKING.md)，代码证据见 [CODE_AUDIT.md](CODE_AUDIT.md)。

## Top 5 清单

| 排名 | 任务 | 对池化的主要价值 | 定位 |
|---:|---|---|---|
| 1 | [kv-28：hybrid KV load 失败传播](issue_kv-28_C2_multi_group_failure.md) | 建立 multi-group/hybrid 加载原子性，禁止残缺 KV 进入 forward | 核心正确性边界 |
| 2 | [kv-25：transfer thread 终止式失败协议](issue_kv-25_C1_transfer_thread_exception.md) | 统一启动、fatal、取消、等待方通知和资源清理生命周期 | 核心执行层基础 |
| 3 | [kv-26：Backend.put per-key 结果](issue_kv-26_C1-1_backend_put_failure.md) | 让写入结果、KV event 和 backend 状态真实可知，并支撑安全 batching | 核心后端契约 |
| 4 | [kv-27：ZMQ lookup 恢复](issue_kv-27_C1-2_zmq_lookup_failover.md) | 消除 lookup 永久阻塞，建立 timeout、错误响应和 socket 恢复 | 核心控制面可靠性 |
| 5 | [kv-07：non-layerwise backend batching](issue_kv-07_P07_non_layerwise_io_merge.md) | 减少普通 load/save 的 backend 调用次数，是覆盖面最广的性能候选 | 通用数据面优化 |

## 任务边界

### kv-28

- 近期目标：non-layerwise multi-group 任一加载失败即 fail-fast，受影响 request 不得进入 forward。
- 长期目标：建立 request-level failed IDs 或等价失败通道，只终止受影响请求。
- 不把当前不支持的 hybrid per-block recompute 当作既定解法。

### kv-25

- 覆盖 transfer thread startup、fatal 状态、queue accounting、后续入队拒绝和等待方失败传播。
- 按普通 send、async recv、layerwise key/GVA 分别做故障注入和资源所有权验证。
- cleanup 必须幂等，不能覆盖首次 fatal 原因，也不能把 event set 解释为成功。

### kv-26

- 先按实际 Mooncake、MemCache、Yuanrong 版本确认返回码和重复对象语义。
- 统一与输入 keys 等长的 `AVAILABLE/FAILED/UNKNOWN` 结果。
- 只为确认 `AVAILABLE` 的 key 发布 `BlockStored`；远端失败不阻止本地 GPU block 生命周期完成。

### kv-27

- client 使用有界 timeout；超时或协议错误后关闭并重建 REQ socket。
- server 对已接收请求返回 typed success/error response，保持 REP 状态机可恢复。
- timeout 和重试参数必须根据目标机器正常 lookup 延迟分布确定。

### kv-07

- 先执行 Phase 0，统计同步 load、async load 和 save 的实际 batch 机会及 backend 调用成本。
- 只实现机器数据证明有价值的路径，不要求三条路径一次性全部重写。
- 必须复用 kv-26 的 per-item 结果，并与 kv-25 的取消/fatal 语义一致。

## 建议实施关系

```text
kv-28 ── failure propagation ──┐
                              ├──> kv-25 统一 transfer failure lifecycle
kv-27 ── bounded lookup ───────┘

kv-26 backend result contract ───> kv-07 safe batching
```

实施时不必严格串行：

1. `kv-28` 与 `kv-27` 可先并行关闭错误计算和永久阻塞。
2. `kv-25` 统一 transfer failure lifecycle，并与 `kv-28` 的 async 失败传播对齐。
3. `kv-26` 建立 backend 结果契约。
4. `kv-07` 先做 Phase 0；确认收益后基于 `kv-25`、`kv-26` 的契约实施 batching。

## 最终记录

本目录确定的 AscendStore 池化优化 Top 5 为：

`kv-28、kv-25、kv-26、kv-27、kv-07`

其中前四项补齐池化系统的正确性、执行层、后端和控制面基础；`kv-07` 是在这些契约之上最值得优先验证的通用性能优化。
