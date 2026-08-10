# PD 分离补充验证复盘与重跑方案

## 这次补充验证的教训

1. 不要复用旧 run 目录
   - 旧目录里的 `mooncake.json` 会污染新一轮启动参数。
   - 这次一度还在连旧的 master 端口，导致判断失真。

2. Mooncake master 要同时检查 `rpc_port` 和 `metrics_port`
   - 只换 master 端口不够，`9003` / `9004` 这类 admin/metrics 端口也可能被占用。
   - master 没完整起来时，prefill / decode 会一直 `Connection refused`。

3. 先确认 master 真正 ready，再起 prefill / decode
   - 不能只看启动命令返回 PID。
   - 必须看日志里是否有 `Master service started`，且没有端口绑定失败。

4. 先用轻模型和更小的请求压力
   - PD 分离先用 `/mnt/weight/Qwen3-0.6B`。
   - 不要一上来用更大的模型，避免把显存问题和链路问题混在一起。

5. 端口要一次性规划好
   - master、metrics、prefill、decode、proxy 都要先确认空闲。
   - 避免后面重跑时继续撞端口。

## 重新验证方案

### 目标
只验证 PD 分离这条补充链路：`kv_producer` / `kv_consumer` + Mooncake + proxy。

### 方案
1. 新建全新 run 目录，不复用旧目录。
2. 先探测并固定一组空闲端口。
3. 启动 Mooncake master，确认日志里没有 bind 冲突。
4. 启动 prefill / decode，使用轻模型 `Qwen3-0.6B`。
5. 启动 proxy，确认能看到 prefill / decode 两端都注册成功。
6. 发送 2 条普通请求，确认请求返回 `200`。
7. 如普通请求通过，再补 2~4 条流式请求。
8. 保存所有日志、summary、请求明细和最终结论。

### 通过标准
- master 日志无端口冲突
- prefill / decode 能真正 ready
- proxy 能路由请求
- 普通请求和流式请求都成功
- 记录里保留完整日志和请求结果

### 备注
如果再重跑 PD 分离，优先级是：
- 先保证 master 完整启动
- 再保证 prefill / decode ready
- 最后才测请求
