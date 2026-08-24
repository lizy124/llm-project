# Finding Matrix

## 静态 finding 矩阵

| ID | 等级 | 静态状态 | 触发条件 | 主要影响 | 服务器确认重点 |
|---|---|---|---|---|---|
| F-001 | P1 | 已由静态逻辑确认；影响表现需服务器确认 | external hit > 0；`load_async=True` 会进入异步分支 | KV 未加载、输出错误或请求长期等待；没有 save | 本地 KV 内容、forward 输出、`WAITING_FOR_REMOTE_KVS`、`finished_recving`、远端写入 |
| F-002 | P1 | 静态确认，部署边界条件性 | 非可信客户端可访问 endpoint | `cloudpickle` 任意反序列化；潜在 RCE/DoS | 绑定地址、ACL；隔离进程中的无害反序列化探针 |
| F-003 | P1 | 静态确认，攻击可达性需服务器确认 | 知道 identity 的未认证新 session 注册 | 驱逐/替换活动服务；factory 失败时旧服务已丢失 | 同 identity 新 session、旧服务 close、factory 失败后的 count/lookup |
| F-004 | P2 | 静态确认，容量触发需服务器确认 | scheduler service 已发布后绑定提交失败 | 客户端注册失败但 server 残留 service | 填满 worker executor 后注册，检查 count/lease/unregister |
| F-005 | P2 | 静态确认，backend close 行为需服务器确认 | owner close 抛异常 | 旧资源可能仍在，随后同 session 创建新 service | 注入 close failure，检查旧/新 service、线程和 socket |
| F-006 | P2 | 静态确认 | identity 反复替换/重启 | `retired_sessions` 无界增长 | retired set 大小、RSS、长时间趋势 |
| F-007 | P2 | 静态确认，阈值需服务器确认 | 大 payload 或高并发客户端 | queue/RSS/CPU 无界，可能 DoS | 逐级增加 frame bytes、hash 数和并发，记录 HWM/队列/错误 |
| F-008 | P2 | 静态确认，错误 key 表现需服务器确认 | 配置 `use_layerwise=True` | 静默按非 layerwise lookup，错误命中/漏命中 | 记录配置、实际对象属性、lookup keys 和 hook 调用 |
| F-009 | P2 | 静态规则冲突已确认；本轮未运行 checker | pre-commit 或最小依赖环境 | import policy/依赖门禁失败 | 在 CI 隔离运行既有 checker 和最小安装 |

## 结论标签说明

- `已由静态逻辑确认`：方法实现、调用点和状态转移已在本轮逐段阅读并闭合。
- `需服务器确认`：只用于确定触发频率、具体 backend 影响、网络边界或资源阈值，不否定静态缺陷本身。
- `目前证据不足`：不进入正式 finding，只记录在 `coverage_gaps.md`。

## 优先级顺序

1. 先处理 F-001，明确 MP connector 是完整 transfer connector 还是 lookup-only connector。
2. 在任何跨主机部署前处理 F-002/F-003，避免把未认证反序列化和 identity replacement 带入服务器。
3. 处理 F-004/F-005 的失败回滚和 close 状态，再做 lease/recovery 压测。
4. 处理 F-007/F-008/F-009 的资源、配置和仓库门禁问题。
5. 用服务器验证计划补足 F-006 及所有“需服务器确认”项后进行二次复核。
