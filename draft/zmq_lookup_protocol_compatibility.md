# ZMQ lookup 协议兼容性说明

本文解释这次 `lookup_hash_mode=suffix` 的 ZMQ 协议改动到底改了什么，为什么它会带来兼容性风险，以及这个风险和“性能收益”之间是什么关系。

## 结论先说

这个 PR 的核心收益是成立的：当 scheduler 已经知道 HBM prefix 命中时，它可以只把 suffix hashes 发给 lookup worker，减少 payload bytes 和 ZMQ RTT。

但它也确实引入了一个兼容性前提：**lookup 请求和响应必须是同一套新协议**。当前实现没有做旧协议回退，也没有做版本协商，所以混跑旧 client / 新 server，或者新 client / 旧 server，都会有风险。

换句话说：

- **同版本一起升级**：可以工作。
- **混版本滚动升级**：没有保证。

## 1. 这次协议到底改了什么

### 旧协议

旧的 lookup 请求本质上是 4 段：

1. `token_len`
2. `kv_group_ids`
3. `hbm_hit_tokens`
4. `hashes`

### 新协议

新协议在 `hbm_hit_tokens` 后面插入了一段新的 frame：

1. `token_len`
2. `kv_group_ids`
3. `hbm_hit_tokens`
4. `lookup_hash_mode`
5. `hashes`

其中 `lookup_hash_mode` 目前只支持：

- `full`
- `suffix`

代码里可以直接看到这个位置依赖：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:1191`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:317`

客户端发送时会把 mode frame 插在 hash frames 前面：

```python
all_frames = [
    token_len.to_bytes(4, byteorder="big"),
    *kv_group_frames,
    hbm_hit_tokens.to_bytes(4, byteorder="big"),
    *lookup_mode_frames,
    *hash_frames,
]
```

服务端则按固定位置解析：

```python
token_len = int.from_bytes(all_frames[0], byteorder="big")
kv_group_ids = self.decoder.decode([all_frames[1]])

hbm_hit_tokens = int.from_bytes(all_frames[2], byteorder="big")
lookup_hash_mode = LookupHashMode(self.decoder.decode([all_frames[3]]))
hashes_str = self.decoder.decode(all_frames[4:])
```

这说明它不是“在旧协议后面追加一个可忽略字段”，而是**改变了 frame 的位置语义**。

## 2. 为什么这不是向后兼容的扩展

这里的关键点不是“多了一个字段”，而是：**当前实现没有协商机制，也没有默认回退**。

### 情况 A：旧 client -> 新 server

旧 client 只会发 4 段，不会发 `lookup_hash_mode`。

新 server 却固定读取 `all_frames[3]` 作为 mode。这样就会出现两种问题：

- 如果 frame 数不够，server 侧会在解析阶段直接异常；
- 如果 frame 数量碰巧够，但内容不是 mode，server 会把错误内容当成 mode 解析。

由于 lookup server 是 REQ/REP 模式，**只要 server 在 reply 之前抛异常，就不会返回 response**。client 端最终看到的就是超时。

我实际验证到的旧格式请求现象就是这样：client `recv()` 超时，而不是拿到一个明确的错误响应。

### 情况 B：新 client -> 旧 server

旧 server 不认识新增的 `lookup_hash_mode` frame，也没有为它做兼容分支。

在这种情况下，行为取决于旧 server 的解析实现，但结论一样：

- 没有协议协商；
- 没有 frame 级 fallback；
- 没有“缺 mode 就默认 full”的兼容逻辑。

所以不能保证旧 server 会正确理解新请求。

## 3. 为什么它会表现成“超时”而不是“明确报错”

因为这里用的是 ZMQ REQ/REP。

REQ 端发出请求后，必须等 REP 端回一个响应。现在 server 端的处理逻辑是：

1. 收到 multipart frames；
2. 按固定位置解析；
3. 调 `pool_worker.lookup_scheduler(...)`；
4. 再 `send(response)`。

如果在第 2 步或第 3 步就抛异常，response 就不会发出去。

于是 client 端只能等到 socket 超时，外部表现就是：

- `Again Resource temporarily unavailable`
- 或者请求一直挂住直到超时

这也是为什么这个风险看起来不像“功能报错”，而更像“协议不兼容导致的请求无响应”。

## 4. 这个风险对合入意味着什么

这要分两层看。

### 如果你的部署是“同一版本整套升级”

也就是：

- scheduler / lookup client
- lookup server
- worker

都来自同一个 commit 或同一轮发布包，那么这个 PR 是可用的。因为双方都知道新 frame 语义。

### 如果你的部署会“滚动升级”或“混版本运行”

那就有风险。

因为当前协议没有版本号，也没有兼容分支，所以旧新版本之间**没有 wire compatibility 保证**。这意味着：

- 升级过程中如果 client 先升级、server 还没升级，可能超时；
- 反过来也一样；
- 这不是“性能退化”，而是“协议直接不通”。

所以，合入本身不一定有问题，但**合入说明里必须明确这是一个需要同版本配套的协议变更**。

## 5. 这个问题和性能收益并不矛盾

协议变更的目的，是为了在 `suffix` 模式下减少 hash 发送量。

当 scheduler 已经知道 HBM prefix 命中时，它会把已命中的前缀从 `block_hashes` 里剪掉，只发送 suffix 部分。

我跑出来的协议级 benchmark 已经证明了这一点：

- 16K tokens，50% HBM hit：payload bytes 减半左右，RTT 也下降；
- 64K tokens，50% HBM hit：payload 从 `270354` 降到 `135188`；
- 64K tokens，75% HBM hit：payload 从 `270354` 降到 `67604`，RTT 均值也明显下降。

所以它的收益是真实的，但这个收益建立在“新旧两端都升级到同一协议”的前提上。

## 6. 如果要把兼容性补齐，通常有三种办法

### 方案 1：给协议加版本号

比如请求里增加 `protocol_version` 或 `features`。

server 看到老版本时，按旧协议解析；看到新版本时，按新协议解析。

优点：最稳。

缺点：改动相对多。

### 方案 2：把 `lookup_hash_mode` 变成可选字段，缺省按 `full`

也就是：

- 旧请求没有 mode 时，server 仍然能工作；
- 新请求带 mode 时，启用 suffix 逻辑。

优点：兼容性最好。

缺点：需要严格定义 frame 边界和默认行为。

### 方案 3：明确声明这是锁步协议，不做兼容

也就是不修兼容性，而是把升级要求写清楚：

- client/server 必须同版本；
- 不能混跑；
- 滚动升级期间不保证可用。

优点：实现最简单。

缺点：运维和发布风险最大。

## 7. 当前这次验证能支持什么结论

可以支持的结论：

- 新协议可用；
- `suffix` 模式可传到 worker；
- payload bytes 和 ZMQ RTT 都有收益；
- 最小 `AscendStoreConnector + suffix` 服务可以跑起来。

不能支持的结论：

- 新旧协议混跑没问题；
- 旧 client / 新 server 自动兼容；
- 新 client / 旧 server 自动兼容；
- 异常 frame 会被明确返回错误，而不是超时。

## 8. 最简短的建议

如果这个 PR 以“性能优化”名义合入，我建议在说明里明确一句：

> 这是一个需要 client/server 同版本配套的 lookup 协议变更，暂不保证旧版本兼容。

如果项目要求滚动升级兼容，那就应该再补一层协议回退或版本协商，而不是直接按当前实现合入。
