# MLA KV Cache 读取去重优化讨论

> 用于会议讨论，梳理前因后果、现状证据、优化提案、收益/风险与待验证点。
> 代码路径基于 `D:\lzy\project\kv_pool\code\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool`。
> 日期：2026-08-08

---

## 1. 背景

### 1.1 MLA 的 KV cache 特性

MLA（Multi-head Latent Attention）把多个 head 的 K/V 压缩成一个 **latent vector**，相比普通 KV cache：

- 数据量小（如 DeepSeek-V3 的 latent ~512 维，远小于 `num_head × head_dim`）
- **TP 下不按 head 切分**：所有 TP rank 持有的 latent 内容完全相同
- 每个 TP rank 的 buffer 是**独立物理内存**，内容相同但物理位置不同

### 1.2 kv_pool 存取架构（简述）

`ascend_store` 连接器按 scheduler/worker 双角色拆分，KV cache 通过可插拔后端（mooncake/memcache/yuanrong）存取。每个 block 的 pool key 由 `PoolKey` 唯一标识，包含维度：`model_name / pcp_rank / dcp_rank / head_or_tp_rank / pp_rank / kv_cache_group_id / cache_role / cache_family / chunk_hash`。

其中 `head_or_tp_rank` 是 TP 维度的 key 索引，**这是 MLA 去重的核心字段**。

---

## 2. 现状：写侧已去重，读侧未去重

### 2.1 MLA 下 key 的 TP 维度归一（写/读共用）

[pool_worker.py:188-208](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L188-L208) 的 `_init_key_head_config`（scheduler 侧 [pool_scheduler.py:183-193](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L183-L193) 有同样逻辑）：

```python
if self.use_mla:
    self.num_kv_head = 1                    # MLA 强制 KV head 数 = 1
else:
    self.num_kv_head = model_config.get_total_num_kv_heads()

if self.num_kv_head < self.tp_size:
    self.put_step = self.tp_size // self.num_kv_head   # MLA: tp_size // 1 = tp_size
    self.head_or_tp_rank = self.tp_rank // self.put_step  # MLA: tp_rank // tp_size = 0（所有 rank）
else:
    self.head_or_tp_rank = self.tp_rank
    self.put_step = 1
```

**MLA 下的连锁结果**（以 `tp_size=8` 为例）：
- `num_kv_head = 1`
- `put_step = 8`
- `head_or_tp_rank = tp_rank // 8 = 0`（对 `tp_rank ∈ [0,7]` 全部都是 0）

也就是说，**所有 TP rank 构造出的 PoolKey 完全相同**。

### 2.2 写侧：只让 rank 0 写，其他 rank 跳过

[pool_worker.py:1017-1021](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1017-L1021)（`_process_save_for_layer_batch`）：

```python
# Only the first rank in each put_step group saves to the
# pool.  Other ranks in the same group share the same KV cache
# (e.g. MLA latent), so they skip save to avoid redundant writes.
if self.tp_rank % self.put_step != 0:
    return
```

**写侧已显式去重**：MLA 下 `put_step=tp_size`，只有 `tp_rank % tp_size == 0`（即 rank 0）真正执行 put，其他 rank 直接 return 跳过。

### 2.3 读侧：每个 rank 各自取，无 rank 间去重

读侧 `start_load_kv` 的同步 load 路径 [pool_worker.py:867-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L867-L980)：

```python
for group_id in load_group_ids:
    ...
    for start, end, key, _block_hash, block_id in \
        self.token_database.process_token_key_strings_with_block_ids(...):
        addr, size, block_id = self.token_database.prepare_value(...)
        key_list.append(key)
        addr_list.append(addr)
        ...
# 关键：用 tp_rank 做 circular_shift，说明每个 rank 都会执行 get
key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
...
ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
```

**读侧没有 `if self.tp_rank % put_step != 0: return` 之类的跳过**。每个 rank 都会：
1. 用 `head_or_tp_rank=0`（MLA 下）生成**相同的 key 列表**
2. 调 `self.m_store.get(...)` 各自把数据取到**自己的 buffer**（`addr_list` 是本 rank `register_buffer` 注册的地址）

> 注：`circular_shift`（[pool_worker.py:959-962](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L959-L962)）是负载均衡技巧——让不同 rank 从不同 key 开始取，避免所有 rank 同时打同一个后端桶，**不是去重**。

### 2.4 不对称的物理根因

| 操作 | 去重方式 | 物理约束 |
|------|----------|----------|
| 写 put | 只 rank 0 写，其他跳过 | 池子只需一份 latent，rank 0 写入即可 |
| 读 get | 每个 rank 各取到各 buffer | 各 rank buffer 物理独立，RDMA 必须写到各 rank 自己的地址 |

写可以省（池子只需要一份），读省不了（每个 rank buffer 独立，RDMA `get` 必须把数据搬到每个 rank 的 buffer 地址）。这是**物理约束**决定的现状，但并不意味着不能优化——优化空间在于"换一种搬运方式"。

---

## 3. 优化提案

### 3.1 核心思路

**"1 次 RDMA get + 1 次 TP 组内 broadcast" 替代 "N 次 RDMA get"**（N=tp_size）。

流程：
1. 只让 rank 0（或 `tp_rank % put_step == 0` 的 rank）从池子里 `get` 一份 latent 到自己的 buffer
2. rank 0 取完后，在 TP 组内做 broadcast（HCCL/NVLink 等片内高速互联）
3. 其他 rank 通过 broadcast 拿到 latent，写入自己的 buffer

### 3.2 思路成立的依据

写侧已经用了完全对应的思想（[pool_worker.py:1020](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1020)），读侧没对齐这个优化，是个合理的改进点。MLA 的 latent 内容各 rank 相同，broadcast 后语义正确。

---

## 4. 收益分析

### 4.1 后端读压力降 N 倍

MLA 下 TP=8 时，同一份 latent 被从池子 get 8 次。改成 rank 0 取 + broadcast，后端 get 次数降到 1 次。多请求并发、多 P/D 配对同时取时，后端（mooncake/memcache）的 RDMA 连接、key 查找、lease 获取压力直接 ÷N。

### 4.2 MLA latent 小，反而让优化更值

MLA 的核心价值就是 latent 压缩，数据量小意味着：单次 RDMA get 的**固定开销**（连接建立、key 路由、协议握手）占比高。8 个 rank 各取一小份，固定开销白白 ×8。这正是"去重取"收益最大的场景——**小数据 + 高频次，省的是固定开销不是带宽**。

### 4.3 layerwise GVA 模式下 lease 也省

[kv_transfer.py:1640-1647](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1640-L1647) 最后要 `batch_remove_lease`。N 个 rank 各持 lease 是 N 份，改后只需 rank 0 持 lease。

---

## 5. 成本与风险

### 5.1 引入 TP 组同步点（核心 trade-off）

当前每个 rank 独立异步取，**没有 rank 间等待**。改成 rank 0 取 + 广播后：
- rank 0 执行 get（跨节点 RDMA 延迟）
- 其他 rank idle 等待
- rank 0 取完 → broadcast（NVLink/HCCL 延迟）

**延迟上可能不降反升**：原来 8 rank 并行取 ≈ 1 次 RDMA 延迟；改后 ≈ 1 次 RDMA + 1 次 broadcast。但**吞吐上**后端压力 ÷8，多请求并发时整体吞吐显著改善。这是**延迟换吞吐**的 trade-off。

### 5.2 异步路径改造复杂

非 layerwise 异步 load（`load_async=True`）走 [pool_worker.py:910-913](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L910-L913) 的 `kv_recv_thread` 后台线程。在后台线程里插入 HCCL broadcast 需要：所有 rank 的 recv 线程**同步到同一点**才能做 collective，而线程时序本来就是为了解耦而设计的。这块改造比同步路径难得多。

### 5.3 是否有现成 TP broadcast 通道

kv_pool 模块目前在 worker 里用的是 `m_store`（后端 store），没看到现成的 TP 组通信原语。需要确认 vLLM/ascend 侧有没有可直接复用的 HCCL broadcast group（如 `get_tp_group().broadcast`）。如果没有现成的，要新建通信通道，工作量和风险都不小。

### 5.4 后端可能已有 batch 优化

mooncake/memcache 的 `get` 如果内部对同一 key 做了 batch 合并/去重，那 N 次小 get 的实际开销可能比想象的小，优化收益要打折扣。这点需要看后端实现确认。

### 5.5 异常处理路径复杂化

当前某个 rank get 失败只影响自己（[pool_worker.py:972-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L972-L980) 的 `record_failed_blocks`）。改成 rank 0 取后，rank 0 失败会影响整个 TP 组，需要额外的失败传播与回退机制。

---

## 6. 待验证关键点

优化是否划算，取决于以下事实，建议会前或会后优先验证：

| 验证点 | 倾向性影响 | 验证方式 |
|--------|-----------|----------|
| MLA latent 实际大小（buffer lengths） | 越小越值得做（固定开销占比高） | 看 `register_buffer` 注册的 lengths / 模型配置 |
| 后端 `get` 是否对同 key batch 优化 | 已优化则收益打折 | 看 mooncake/memcache backend 实现 |
| 是否有现成 TP broadcast 通道可复用 | 没有则改造成本陡增 | 搜 vLLM/ascend 的 `get_tp_group` / HCCL broadcast |
| load 路径主要走同步还是异步 | 异步路径改造复杂 | 看线上 config 默认值 |
| 线上 MLA 模型实际 TP 规模 | TP 越大收益越大 | 看部署 config |

---

## 7. 倾向性建议

**思路成立，且 MLA 场景尤其值得做**，但建议：

1. **优先在非 layerwise 同步路径试点**（改动小、时序清晰，作为 PoC）
2. **异步路径和 layerwise GVA 暂缓**（线程同步复杂度高，待 PoC 验证收益后再推进）
3. **关键是先确认有无现成 TP broadcast 通道**：
   - 有现成接口（如 `get_tp_group().broadcast`）→ 性价比高，建议推进
   - 没有要自己搭 → 重新评估工作量和风险
4. **收益要先量化**：在 PoC 阶段用 benchmark 测延迟/吞吐变化，特别是多请求并发场景，避免"理论上省了实际被 broadcast 延迟吃掉"

---

## 8. 结论

- **现状**：MLA 写侧已去重（只 rank 0 写），读侧未去重（每 rank 各取各的），存在不对称的优化空间
- **提案**：读侧改为 rank 0 取 + TP 组 broadcast
- **判断**：思路成立，MLA 场景（小数据 + 高频次）收益潜力大，但需先验证"现成 broadcast 通道"和"后端是否已 batch 优化"两个关键事实，再决定是否推进
- **优先级**：建议先做 PoC 在同步路径验证，再考虑推广

---

## 附录：关键代码位置索引

| 内容 | 文件 | 行号 |
|------|------|------|
| MLA 下 key TP 维度归一（worker） | pool_worker.py | [188-208](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L188-L208) |
| MLA 下 key TP 维度归一（scheduler） | pool_scheduler.py | [183-193](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L183-L193) |
| 写侧只 rank 0 put | pool_worker.py | [1017-1021](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1017-L1021) |
| 读侧 start_load_kv（同步路径） | pool_worker.py | [867-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L867-L980) |
| 读侧 circular_shift（负载均衡非去重） | pool_worker.py | [959-962](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L959-L962) |
| 读侧异步路径（kv_recv_thread） | pool_worker.py | [910-913](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L910-L913) |
| register_buffer（各 rank 独立 buffer） | pool_worker.py | [864](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L864) |
| layerwise lease 释放去重 | kv_transfer.py | [1640-1647](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1640-L1647) |
| PoolKey 定义（含 head_or_tp_rank） | config_data.py | [94-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L94-L124) |
| infer_tp_mismatch_info（MLA 禁用 sub-key） | config_data.py | [36-69](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L36-L69) |
| scheduler 生成 query key | pool_scheduler.py | [247-283](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L283) |
| GVA hit check 注释（MLA 共享 key） | pool_scheduler.py | [328-339](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L328-L339) |
