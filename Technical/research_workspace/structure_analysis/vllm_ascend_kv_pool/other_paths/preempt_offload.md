# PreemptOffload：抢占时保存 KV，恢复时避免重复计算

基线见 [00](../00_source_baseline.md)。当前源码是 `kv_offload/preempt_offload/`，配置名 `PreemptOffloadConnector`，实现类 `PreemptOffloadConnectorV1`。

旧文件名和部分代码注释中的 recompute 指向“原本会重算的抢占场景”，不能解释为“只保存 token/metadata、不保存 KV”。Worker 确实分配 `cpu_kv_caches` 并执行 tensor copy。

## 1. 为什么是独立路径

普通外部 prefix store 依赖完整 hash 对象；被抢占请求可能有：
- 尚未形成完整 hash 的 partial block。
- SWA window 内的稀疏有效 blocks。
- Mamba running/temporal/conv 状态。
- speculative 接受偏移和需裁剪的 scratch tail。

这些状态不能简单转换成外部完整 prefix key。PreemptOffload 保存 request 的真实可恢复 KV/state 及 block 映射，服务于该请求恢复。

## 2. 组成与调用方

| 文件 | 职责 |
|---|---|
| `preempt_offload_connector.py` | role 分流、容量解析、通用接口与扩展 hooks |
| `manager.py` | CPU block pool、抢占状态、store/load specs、事件聚合 |
| `worker.py` | CPU tensor、NPU streams/events、真实 D2H/H2D、Mamba conv 修正 |
| `metadata.py` | 两方向 block 映射、event ids、MambaConvLoadMeta、完成计数 |

还必须跟读 `vllm_ascend/core/recompute_scheduler.py`：它在抢占前调用 `update_state_before_preempt()`。这是 Ascend 扩展 hook，不能仅看 vLLM 基类就假定普通上游 Scheduler 会自动调用。

`AscendMultiConnector` 识别 `has_preempted_request()`，优先恢复此状态，防止其他完整 prefix connector 抢先接管而丢掉 partial tail。

`RecomputeScheduler._preempt_or_recompute()` 是 best-effort 接入：offload 不可用或失败时，通常把请求交回 P 重做 prefill；启用相关 fine-grained TP split 时走 abort 分支。配置了 connector 不等于每次抢占都能成功 offload。

## 3. 两本 block 账本

vLLM 管 NPU blocks；`PreemptOffloadScheduler` 建 CPU block pool 与相应 CPU cache config。CPU 页数来自容量、原 GPU tensor 几何和 host ratio，必须与 Worker 分配一致。

```text
被抢占 request
  -> group-wise NPU block ids
  -> 分配 CPU blocks
  -> 保存 NPU↔CPU 映射、computed tokens、state offsets
```

CPU 缓存可以选择是否维护可复用 hash；但主要命中入口先查 `_preempted_req_states[request_id]`，不能把它描述成一个通用远端共享 KV Store。

## 4. 抢占保存必须早于覆盖

```text
RecomputeScheduler 决定抢占
  -> update_state_before_preempt(request, blocks, num_computed_tokens)
  -> manager 形成 preempt_store specs / event
  -> metadata 交 Worker
  -> handle_preemptions()
  -> 必要时 flush
  -> 同步 D2H 保存真实 KV/state
  -> 新请求的 zero / forward 才可覆盖原 NPU block
```

Worker 注释明确指出同一 step 就可能复用被抢占 block，所以 D2H 必须位于输入准备的清零/状态更新之前。

`save_kv_layer()` / `wait_for_save()` 在此 connector 中是 no-op；实际保存入口是 `handle_preemptions()`。不能仅看到通用 save 方法为空就认定它没有存 KV。

## 5. 保存完成后才可恢复

`get_num_new_matched_tokens()`：
- 没有该 request 状态 → `(0, False)`。
- D2H store event 尚未收齐所有 Worker → `(None, False)`，表示暂时不能确定/恢复，不是普通 miss。
- ready → 返回 `restorable_tokens - local_tokens` 的非负贡献，并标记 async restore。
- 已无需恢复 → 清理相应 CPU request 状态。

`PreemptOffloadWorkerMetadata.completed_store_events` 在单轮 aggregate 后，manager 继续跨轮累积；达到预期 Worker 数才把 state 标为 ready。一个 rank copy 完成不证明该请求在所有 ranks 都可恢复。

## 6. Allocation 后加载到新地址

```text
vLLM 为恢复请求分配新的 NPU blocks
  -> update_state_after_alloc
  -> manager 将 CPU blocks 映射到新 NPU blocks
  -> preempt_load metadata
  -> start_load_kv: 同步提交/等待当前 H2D helper
  -> forward stream 的必要等待
  -> get_finished 报接收完成
  -> manager 清 CPU 状态及临时引用
```

逻辑上使用外部 async 恢复调度协议，但当前 Worker `_submit_transfer(..., sync=True)` 的执行并不意味着 H2D 与本次 forward 无约束并行。必须分别看 Scheduler async 标志与设备 helper 的同步行为。

SWA 跳过 null blocks 并保持 window 偏移；MTP tail 不能按整个 block list 生搬；保存的有效 computed token 水位不能仅因页对齐就扩大。

## 7. Mamba temporal 与 conv state 不可混同

普通 block copy 对 temporal/SSM state 仍然有效。conv state 可能保存在 running-state block，需要根据 accepted speculative-token offset 做历史切片修正。

`MambaConvLoadMeta` 明确给出：
- `gpu_block_id`。
- `cpu_block_id`。
- `source_offset`。

Worker `_copy_mamba_conv_loads()` 在正常 H2D 后只修正相应 conv 历史，不能用这份 conv 切片覆盖其他状态。不同 Mamba backend 的状态角色判定有显式分支，未知语义会拒绝。

## 8. 容量与开关

| 配置 | 默认/优先级 | 约束 |
|---|---|---|
| `cpu_bytes_to_use_per_rank` | 优先于总量 | 有效 per-rank bytes 必须为正 |
| `cpu_bytes_to_use` | 无 per-rank 时除以 world size | 同上 |
| `offload_host_memory_ratio` | 1；无显式 bytes 时推导 CPU capacity | 正有限数 |
| `enable_offload_prefix_caching` | connector 默认 False | 必须为 bool |

manager 构造函数自身有其默认参数，但经 connector 创建时应以 connector 传值为准，不能从内部默认值推断用户默认行为。

`supports_divergent_local_hybrid_hits` 返回 True，Scheduler 本地命中处理需保留 hybrid 差异语义。

## 9. 请求结束、reset 与释放

- ready 状态结束可直接清 CPU request blocks。
- store 尚未 ready 时结束，先标 `finished`，等待保存事件收齐后再清。
- restore 完成清 load event 映射、相关临时 NPU 引用和 CPU blocks。
- `reset_cache()` 在有 pending transfers/request state 时返回 False，避免清空在途状态。
- `request_finished*()` 当前返回 False，不采用 request delayed-free 返回值。

测试重点见 `tests/ut/kv_offload/test_preempt_offload_connector.py`：容量、SWA offsets、Mamba conv、MTP tail、event readiness、pending/reset、view 去重，以及 `tests/ut/core/test_recompute_scheduler.py` 的实际调度 hook。

旧入口迁移说明见 [recompute_cpu_offload.md](recompute_cpu_offload.md)；与 native/simple 的差异见 [kv_offload.md](kv_offload.md)。
