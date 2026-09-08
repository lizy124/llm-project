# Verification Report — vllm-ascend Issue #15842: TP Mismatch Strided KV I/O Regression

## 1. 问题概述

Issue **#15842** 报告：Prefill 与 Decode 的 TP（tensor parallel）不一致时，KV 池化出现回归——前缀缓存命中率掉到 0。

本报告记录在 165 节点容器 `pr15832_165` 内的完整实机复现过程与取证。

**结论先行：issue 真实存在。** 复现结果表现为 TP mismatch 下 KV 只写入一半 TP-rank 的子 key，`exists_count` 缺半，strided I/O 路径零触发。

## 2. 背景与根因假设

### 2.1 机制

TP mismatch 场景（Prefill TP=2 / Decode TP=1）：

- KV 头需跨 TP 拆分或合并，通过按 stride 拆分子 key 实现传输。
- 每个 `head_or_tp_rank` 子 key 各占一份，全部存在才构成完整 KV 组。

### 2.2 根因假设

KV 池化重构（PR #11444）后，**三处生产调用点丢失了 worker 传参**，导致 TP mismatch 的 strided I/O 路径永不触发：

1. `KVCacheStoreSendingThread` 构造时未传 `worker`
2. `KVCacheStoreRecvingThread` 构造时未传 `worker`
3. `start_load_kv()` 未路由到 `_load_kv_tp_mismatch()`

修复 PR #15835 尚未合并进 `main`。

## 3. 复现环境

| 项 | 值 |
|----|----|
| 节点 | 192.168.13.165（共享服务器，16 张 NPU） |
| 容器 | `pr15832_165`（`nightly-main-a3` 镜像，`/home` 挂载） |
| vLLM-Ascend HEAD | `f16a0fa46`（"Mooncake SSD offload in subprocesses"） |
| 修复 #15835 | **未合并**（`git log --grep=15835` 为空）→ bug 存活 |
| 模型 | `/mnt/weight/Qwen3-0.6B`（dense GQA，`num_kv_heads=8`） |
| 拓扑 | PD 分离，MultiConnector(P2P) + AscendStore(Mooncake) + proxy，Mooncake master 50088/9008 |
| NPU 分配 | prefill TP2 → device index 8,9；decode TP1 → device index 10 |
| KV 后端 | mooncake_master @ 127.0.0.1:50088 |

## 4. 代码取证（静态核对）

### 4.1 线程构造未传 worker

`pool_worker.py:724` 与 `pool_worker.py:739` —— 构造两个线程时，参数列表以 `self.enable_kv_events` / `invalid_block_ids=...` 结束，**未传 `worker=self`**。

而线程内部（`kv_transfer.py:623`、`kv_transfer.py:890`）定义了 `worker: Any = None` 参数，且其 `_handle_request` 有 TP mismatch 分支：

```python
# kv_transfer.py:666 (sending)
def _handle_request(self, req_meta: ReqMeta):
    if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
        self.worker._store_kv_tp_mismatch(req_meta)
    ...
# kv_transfer.py:916 (recving) —— 同构, 调 _load_kv_tp_mismatch
```

**路由代码存在，但因构造时 `worker=None`，分支永不触发。**

### 4.2 start_load_kv 未路由 TP mismatch

`pool_worker.py:1027` 起遍历 `metadata.requests`，只区分 `load_async → kv_recv_thread.add_request` 与普通同步路径，**无 `tp_mismatch` 分支**，`_load_kv_tp_mismatch` 仅定义、从未被调用。

### 4.3 结论

三处缺失实锤，与 issue 描述一致。

## 5. 实机部署过程

### 5.1 端口/资源冲突处理

- **Mooncake master 端口冲突**：50088/9008 曾被遗留 master 占用，清理后释放。
- **NPU 显存占用**：机上有遗留 DeepSeek-V2 服务（dev index 0-7）占满显存，改为使用空闲的 device index 8,9,10。
- **FabricMem 报错**：初次起 prefill 时 `FabricMemEngine` 报 `Unsupported option`（`ASCEND_ENABLE_USE_FABRIC_MEM=1` 触发），移除该环境变量后解决。

### 5.2 服务启动

两个服务均成功 `ready`，两侧 worker 均打印 TP mismatch detection：

- prefill(TP2)：`local_tp=2, peer_tp=1, effective_tp=2, local_heads_per_rank=4, effective_heads_per_rank=4, num_sub_keys=1`
- decode(TP1)：`local_tp=1, peer_tp=2, effective_tp=2, local_heads_per_rank=8, effective_heads_per_rank=4, num_sub_keys=2`，并打印 `TP mismatch strided I/O: per_token_bytes=2048, sub_size_bytes=1024`

## 6. 验证判据与结果

### 6.1 判据定义

修复前（bug 存活）预期看到：

1. strided I/O 路径（`_store_kv_tp_mismatch` / `_load_kv_tp_mismatch`）零触发
2. decode KV pool put 只写 `head_or_tp_rank:0` 一份
3. KV pool lookup `exists_count` 缺半
4. `cached_tokens=0`（前缀命中率为 0）

### 6.2 判定证据

**① strided I/O 路径触发次数 = 0（核心铁证）**

```
Store/Load tp_mismatch 调用总数: 0
prefill KV pool put (普通路径): 3
```

**② decode 只写 head_or_tp_rank:0 一份**

prefill(TP2) 两侧 worker 各生成 key，但实际 put 落库仅 `head_or_tp_rank:0`：

```
KV pool put key 分布: head_or_tp_rank:0 × 1（head_or_tp_rank:1 缺失）
```

**③ exists_count 缺半**

```
首轮 lookup: exists_count=0/2  exists_sample=[0,0]
二轮 lookup: exists_count=1/2  exists_sample=[1,0]
            （head_or_tp_rank:0 ✓ 存在 / head_or_tp_rank:1 ✗ 缺失）
```

**④ 两轮请求均 cached_tokens=0**

```
round1: cached_tokens=0（首次 prefill，符合预期）
round2: cached_tokens=0（本应复用的前缀未命中）
```

### 6.3 数值对应说明

本复现用 TP2/1（`multi_tp_keys=2`），表现为 `exists_count=1/2`；issue 用 TP4/1（`multi_tp_keys=4`）表现为 `exists_count=4/8`。**两者机制完全一致**，数值差异仅来自 TP 规模。

## 7. 结论

- **Issue #15842 属实且在实机无误复现。**
- 机制：PR #11444 重构后三处接线丢失（线程构造不传 `worker`、`start_load_kv` 未路由），strided GET/PUT 不可达，TP mismatch 下 KV 只写一半 TP-rank 子 key，`exists_count` 缺半，前缀缓存命中率 0。
- 修复方向：#15835 补回三处 worker 传参与路由。

## 8. 复现操作清单（可复跑）

> 服务器工作区 `/home/lizhongyang/map_165_15842`，容器 `pr15832_165`，`/home` 跨容器共享挂载。

```bash
# 0) 前置: 确保 mooncake master 存活 (50088/9008)
# 1) 启动 prefill (TP2, dev 8,9)
docker exec pr15832_165 bash /home/lizhongyang/map_165_15842/test/01_prefill.sh
# 2) 等待 prefill ready 后启动 decode (TP1, dev 10)
docker exec pr15832_165 bash /home/lizhongyang/map_165_15842/test/02_decode.sh
# 3) 双端 ready 后启动 proxy
docker exec pr15832_165 bash /home/lizhongyang/map_165_15842/test/03_proxy.sh
# 4) 发两轮请求
docker exec pr15832_165 bash /home/lizhongyang/map_165_15842/test/04_requests.sh
# 5) 取证
docker exec pr15832_165 bash /tmp/cv.sh
```

关键日志路径：`/home/lizhongyang/map_165_15842/run/{prefill,decode}/server.log`、`run/requests/requests.log`。