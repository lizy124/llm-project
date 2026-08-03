# KV Pool 后端与 Connector 简表

## 结论

`vllm_ascend/distributed/kv_transfer/kv_pool` 里需要区分两类东西：

- `ascend_store/backend` 保留公共后端接口和 `backend_map`，真正的池化后端实现已拆到 `ascend_store` 下的同级目录，目前 3 个：`mooncake_backend`、`memcache_backend`、`yuanrong_backend`。
- `lmcache_ascend_connector`、`ucm_connector` 已移入 `ascend_store` 下作为同级 connector 目录；`simple_cpu_offload`、`recompute_cpu_offload` 仍在 `kv_pool` 下，属于 KVConnector / offload 方案，不是 `ascend_store.backend.backend_map` 注册的池化后端。
- `ascend_store/store_utils` 放 AscendStore 的公共 connector、调度、worker、metadata、transfer、coordinator 逻辑。

## 当前 ascend_store 结构

```text
kv_pool/ascend_store/
  backend/
    __init__.py
    backend.py
  mooncake_backend/
    __init__.py
    mooncake_backend.py
  memcache_backend/
    __init__.py
    memcache_backend.py
  yuanrong_backend/
    __init__.py
    yuanrong_backend.py
  lmcache_ascend_connector/
    __init__.py
    lmcache_ascend_connector.py
  ucm_connector/
    __init__.py
    ucm_connector.py
  store_utils/
    __init__.py
    ascend_store_connector.py
    config_data.py
    coordinator.py
    kv_transfer.py
    pool_scheduler.py
    pool_worker.py
```

## 池化后端

来源：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py`

| backend | 实现类 | 文件 | 说明 |
|---|---|---|---|
| `mooncake` | `MooncakeBackend` | `ascend_store/mooncake_backend/mooncake_backend.py` | AscendStore 的 Mooncake 后端 |
| `memcache` | `MemcacheBackend` | `ascend_store/memcache_backend/memcache_backend.py` | AscendStore 的 Memcache 后端 |
| `yuanrong` | `YuanrongBackend` | `ascend_store/yuanrong_backend/yuanrong_backend.py` | AscendStore 的 Yuanrong 后端 |

## store_utils

`store_utils` 是 AscendStore 的公共实现层，不属于某个具体后端。

| 文件 | 角色 |
|---|---|
| `store_utils/ascend_store_connector.py` | `AscendStoreConnector` 主入口，对接 vLLM KVConnector 接口 |
| `store_utils/config_data.py` | key、metadata、request/transfer 数据结构与通用工具 |
| `store_utils/kv_transfer.py` | KV 传输线程、layerwise 传输、失败 block 记录等逻辑 |
| `store_utils/coordinator.py` | 外部缓存命中、hybrid cache mask 和可达性协调 |
| `store_utils/pool_scheduler.py` | scheduler 侧状态管理、lookup、metadata 构造 |
| `store_utils/pool_worker.py` | worker 侧 backend 初始化、KV cache 注册、load/save 执行 |

## lmcache / ucm / yuanrong 代码量

| 类型 | 文件 | 行数 | 说明 |
|---|---:|---:|---|
| `lmcache` | `kv_pool/ascend_store/lmcache_ascend_connector/lmcache_ascend_connector.py` | 6 | 极薄封装：导入 `lmcache_ascend`，复用 vLLM 的 `LMCacheConnectorV1` |
| `ucm` | `kv_pool/ascend_store/ucm_connector/ucm_connector.py` | 325 | `UCMConnectorV1` 适配层，主要把 vLLM KVConnector 接口委托给外部 `ucm.integration.vllm.ucm_connector.UCMConnector` |
| `yuanrong` | `kv_pool/ascend_store/yuanrong_backend/yuanrong_backend.py` | 239 | 具体后端实现：环境配置、key 规范化、buffer 注册、`exists/get/put` |

合计约 570 行。

## simple_cpu_offload

文件：`kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py`

不是池化后端，而是 Ascend/NPU 版 CPU offload KVConnector。

特点：

- 继承 vLLM 上游 `SimpleCPUOffloadConnector`。
- scheduler 侧基本复用上游实现。
- worker 侧替换为 `SimpleCPUOffloadNPUWorker`。
- 用于把 KV cache 在 NPU 和 CPU 之间搬运，适配 `torch.npu` stream/event、NPU 拷贝实现。
- 当前文件约 61 行，主要是适配和替换 worker。

## recompute_cpu_offload

目录：`kv_pool/recompute_cpu_offload/`

也不是池化后端，而是一个面向 recompute/preemption 场景的 CPU KV 暂存 Connector。

作用：

- 请求被 preempt 前，把已计算 KV block 从 NPU/GPU KV cache 保存到 CPU。
- 请求恢复时，再把 CPU 上的 KV block 拷回 NPU/GPU。
- 目标是减少被抢占请求恢复时的重复计算。

主要组件：

| 文件 | 角色 | 行数 |
|---|---|---:|
| `recompute_cpu_offload_connector.py` | `RecomputeCPUOffloadConnectorV1`，对接 vLLM KVConnector 接口 | 247 |
| `manager.py` | `RecomputeCPUOffloadScheduler`，scheduler 侧状态管理、CPU block 分配、preempt/load 事件管理 | 631 |
| `worker.py` | `RecomputeCPUOffloadWorker`，worker 侧 CPU/NPU KV block 拷贝 | 320 |
| `metadata.py` | scheduler/worker 间 metadata 定义 | 53 |

配置点：

- 默认 CPU 容量：8GB。
- `cpu_bytes_to_use`：总 CPU offload 容量。
- `cpu_bytes_to_use_per_rank`：每 rank 容量，若配置则覆盖按 world size 平均后的值。
- `enable_offload_prefix_caching`：是否启用 offload prefix caching。

## 分类汇总

| 名称 | 类型 | 是否 AscendStore 池化后端 | 主要定位 |
|---|---|---:|---|
| `mooncake` | backend | 是 | 外部 KV pool 后端 |
| `memcache` | backend | 是 | 外部 KV pool 后端 |
| `yuanrong` | backend | 是 | 外部 KV pool 后端 |
| `lmcache_ascend_connector` | connector | 否 | LMCache connector 薄封装 |
| `ucm_connector` | connector | 否 | UCM connector 适配层 |
| `simple_cpu_offload` | connector/offload | 否 | 简单 CPU KV offload，Ascend worker 适配 |
| `recompute_cpu_offload` | connector/offload | 否 | preemption/recompute 场景下的 CPU KV 暂存与恢复 |
