# vLLM Ascend KV Pool 源码梳理

本目录是 `vllm_ascend/distributed/kv_transfer/kv_pool` 的源码学习资料。目标不只是"知道代码是什么"，而是支撑对代码的深刻理解：每个组件的边界、每个设计决策的原因、每条不变量背后的竞态。按顺序读完主线，应能独立定位池化相关问题、评估改动影响面。

## 目录结构

```text
01_kv_pool_overview.md      整体定位：kv_pool 与 vLLM 各层的边界
ascend_store/               核心主线（外部 KV Store 路径，7 篇连续编号）
other_paths/                旁路（本地 offload / 重计算，与主线独立）
notes.md                    原始读码笔记（升格文档的信息源，保留作参考）
```

## 阅读顺序

### 第一层：定位

1. [01 整体定位与架构](01_kv_pool_overview.md)——kv_pool 是 connector 实现集合而非新 Scheduler；vLLM block 账本、connector 契约、传输数据面的三层边界。

### 第二层：ascend_store 主线（按序深读）

2. [ascend_store/01 总览](ascend_store/01_overview.md)——源码地图、一次请求的完整生命周期（命中→分配→metadata→load→save→完成回传）、运行模式矩阵、阅读路线。
3. [ascend_store/02 挂载、契约与控制面](ascend_store/02_connector_and_control_plane.md)——entry point 注册链路、connector 23 个方法全景、KVPoolScheduler 命中查询与状态管理、coordinator 的 hybrid 语义。
4. [ascend_store/03 Metadata 与 Layout](ascend_store/03_metadata_and_layout.md)——PoolKey/LayerPoolKey/RequestTracker/ReqMeta 四层描述语言、ChunkedTokenDatabase 地址模型、layerwise 物理布局。
5. [ascend_store/04 Worker Pipeline](ascend_store/04_worker_pipeline.md)——NPU cache 注册、block id 展开为地址、普通/layerwise/GVA/partial/TP mismatch 五路分流。
6. [ascend_store/05 存储模型、传输线程与 Backend](ascend_store/05_transfer_backend_storage.md)——HBM/DRAM 两层存储与两套地址账本、六类传输线程、三种 backend 能力差异、同步边界与错误传播。
7. [ascend_store/06 并发、同步与配置](ascend_store/06_concurrency_and_config.md)——三层完成语义（queue 完成 ≠ tensor 可读 ≠ 资源可复用）、attention fence、delayed_free、配置组合如何改变数据流。
8. [ascend_store/07 查询路径设计决策](ascend_store/07_lookup_path_design.md)——ZMQ/直连分叉的本质、减法/合并模型、buffer 复用轮转协议、两条正交的轴（最深入的一篇）。

### 第三层：旁路（可选，与主线独立）

9. [other_paths/kv_offload](other_paths/kv_offload.md)——native/simple 两条本地 CPU/NPU 搬运路径，与外部 store 的本质区别。
10. [other_paths/recompute_cpu_offload](other_paths/recompute_cpu_offload.md)——不保存完整 KV 的重计算恢复路径。

## 深度锚点（读完后应能回答）

```text
为什么非 layerwise 查询走 ZMQ 而 layerwise 直连？          -> ascend_store/07
为什么 layerwise 查询从 block 0 起且开复用后不信本地？      -> ascend_store/07
coordinator 什么时候存在、什么时候被闲置？                 -> ascend_store/02 §6.4
queue 完成、tensor 可读、block 可释放为什么是三个事件？    -> ascend_store/06
register_buffer 登记的为什么不是池本身？                   -> ascend_store/05 §1
connector 契约外的方法如何识别？                          -> ascend_store/02 §3.4
```

## notes.md 的定位

[notes.md](notes.md) 是走码时的原始笔记，ascend_store/07 等深读篇由它升格而来。内容有重叠但保留了走码过程中的原始推理链，可作交叉验证；正式学习以编号文档为准。

## 核心源码入口

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/`
- `vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/`

## 一句话总览

上层 connector 把请求级 KV transfer 转换成 pool 操作；`ascend_store` 负责元数据、调度和后端存取，offload 模块负责本地 CPU/NPU 搬运，重计算模块在无法直接恢复 KV 时提供替代路径。
