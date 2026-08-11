# [Perf] 非-GVA layerwise prefetch 默认值与 submit 逻辑【重磅】

> 编号：kv-06 | 维度：Perf | 严重程度：高 | 建议优先级：P1/P2（需实测）
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

layerwise 架构的核心价值是"I/O 传输与 attention 计算重叠"，但**非-GVA 路径默认 `layerwise_prefetch_layers=1`**——本层 load 在 `wait_for_layer_load` 调用时才提交并立即等待，**load 与 attention 完全串行，重叠设计完全失效**。对比 GVA 路径默认 prefetch=8，差距巨大。

更细致的限制（已核实）：`submit_count = self.num_prefetch_layers if self.current_layer == 0 else 1`——**只有第 0 层用 num_prefetch_layers，其他层固定提交 1 层**。即使配 prefetch=8，首层之后每层只提交 1 层预取。

**影响**：非-GVA layerwise 模式下，每层 attention 前计算线程都阻塞等网络 I/O，NPU 计算流排空空闲，num_layers 次串行等待，等于把传输延迟加到 forward 延迟上。可能是线上最大的单点性能损失。

## 任务

1. **先实测**：在不同 prefetch 配置下测端到端延迟 / NPU 计算流利用率，量化收益
2. **最低成本**：默认值改 2+（GVA 路径已默认 8，非-GVA 应对齐），配合放开首层限制
3. **更深**：让 `submit_count` 在所有层都遵循 `num_prefetch_layers`，而非只首层

## 验收标准

### 1. 功能正确性
- 改动后非-GVA layerwise 模式输出不变（精度无回归）
- 不引入死锁 / 资源越界

### 2. 性能验证
- NPU 计算流利用率提升（给出 nsight / profiler 数据）
- 端到端延迟下降幅度（prefetch=1/2/4/8 对比曲线）
- 不同 num_layers / 序列长度下的收益

### 3. 交付件
- PR + 设计说明 + 性能数据曲线 + 单测

## 证据

- 默认值：[pool_worker.py:425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L425) `int(self._extra_config.get("layerwise_prefetch_layers", 1))`
- 提交逻辑：[pool_worker.py:1683](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1683)
- 等待逻辑：[pool_worker.py:1691-1707](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1691-L1707)

## 重点关注

- 预取窗口放大需评估 HBM / buffer 占用上限
- 与 kv-16（gate 对非复用预取层过度同步）联动：预取层增多时 gate 行为更关键
- 与 kv-14（最后一层 save 同步等待）同属"同步等待"主题

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数 + TP）
- 关联任务池：#9079
- 验收人：@赵鹏博
