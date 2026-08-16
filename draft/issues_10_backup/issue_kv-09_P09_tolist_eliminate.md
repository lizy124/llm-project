# [Perf] 消除 `batch_copy` 前的 `.tolist()`【隐藏拷贝】

> 编号：kv-09 | 维度：Perf | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

layerwise GVA 路径用 numpy 向量化精心构建 `addr_arr/size_arr/gvas_arr`，但在传给 C++ 扩展 `batch_copy` 前，用 `.tolist()` 把每个 ndarray 转回 Python list——为每个元素新建一个 PyLong 对象，**完全抵消了 numpy 向量化的内存效率**，在 C++ 边界前引入一次 O(N) 的 Python 对象分配 + 拷贝。每 layer save+load 各一次，3 个数组 × 2 = 6 次 tolist / forward。

```python
res = self.m_store.store.batch_copy(
    split_gvas.tolist(),    # ndarray → list[PyLong]
    split_addrs.tolist(),   # ndarray → list[PyLong]
    split_sizes.tolist(),   # ndarray → list[PyLong]
    direction,
)
```

## 任务

让 C++ 扩展 `batch_copy` 支持 buffer protocol / `memoryview` / 直接传 ndarray，避免 `.tolist()`。若 C++ 侧改动成本高，可退而传 `ctypes` 数组或 `array.array`。

## 验收标准

### 1. 功能正确性
- 改动后 `batch_copy` 行为与改动前一致（数值 / 顺序 / 方向）
- 现有单测全绿

### 2. 性能验证
- 消除 tolist 后，GVA 路径每 layer 的 Python 对象化开销下降（profile 对比）
- 长 prompt（blocks 大）下端到端收益

### 3. 交付件
- PR（含 C++ 扩展改动）+ 设计说明 + 性能数据 + 单测

## 证据

- [kv_transfer.py:477-481](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L477-L481)

## 重点关注

- **依赖 C++ 扩展支持 buffer protocol，需跨团队协调**——这是本 issue 的主要成本来源
- 评估替代方案（ctypes / array.array）作为低改动 fallback
- 改动需兼容 mooncake / memcache 两个 backend 的 batch_copy 调用

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
