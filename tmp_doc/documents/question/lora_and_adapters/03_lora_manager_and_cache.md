# 03. LoRA manager 如何加载、缓存和卸载 adapter？

源码位置：

- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/models.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/adapter_commons/`

本问题关注：Worker 侧 LoRA manager 如何管理 adapter 生命周期，包括加载、缓存、pin、卸载和查询。

---

## 1. 一句话回答

LoRA manager 是 Worker 侧 adapter 的资源管理器：

```text
它负责把 LoRA checkpoint 加载成内存中的 adapter，
维护 adapter cache，
控制哪些 adapter 常驻或可淘汰，
并向 LoRA layer 提供可执行权重。
```

---

## 2. manager 职责占位

```text
- add_lora；
- remove_lora；
- pin_lora；
- list_loras；
- 检查 adapter 是否已加载；
- 根据容量限制淘汰 adapter；
- 将 adapter 映射到内部 lora id；
- 维护 loaded / active / pinned 状态。
```

---

## 3. adapter cache 占位

需要梳理：

```text
- 最大 LoRA 数量；
- 最大 LoRA rank；
- CPU / GPU 缓存；
- pinned adapter；
- eviction 策略；
- adapter 重复加载处理；
- 多 worker 下的一致性。
```

---

## 4. 和请求执行的关系

```text
请求中有 LoRARequest
  → Worker 确认 adapter 已加载
  → ModelRunner 激活当前 batch LoRA
  → LoRA layer 使用 manager 中的权重
```

---

## 5. 一句话总结

```text
LoRA manager 负责“adapter 资源”，ModelRunner 负责“本轮用哪些 adapter”。
```
