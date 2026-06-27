# 10. LoRA 生命周期和控制接口如何工作？

源码位置：

- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/entrypoints/`

本问题关注：add_lora、remove_lora、pin_lora、list_loras 等控制接口如何从 Engine / Executor 传到 Worker。

---

## 1. 一句话回答

LoRA 生命周期由控制面管理：Executor 接收控制请求并广播到 Worker，Worker 通过 LoRA manager 加载、卸载、pin 或查询 adapter。

---

## 2. 控制接口占位

```text
add_lora：
  加载一个 adapter。

remove_lora：
  卸载一个 adapter。

pin_lora：
  标记 adapter 常驻，避免被淘汰。

list_loras：
  查询当前已加载 adapter。
```

---

## 3. 主链路占位

```text
Engine / API control call
  → Executor collective_rpc
  → Worker method
  → LoRA manager
  → adapter cache 更新
```

---

## 4. 并发和安全占位

需要梳理：

```text
- 正在被请求使用的 LoRA 能否 remove；
- pin adapter 是否占用容量；
- 多 worker 加载失败如何回滚；
- add_lora 与正在执行 batch 的关系；
- shutdown 时如何清理 adapter。
```

---

## 5. 一句话总结

```text
LoRA 生命周期控制是执行层控制面能力，不属于每轮 token 调度逻辑。
```
