# 02. Python 层如何调用自定义算子？

源码位置：

- `vllm/vllm/_custom_ops.py`
- `vllm/vllm/_C/`
- `vllm/csrc/`
- `vllm/vllm/platforms/`

这个问题关注：Python 层如何通过 wrapper、torch extension、torch.ops 或 fallback 路径调用底层 native op，以及这些入口如何屏蔽不同平台、backend、dtype、shape 的差异。

---

## 1. 一句话回答

Python 层通常不直接操作 CUDA kernel，而是通过 **薄 wrapper + backend dispatch** 调用 native op。

可以先记成：

```text
Python model layer
  → vLLM op wrapper
  → torch.ops / extension binding / Triton function / torch fallback
  → concrete kernel
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. _custom_ops.py 暴露哪些常用算子入口；
2. Python wrapper 如何做参数整理和 fallback；
3. torch.ops 与 vLLM C++/CUDA extension 如何绑定；
4. Triton kernel 如何从 Python 侧发起；
5. 平台差异如何通过 current_platform / env flags 影响路径；
6. import 失败、op 不存在、dtype 不支持时如何回退或报错。
```

---

## 3. 需要串起来的主线

```text
model_executor/layers/*
  → _custom_ops.py 或 backend-specific module
  → native extension / Triton / torch implementation
  → output tensor
```

---

## 4. 后续补充重点

```text
- native op 注册机制；
- custom op wrapper 的命名和职责；
- fallback 的判断条件；
- 如何确认实际调用路径；
- 和 build / install / platform capability 的关系。
```
