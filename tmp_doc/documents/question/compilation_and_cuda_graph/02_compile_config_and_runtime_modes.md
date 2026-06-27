# 02. CompileConfig 和 runtime mode 如何决定执行路径？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/config.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/forward_context.py`

本问题关注：编译和 CUDA graph 相关配置如何影响每轮执行路径选择。

---

## 1. 一句话回答

CompileConfig 决定系统是否启用 torch.compile / CUDA graph，以及 capture 哪些 shape；runtime mode 决定当前这一轮实际走 eager、capture 还是 replay。

---

## 2. 配置项占位

后续补充：

```text
- compile level；
- use_cudagraph；
- cudagraph capture sizes；
- max capture batch size；
- dynamic shape；
- splitting / full graph；
- compilation backend；
- enforce eager；
- cudagraph warmup runs。
```

---

## 3. runtime mode 占位

```text
可能的运行形态：

- eager：普通 PyTorch forward；
- compiled：torch.compile 后 forward；
- capture：初始化或 warmup 中捕获 graph；
- replay：运行时复用已 capture graph；
- fallback：条件不满足时退回 eager / compiled。
```

---

## 4. 决策因素占位

```text
- 当前 batch size / token 数是否命中 capture size；
- 是否需要 padding；
- 是否包含不支持 graph 的功能；
- 是否处于 warmup / capture 阶段；
- attention metadata 是否支持 capture；
- LoRA / multimodal / pooling / spec decode 是否影响 shape；
- PP / TP / DP 是否有额外限制。
```

---

## 5. 一句话总结

```text
配置决定“能不能 graph”，runtime mode 决定“这一轮实际怎么跑”。
```
