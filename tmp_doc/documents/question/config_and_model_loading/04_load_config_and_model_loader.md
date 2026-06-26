# 04. LoadConfig 如何决定模型加载方式？

源码位置：

- `code/vllm/vllm/config/load.py`
- `code/vllm/vllm/model_executor/model_loader/__init__.py`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py`
- `code/vllm/vllm/model_executor/model_loader/loader.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/model_loader/weight_utils.py`

本问题关注：`LoadConfig` 如何决定使用哪种 loader、从哪里查找权重、如何处理 safetensors / PyTorch bin / sharded state / dummy / tensorizer / bitsandbytes 等加载路径。

---

## 1. 一句话回答占位

占位：后续补充 `LoadConfig` 描述“权重从哪里来、用什么格式加载”。

```text
LoadConfig
  → get_model_loader()
  → BaseModelLoader 子类
  → get_model()
  → initialize_model()
  → load_weights()
```

---

## 2. LoadConfig 关心什么占位

```text
- load_format；
- download_dir；
- model_loader_extra_config；
- ignore_patterns；
- pt_load_map_location；
- safetensors / npcache / dummy / tensorizer / bitsandbytes 等格式；
- 是否跳过或延迟加载某些权重；
- 权重文件查找和过滤策略。
```

---

## 3. ModelLoader 职责占位

```text
BaseModelLoader：
  loader 抽象基类。

get_model_loader：
  根据 LoadConfig 选择具体 loader。

get_model：
  对外统一模型加载入口。

initialize_model：
  根据 ModelRegistry 解析出的模型类实例化模型。

weight_utils：
  提供权重文件查找、迭代、safetensors 读取等工具。
```

---

## 4. 后续待补源码证据

占位：补充 `LoadConfig` 字段、`get_model_loader()` 分支、`BaseModelLoader.load_model()` 主流程。
