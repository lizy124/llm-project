# UCM Connector：本地包装与外部引擎的边界

基线见 [00](../00_source_baseline.md)。本地源码为 `kv_pool/ucm_connector/connector.py`，注册名 `UCMConnector`，包装类 `UCMConnectorV1`。

## 1. 与 AscendStore 的关系

UCM 是独立 connector 路径，不是 `ascend_store/backend_map` 中的第四种 backend。它直接导入外部包 `ucm.integration.vllm.ucm_connector.UCMConnector` 并保存为 `_ucm_engine`。

```text
vLLM connector lifecycle
  -> UCMConnectorV1
  -> 外部 UCM dispatcher / selected inner connector
  -> 由对应引擎执行缓存策略与传输
```

本地实现继承 `KVConnectorBase_V1`、`SupportsHMA`，转发 scheduler、worker、metadata、stats/events 等接口。不能由包装层推断其远端介质、淘汰策略或内部并发协议。

## 2. Reserved hooks 的转发问题

外部 UCM dispatcher 本身可能继承了基类 no-op，却未显式声明某个保留 hook；直接调用 dispatcher 可能绕过真正实现该 hook 的 inner connector。

`_get_ucm_delegate_for(name)`：
1. dispatcher 类型显式定义该名字时，优先 dispatcher。
2. 否则若存在 inner `connector`，优先 inner。
3. `_call_ucm_reserved_hook()` 对可调用 hook 继续转发。

这用于 shutdown、host transfer ops、preemption 等接口。分析时不能把“包装对象有这个属性”当成“真正执行者收到了 hook”。

## 3. 调试与维护入口

先检查外部 UCM 包是否可导入、版本是否与当前 vLLM 契约配对，再检查 metadata bind/clear、reserved hooks 和 inner delegate。若问题在外部实现，需要继续阅读实际安装的 UCM 源码；本次仅依据用户指定的两个仓库，不将未提供的外部代码行为写成已验证结论。

本地 CPU offload 的设备适配见 [kv_offload](kv_offload.md)；Store 的外部协议见 [05](../ascend_store/05_transfer_backend_storage.md)。
