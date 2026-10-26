# 旧入口迁移：Recompute CPU Offload → PreemptOffload

当前基线见 [00](../00_source_baseline.md)。源码已从旧 `kv_pool/recompute_cpu_offload/` 迁到 `kv_pool/kv_offload/preempt_offload/`，注册名为 `PreemptOffloadConnector`。

完整分析见 [PreemptOffload：抢占时保存 KV，恢复时避免重复计算](preempt_offload.md)。

旧版本本文曾把它解释为“只保存 metadata、不保存完整 KV 的重计算路径”，该结论不适用于当前实现：`PreemptOffloadWorker` 创建 CPU KV tensors，并在抢占/恢复时实际执行 NPU→CPU / CPU→NPU copy。当前代码中仍出现的 recompute scheduler/注释指的是抢占重算场景，不是 metadata-only 存储语义。

本文件保留以兼容旧知识库链接，后续内容维护在新专篇。
