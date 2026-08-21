# PR CZ validation

这套目录整理的是 `vllm-ascend` 里 ZMQ lookup suffix 变更的端到端验证材料。

## 内容

- [validation_overview.md](validation_overview.md) - 验证目标、环境和结论摘要
- [launch_scripts.md](launch_scripts.md) - 复现实验的拉起脚本和运行方式
- [process_data.md](process_data.md) - 关键过程数据、性能数值和 e2e 结果
- [key_logs.md](key_logs.md) - 关键日志摘录
- [analysis_conclusion.md](analysis_conclusion.md) - 结果分析和最终结论
- [benchmark_zmq_lookup_payload.py](benchmark_zmq_lookup_payload.py) - 协议级对比脚本
- [e2e_ascend_store_suffix_lookup_probe.py](e2e_ascend_store_suffix_lookup_probe.py) - 端到端探针脚本

## 一句话结论

协议级验证已经证明 `suffix` 能显著减少 lookup payload 和 RTT；真实 e2e 里已经抓到 `computed=384` 与 `hit_tokens=384` 的命中证据，但端到端时间收益仍然被模型前向和调度开销淹没，当前观察到的整体收益约 53 ms，约 3.2%。
