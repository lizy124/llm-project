# validation overview

## 目标

验证 ZMQ lookup 的 `lookup_hash_mode=suffix` 改动是否：

1. 在协议层减少 scheduler -> worker 的 lookup payload；
2. 在真实服务里能命中 HBM prefix 后的 suffix lookup 路径；
3. 返回的 hit tokens 与 full 模式保持一致；
4. 端到端有可见收益。

## 环境

- 工作树: `/tmp/vllm-ascend-pr`
- 分支: `test/zmq-lookup-payload-omit`
- 模型: `/mnt/weight/Qwen3-0.6B`
- 设备: Ascend NPU
- 关键配置:
  - `VLLM_BATCH_INVARIANT=1`
  - `ASCEND_RT_VISIBLE_DEVICES=0`
  - `lookup_hash_mode=suffix`
  - `backend=mooncake`

## 结论摘要

- 协议级验证通过。
- e2e 探针确认真实服务里出现了 `computed=384` 的 HBM prefix 命中。
- 长 suffix 场景下，真实 lookup 返回 `hit_tokens=384`。
- 整体端到端时间收益可见，但不大，约 53 ms，约 3.2%。

## 相关文件

- [launch_scripts.md](launch_scripts.md)
- [process_data.md](process_data.md)
- [key_logs.md](key_logs.md)
- [analysis_conclusion.md](analysis_conclusion.md)
