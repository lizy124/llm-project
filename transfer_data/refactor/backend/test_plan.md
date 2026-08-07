# PR 13354 测试计划

## 目标
验证 `attention_fence` 迁移到 `ascend_store` 后，导入路径、UT 和端到端推理链路不受影响。

## 验证范围
1. import + UT
   - import 新路径：`attention_fence`、`metadata`、`pool_worker`、`kv_transfer`、`backend.base`
   - 跑 `tests/ut/distributed/ascend_store -q`
   - 补跑 `tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py -q`
   - 视环境补跑受影响的 attention UT

2. baseline 端到端
   - 复用 `kvpool/test_script/03_start_server_baseline.sh`
   - 用 `05_send_requests.py` 发送普通请求
   - 证明普通 attention 推理路径未被影响

3. KV Pool 端到端
   - 复用 `02_start_mooncake_master.sh` 和 `10_start_server_kvpool_custom.sh`
   - 配置 `KV_ROLE=kv_both`、`KV_BACKEND=mooncake`
   - 用 `05_send_requests.py` 验证请求成功

4. 流式请求补充
   - 用 `09_send_stream_requests.py` 验证流式输出和 TTFT
   - 重点观察是否卡死或出现 fence/store 相关异常

## 通过标准
- 相关 UT 通过
- baseline 和 KV Pool server 均可启动并成功响应请求
- `summary.json` / `stream_summary.json` 无失败请求
- 日志无 `ImportError`、`ModuleNotFoundError`、旧路径残留、循环依赖、metadata/backend/pool worker/fence 相关异常
