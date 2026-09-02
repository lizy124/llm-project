# 场景：PD 分离（MultiConnector 双 connector + proxy）

> 目的：PD 分诊下 **MultiConnector 初始化链路无异常**（waiter 转发 /
> `set_external_slot_release_waiter` 路径的回归点）+ **KV 传输链路完整**。
> 判据：成功率 100%（每请求带回真实 token 计数）+ P/D 零 AttributeError +
> P 侧池写正计数 + D 侧逐层接收完整。
> 前置：[common-prerequisites.md](common-prerequisites.md) **全部**（§1–3 + §4——
> P 侧双 connector，memcache 与 mooncake 两个后端服务都要活）。
> 拓扑：proxy :9000 → P :8100（TP=4，MultiConnector）→ D :8200（TP=4，consumer）。单机 8 卡起步。
> **启动顺序硬约束：P ready → D ready → proxy**（proxy 启动时探 P/D 存活）。

## 1. 环境变量设置（容器内三个 shell 各自 export）

P/D 共通：

```bash
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_LOGGING_LEVEL=DEBUG          # 判据要 grep DEBUG 行(load_gvas/recv)
export ACL_OP_INIT_MODE=1                # A3/HCCS 分支
export ASCEND_ENABLE_USE_FABRIC_MEM=1
```

P 侧额外（双 connector 两后端都要）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3          # P 前半卡
export MMC_LOCAL_CONFIG_PATH=$CONF/mmc-local.conf   # memcache 侧
export MOONCAKE_CONFIG_PATH=$BASE/test/mooncake.json
export MOONCAKE_MASTER=127.0.0.1:50088
```

D 侧：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7          # D 后半卡
export MOONCAKE_CONFIG_PATH=$BASE/test/mooncake.json
export MOONCAKE_MASTER=127.0.0.1:50088
```

P/D 同机靠 `ASCEND_RT_VISIBLE_DEVICES` 隔离；不设会抢卡。起前 npu-smi 确认两组卡空闲。

## 2. 服务拉起（三个进程）

### 2a. P（prefill）:8100 — MultiConnector 双 connector

P 的 kv-transfer-config（MooncakeLayerwise + AscendStore/memcache layerwise 双 connector）：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "engine_id": "0",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_producer",
        "kv_port": 30000,
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 1, "tp_size": 4},
          "decode": {"dp_size": 1, "tp_size": 4}
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "backend": "memcache",
          "lookup_rpc_port": "0",
          "use_layerwise": true
        }
      }
    ]
  }
}
```

⚠️ `engine_id` 只放 MultiConnector **顶层**，子 connector 项内禁写——框架会注入
engine_id 到子项，重复写报 "got multiple values"。

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 --port 8100 --served-model-name $SERVED_MODEL_NAME \
  --trust-remote-code --enforce-eager --tensor-parallel-size 4 \
  --max-num-seqs 20 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 --block-size 128 --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CONFIG" > prefill.log 2>&1 &
echo $! > prefill.pid
```

### 2b. D（decode）:8200 — MooncakeLayerwise consumer

```json
{
  "kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": 31000,
  "engine_id": "1",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 1, "tp_size": 4},
    "decode": {"dp_size": 1, "tp_size": 4}
  }
}
```

启动命令同 P（端口 8200、chips 4-7、日志 decode.log）。P/D 的 engine_id 互异（0/1）。

### 2c. proxy :9000 — layerwise 分诊代理（P/D 就绪后才能起）

```bash
# 脚本在代码仓 examples/ 下,随验证分支走(路径以实际 checkout 为准)
PROXY_SCRIPT=$VLLM_ASCEND_DIR/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py
nohup python3 "$PROXY_SCRIPT" \
  --host 127.0.0.1 --port 9000 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200 \
  > proxy.log 2>&1 &
echo $! > proxy.pid
# 就绪:curl http://127.0.0.1:9000/healthcheck 有输出
```

`--host` 绑 127.0.0.1（**禁 0.0.0.0**，共享服务器）。

## 3. 请求方式（全部经 proxy :9000，不打 P/D）

```python
import json, urllib.request
base = "http://127.0.0.1:9000"   # 打 proxy

shared = ("You are validating disaggregated prefill/decode with a layerwise KV pool. "
          "Keep this shared prefix identical between requests. The numbered facts are: "
          "1 means alpha, 2 means beta, 3 means gamma. Repeat these facts internally "
          "before answering. ") * 80

questions = [
    "Question A: answer with the word alpha only.",
    "Question B: answer with the word beta only.",
    "Question C: answer with the word gamma only.",
]
# 可选:混入真实评测集问题(如 GSM8K-lite jsonl 前 2 条)增强真实性
# POST /v1/chat/completions, content = shared + q, max_tokens=32, temperature=0.0, timeout=600
# 逐发记录 elapsed / prompt_tokens / completion_tokens —— usage 必须带回真实 token 数(死服务给不出)
```

## 4. 判定命令

```bash
# ① 初始化回归点:P/D 日志零 AttributeError;MultiConnector 创建次数 = TP+1(4 workers + EngineCore)
grep -c "Creating v1 connector with name: AscendMultiConnector" prefill.log
grep -Eq "AttributeError" prefill.log decode.log    # 期望无
# ② 成功率 100%(见 §3,usage token 数非空)
# ③ P 侧池写:load_gvas 行数 > 0 + MetaService 计数 > 0
grep -c "load_gvas:" prefill.log
curl -s http://127.0.0.1:8000/metrics | grep -E "^memcache_(alloc_successes|stored_keys|query_successes)"
# ④ D 侧接收:recv 相关行 > 0 + 逐层 LayerMetadata 完整(0..num_layers-1 连续,基地址/block_len)
grep -cE "recv|remote_engine_id" decode.log
grep -E "LayerMetadata|model.layers" decode.log | tail -5
#    remote_engine_id 应与 D 侧 engine_id 一致
# ⑤ 致命错误扫描(两侧)
grep -Eiq 'Traceback|Segmentation fault' prefill.log decode.log
```

通过形态示例：5/5 成功（prompt_tokens=4177/4236/4198）；MultiConnector ×5；load_gvas 16 行；
D 侧 recv 34 行；27 层 LayerMetadata 完整。行数随命中窗口浮动属正常，判据是 **>0**。

## 5. 停止清理（顺序 proxy → decode → prefill，再清残留）

```bash
bash $BASE/test/stop_s1.sh        # 或等价的按 pid 逆序停三进程(common §5 模板)
bash $BASE/test/clean_npu.sh
```

## 6. 已知坑

- 启动顺序硬约束：P → D → proxy（proxy 起时探 P/D，倒了直接 blocked）
- engine_id 只写顶层；子 connector 写了报 "got multiple values"
- PD 场景独占 8 卡——起前 npu-smi 确认（HBM ~3GB 为空闲，~60GB 为被占）
- elapsed 是观察项非判据（首请求 0.66s 或 2.92s 都正常，completion 长度不同）
- memcache put 命名指标不存在（1.2.0 限制）；P 侧池写由 load_gvas + MetaService
  query_successes 承担
- 多机 PD（跨 RoCE）本手册未覆盖——拓扑要按官方 kv_pool §2.3/§3.5 调整后再用
