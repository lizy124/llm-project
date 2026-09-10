# 场景：单实例 + mooncake 非 layerwise 零回归

> 目的：**通用路径在非 layerwise 后端零回归**——layerwise 路径标记必须完全缺席。
> 本场景同时是 [memcache-layerwise.md](memcache-layerwise.md) 的**阴性对照**：
> 同一组 grep 在这里必须得 0（verify_guide.md §6.6⑤）。
> 前置：[common-prerequisites.md](common-prerequisites.md) §4（mooncake master）。
> 模型选型：标准 attention 模型（Qwen3 类）——非 layerwise 不受 NSA 硬限影响，
> 且与 layerwise 场景不同模型可以隔离"模型特性"变量（verify_guide §5.4 控制变量法）。

## 1. 环境变量设置（容器内）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MOONCAKE_CONFIG_PATH=$BASE/test/mooncake.json
export MOONCAKE_MASTER=127.0.0.1:50088
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export ACL_OP_INIT_MODE=1                # A3/HCCS 分支
export ASCEND_ENABLE_USE_FABRIC_MEM=1
# 与 layerwise 场景的差异:无 MMC_LOCAL_CONFIG_PATH(不走 memcache);
# 无 VLLM_LOGGING_LEVEL=DEBUG(判据是标记缺席,无 DEBUG 行依赖)
```

## 2. 服务拉起

kv-transfer-config（**无 use_layerwise 字段** = 非 layerwise 通用路径）：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "0"
  }
}
```

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 --port 8006 \
  --served-model-name $SERVED_MODEL_NAME \
  --trust-remote-code --enforce-eager \
  --tensor-parallel-size 4 \
  --max-num-seqs 20 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 --block-size 128 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CONFIG" \
  > server.log 2>&1 &
echo $! > server.pid
# 就绪等待同 layerwise 场景(/v1/models 轮询)
```

## 3. 请求方式

与 layerwise 场景**完全同构**（smoke → 共享长前缀 ×2，仅换 model/port；
前缀模板见 [memcache-layerwise.md](memcache-layerwise.md) §3）——
同构请求序列是阴性对照成立的前提（同一方法跑两个场景才能对比）。

## 4. 判定命令

```bash
LOG=server.log
# ① 零回归核心:layerwise 路径标记必须缺席(与 layerwise 场景同一组 grep)
grep -cE "load_gvas:|hit_check:" $LOG
#   期望 0;>0 = FAIL(非 layerwise 场景出现 layerwise 路径 = gate 泄漏/回归)
# ② mooncake master 三维证据链(独立证人)
curl -s http://127.0.0.1:9008/metrics | grep -E "^master_(allocated_bytes|key_count|active_clients) "
#   期望 allocated_bytes>0, key_count>0, active_clients=TP 数
# ③ vllm metrics 命中(取维)
curl -s http://127.0.0.1:8006/metrics | grep external_prefix_cache
# ④ master.log Get/ExistKey 速率(取/去重的服务端证据)
grep "Master Admin Metrics" $BASE/run/mooncake_logs/master.log | tail -1
# ⑤ 致命错误扫描(含 mooncake 特有 'Initialize mooncake failed')
grep -Eiq 'Traceback|Segmentation fault|Initialize mooncake failed' $LOG
```

通过形态示例：`load_gvas/hit_check` 0 行；master `allocated_bytes≈939MB key_count=112
active_clients=4`；`external_prefix_cache_hits_total>0`。

## 5. 停止清理

```bash
bash $BASE/test/stop_server.sh $BASE/run/<本场景目录>
bash $BASE/test/clean_npu.sh
# mooncake master 跨场景可复用,勿每场景重启(见 common §4)
```

## 6. 已知坑

- `master_put_start_requests_total=0` ≠ 池化未生效（mooncake 走 batch API，指标恒 0，
  verify_guide §7.1——看 allocated_bytes/key_count）
- active_clients 应 = TP 数；不等查 worker 注册
- 本场景 PASS 的前提是与 layerwise 场景构成对照对：**layerwise 场景 >0 且本场景 = 0**
  才证明判定 grep 本身有效
