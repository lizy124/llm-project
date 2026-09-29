#!/usr/bin/env bash
# =============================================================================
# R9 GATE — 读路径池命中验证（prefix-caching 关闭形态）
# S3 Phase-2a 硬前置 gate（pool_nightly_proposal §5 Phase 2 第 2 步）
#
# 验证两件事（在 a3-800i-16 nightly pod 上，NPU 侧执行）：
#   1. --no-enable-prefix-caching 下请求仍携带 block_hashes（上游 v0.30.0
#      行为无法静态核实，本 gate 运行时判定）；
#   2. memcache 池（AscendStoreConnector, use_layerwise=true, gva 路径）
#      消费它们：第二条共享前缀请求命中池。
#
# 服务形态 = S3 nightly 原样（Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp, TP4,
# block-size 128, MTP1, 1M ctx, int8 indexer）+ 恰好两处 2a 待测变更：
#   prefix-caching OFF  +  kv-transfer-config(AscendStore/memcache/layerwise)
# 与 nightly S3 的唯一偏离：DP4 -> DP1（两条请求必须落在同一 engine，
# 池命中归因才无歧义；nightly S3 的每个 DP rank 本就是这个 TP4 形态）。
#
# 判据（pool_scheduler.py:715-721，logger.info，默认级别可见）：
#   PASS : 请求 #2 的日志行 "kvpool hit tokens: N" 满足 N > 0
#          （期望 ≈ 共享前缀对齐块数，~4000）
#   FAIL : N == 0 且无报错 → R9 哑失效形态，S3 2a 受阻，
#          按 proposal §6 R9 回退菜单处置
#
# 用法（nightly pod / 验证机，任一具备 ≥4 卡 + memcache_hybrid 包的环境）：
#   bash r9_gate.sh                # 全自动：起池、起服务、发压、判据、清理
#   R9_KEEP=1 bash r9_gate.sh      # 任何结局保留服务与日志（排障用）
#   R9_DEBUG=1 bash r9_gate.sh     # VLLM_LOGGING_LEVEL=DEBUG，输出逐块 hit_check
#   MODEL=<本地路径> bash r9_gate.sh   # Eco-Tech 名字解析失败时手工指定
# 可调环境量：GATE_DIR MODEL SERVED API_PORT META_PORT STORE_PORT TP DEVICES
#             PREFIX_CHARS READY_TIMEOUT
# =============================================================================
set -euo pipefail

GATE_DIR=${GATE_DIR:-/tmp/r9_gate}
MODEL=${MODEL:-"Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp"}   # S3 nightly 原样
SERVED=${SERVED:-"r9-gate"}
API_PORT=${API_PORT:-8000}
META_PORT=${META_PORT:-5000}     # S4/_kv_pool 同款端口规划
STORE_PORT=${STORE_PORT:-6000}
TP=${TP:-4}
DEVICES=${DEVICES:-"0,1,2,3"}
PREFIX_CHARS=${PREFIX_CHARS:-17000}   # ~4.2k tokens，31 个 128-token 块
READY_TIMEOUT=${READY_TIMEOUT:-2400}   # 40 min（1M ctx TP4 编译启动慢）
LOG_LEVEL=INFO
[ "${R9_DEBUG:-0}" = "1" ] && LOG_LEVEL=DEBUG

META_CONF="$GATE_DIR/mmc-meta.conf"
LOCAL_CONF="$GATE_DIR/mmc-local.conf"
POOL_LOG="$GATE_DIR/meta_service.log"
SRV_LOG="$GATE_DIR/vllm_server.log"
SERVER_PID=""
POOL_PID=""
KEEP=0          # R9_KEEP=1 时失败现场保留

pass() { echo "[R9-GATE] PASS — $*";          exit 0; }
fail() { echo "[R9-GATE] FAIL — $*" >&2;
         if [ "${R9_KEEP:-0}" = "1" ]; then
           KEEP=1
           echo "[R9-GATE] R9_KEEP=1：保留现场，日志在 $GATE_DIR（手工清理）"
         else
           echo "[R9-GATE] 服务器日志尾部 30 行：" >&2
           tail -n 30 "$SRV_LOG" >&2 2>/dev/null || true
         fi
         exit 1; }
cleanup() {
  [ "$KEEP" = "1" ] && return 0
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$POOL_PID" ]   && kill "$POOL_PID"   2>/dev/null || true
  wait 2>/dev/null || true
}
trap 'cleanup' EXIT
trap 'echo "[R9-GATE] interrupted"; exit 130' INT TERM

mkdir -p "$GATE_DIR"

# ---------- 0. 前置自检 ----------
for p in "$API_PORT" "$META_PORT" "$STORE_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":$p "; then
    fail "port $p 已被占用（共享服务器端口纪律：先预检再用）"
  fi
done
python3 -c "from memcache_hybrid import MetaService" 2>/dev/null \
  || fail "memcache_hybrid 不可导入——请在 nightly pod 镜像环境执行"
command -v vllm >/dev/null 2>&1 || fail "vllm CLI 不在 PATH"
command -v ss   >/dev/null 2>&1 || fail "ss 不可用（iproute2）——本脚本依赖端口探测"
command -v curl >/dev/null 2>&1 || fail "curl 不可用"

# ---------- 1. memcache 池配置（逐字复刻 S4 _kv_pool anchor + runner 注入） ----------
cat > "$META_CONF" <<EOF
ock.mmc.log_level = info
ock.mmc.meta_service_url = tcp://127.0.0.1:$META_PORT
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:$STORE_PORT
EOF
cat > "$LOCAL_CONF" <<EOF
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 10GB
ock.mmc.meta_service_url = tcp://127.0.0.1:$META_PORT
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:$STORE_PORT
EOF

# ---------- 2. 起池服务（MetaService），等 5000/6000 就绪 ----------
MMC_META_CONFIG_PATH="$META_CONF" \
  nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > "$POOL_LOG" 2>&1 &
POOL_PID=$!
deadline=$(( $(date +%s) + 300 ))
until ss -ltn 2>/dev/null | grep -q ":$META_PORT " \
   && ss -ltn 2>/dev/null | grep -q ":$STORE_PORT "; do
  kill -0 "$POOL_PID" 2>/dev/null || fail "MetaService 提前退出，见 $POOL_LOG"
  [ "$(date +%s)" -lt "$deadline" ] || fail "MetaService 5 分钟未就绪，见 $POOL_LOG"
  sleep 2
done
echo "[R9-GATE] memcache pool ready (ports $META_PORT/$STORE_PORT, pid $POOL_PID)"

# ---------- 3. 起 vLLM 服务（S3 形态 + 两处 2a 待测变更，DP1×TP4） ----------
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export OMP_NUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024 ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=3000 VLLM_LOGGING_LEVEL="$LOG_LEVEL"
export MMC_LOCAL_CONFIG_PATH="$LOCAL_CONF"

nohup vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$API_PORT" \
  --no-enable-prefix-caching \
  --max-model-len 1048576 \
  --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --data-parallel-size 1 \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --safetensors-load-strategy prefetch \
  --quantization ascend \
  --api-server-count 1 \
  --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
  --block-size 128 \
  --attention_config.indexer_kv_dtype int8 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":"true","enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"enable_fused_mc2":0}' \
  --kv-transfer-config '{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backend": "memcache", "lookup_rpc_port": "0", "use_layerwise": true}}' \
  > "$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo "[R9-GATE] waiting for /health (up to $READY_TIMEOUT s)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; do
  kill -0 "$SERVER_PID" 2>/dev/null \
    || fail "vLLM 服务提前退出，日志 $SRV_LOG"
  [ "$(date +%s)" -lt "$deadline" ] || fail "服务 $READY_TIMEOUT s 未就绪，日志 $SRV_LOG"
  sleep 10
done
echo "[R9-GATE] vLLM ready (pid $SERVER_PID)"

# ---------- 4. 两条共享前缀请求（串行；req1 完成后其前缀块已入池） ----------
PREFIX_CHARS="$PREFIX_CHARS" SERVED="$SERVED" GATE_DIR="$GATE_DIR" python3 - <<'PY'
import json, os
g = os.environ["GATE_DIR"]
para = ("The reliability report compiles nightly results from every regional cluster. "
        "Each entry lists the tenant, the shard, the duration and the verdict, and "
        "reviewers annotate anomalies before forwarding them to the on-call queue. ")
n = int(os.environ["PREFIX_CHARS"]) // len(para) + 1
prefix = (para * n)[: int(os.environ["PREFIX_CHARS"])]
base = {"model": os.environ["SERVED"], "max_tokens": 64,
        "temperature": 0, "ignore_eos": True}
q1 = "\n\nQuestion: What is 17 + 26? Answer with the number only.\nAnswer:"
q2 = "\n\nQuestion: What is 38 + 44? Answer with the number only.\nAnswer:"
json.dump({**base, "prompt": prefix + q1}, open(f"{g}/req1.json", "w"))
json.dump({**base, "prompt": prefix + q2}, open(f"{g}/req2.json", "w"))
print(f"[R9-GATE] prefix {len(prefix)} chars (~{len(prefix)//4} tokens)")
PY

curl -s --max-time 900 -H "Content-Type: application/json" \
  -d @"$GATE_DIR/req1.json" "http://127.0.0.1:$API_PORT/v1/completions" \
  -o "$GATE_DIR/resp1.json" || fail "req#1 请求失败（服务侧问题，先看 $SRV_LOG）"
sleep 5   # 确保 req#1 的 layerwise put 全部落池
curl -s --max-time 900 -H "Content-Type: application/json" \
  -d @"$GATE_DIR/req2.json" "http://127.0.0.1:$API_PORT/v1/completions" \
  -o "$GATE_DIR/resp2.json" || fail "req#2 请求失败（服务侧问题，先看 $SRV_LOG）"
sleep 3   # 留出日志刷盘时间

# ---------- 5. 判据 ----------
echo "[R9-GATE] ---- kvpool hit tokens 全部命中行 ----"
grep -E "kvpool hit tokens" "$SRV_LOG" | tee "$GATE_DIR/hit_lines.txt" || true
N_LINES=$(grep -cE "kvpool hit tokens" "$SRV_LOG" || true)
if [ "$N_LINES" -lt 2 ]; then
  fail "命中日志行数 $N_LINES < 2：lookup 未按预期为两条请求各打一行（见 $SRV_LOG）"
fi
LAST=$(grep -E "kvpool hit tokens" "$SRV_LOG" | tail -n 1)
N=$(echo "$LAST" | grep -oE "kvpool hit tokens: [0-9]+" | grep -oE "[0-9]+$" || echo 0)
echo "[R9-GATE] 请求#2 判据行: $LAST"
if [ "$N" -gt 0 ]; then
  pass "block_hashes 在 prefix-off 下仍生成且池命中（hit tokens=$N）——S3 2a 放行"
else
  fail "请求#2 hit tokens=0 且无报错——R9 哑失效形态成立，S3 2a 受阻（§6 R9 回退）"
fi
