#!/usr/bin/env bash
# s3_test.sh — 场景3 判定:请求序列 + mooncake 三维证据链 + 零回归(无 layerwise 痕迹)
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s3_mooncake_non_layerwise"
SERVER_PORT="${SERVER_PORT:-8006}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-mooncake}"
LOG_FILE="$RESULT_DIR/server.log"
TEST_LOG="$RESULT_DIR/test.log"
STATUS_FILE="$RESULT_DIR/test_status.txt"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }

[ -f "$LOG_FILE" ] || { echo "BLOCKED: no server.log"; exit 2; }
kill -0 "$(cat "$RESULT_DIR/server.pid")" 2>/dev/null || { echo "BLOCKED: server not running"; exit 2; }

: > "$TEST_LOG"
echo "RUNNING: S3 Qwen3-32B mooncake non-layerwise validation" | tee "$STATUS_FILE"

# ---------- 1. 请求序列 ----------
python3 - "$SERVER_PORT" "$SERVED_MODEL_NAME" <<'PY' 2>&1 | tee -a "$TEST_LOG"
import json, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=300):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
    assert "data" in json.loads(r.read().decode())
print("models endpoint ok")

smoke = post("/v1/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": "Give one short sentence about prefix caching."}],
    "max_tokens": 32, "temperature": 0.0})
assert smoke["choices"][0]["message"]["content"].strip()
print("smoke request ok")

shared = ("You are validating KV cache pool reuse. Keep this shared prefix identical "
          "between requests. The numbered facts are: 1 means alpha, 2 means beta, "
          "3 means gamma. Repeat these facts internally before answering. ") * 80

t0 = time.time()
r1 = post("/v1/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": shared + "Question A: answer with the word alpha only."}],
    "max_tokens": 16, "temperature": 0.0})
print(f"first repeated-prefix ok elapsed={time.time()-t0:.2f}s usage={json.dumps(r1.get('usage', {}))}")

t0 = time.time()
r2 = post("/v1/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": shared + "Question B: answer with the word beta only."}],
    "max_tokens": 16, "temperature": 0.0})
print(f"second repeated-prefix ok elapsed={time.time()-t0:.2f}s usage={json.dumps(r2.get('usage', {}))}")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "request sequence failed"

sleep 3

# ---------- 2. 非 layerwise 确认(零回归:layerwise 路径专属标记不该出现) ----------
# 注: pool_worker.py:445 "layerwise config: num_layers" 是无条件 info 日志,不能作为激活判据。
# 真正的 layerwise 路径标记是 load_gvas:/hit_check: 调试行(pool_worker.py:1501 / pool_scheduler.py:390)。
if grep -qE "load_gvas:|hit_check:" "$LOG_FILE"; then
  grep -E "load_gvas:|hit_check:" "$LOG_FILE" | tail -5 | tee -a "$TEST_LOG" >&2
  fail "unexpected layerwise path markers in non-layerwise scenario"
fi
echo "no layerwise path markers (load_gvas/hit_check absent, as expected)" | tee -a "$TEST_LOG"

# ---------- 3. mooncake master 三维证据链(:9008/metrics) ----------
curl -s --max-time 10 "http://127.0.0.1:9008/metrics" > "$RESULT_DIR/pool_metrics.txt" || fail "mooncake metrics unreachable"
mksum() { grep -E "^$1" "$RESULT_DIR/pool_metrics.txt" | awk '{s+=$2} END {print s+0}'; }
ALLOC=$(mksum "master_allocated_bytes")
KEYS=$(mksum "master_key_count")
CLIENTS=$(mksum "master_active_clients")
echo "mooncake evidence: allocated_bytes=$ALLOC key_count=$KEYS active_clients=$CLIENTS" | tee -a "$TEST_LOG"
[ "${ALLOC%.*}" -gt 0 ] || fail "存: master_allocated_bytes == 0"
[ "${KEYS%.*}" -gt 0 ] || fail "存: master_key_count == 0"
[ "${CLIENTS%.*}" -eq 4 ] || echo "WARN: active_clients=$CLIENTS != TP=4" | tee -a "$TEST_LOG"

# 取/去重:master.log 的 Get/ExistKey 成功计数
grep "Master Admin Metrics" "$BASE/run/mooncake_logs/master.log" | tail -1 > "$RESULT_DIR/master_last_metrics.txt" || true
grep -oE 'Get:\(Req=[0-9.]+/[0-9.]+/[0-9.]+' "$RESULT_DIR/master_last_metrics.txt" | tee -a "$TEST_LOG" || true

# ---------- 4. vllm /metrics external hit ----------
curl -s --max-time 10 "http://127.0.0.1:$SERVER_PORT/metrics" > "$RESULT_DIR/vllm_metrics.txt" || true
grep -E "external_prefix_cache_(queries|hits)" "$RESULT_DIR/vllm_metrics.txt" | grep -v "^#" | tail -6 | tee -a "$TEST_LOG"

# ---------- 5. 致命错误扫描 ----------
if grep -Eiq 'Traceback \(most recent call last\)|Segmentation fault|core dumped|Initialize mooncake failed|Store initialization failed|Configuration loading failed' "$LOG_FILE"; then
  grep -Ein 'Traceback|Segmentation fault|Initialize mooncake failed' "$LOG_FILE" | tail -20 | tee -a "$TEST_LOG" >&2
  fail "server log contains fatal error patterns"
fi

echo "PASS: S3 mooncake non-layerwise (Qwen3-32B) — zero-regression evidence chain" | tee -a "$STATUS_FILE"
