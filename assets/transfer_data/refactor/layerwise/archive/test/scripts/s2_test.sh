#!/usr/bin/env bash
# s2_test.sh — 场景2 判定:激活证据 + valid_gvas>0 + hit_tokens>0 + 三维证据链
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s2_memcache_layerwise"
SERVER_PORT="${SERVER_PORT:-8004}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv2-lite-layerwise}"
LOG_FILE="$RESULT_DIR/server.log"
TEST_LOG="$RESULT_DIR/test.log"
STATUS_FILE="$RESULT_DIR/test_status.txt"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }

[ -f "$LOG_FILE" ] || { echo "BLOCKED: no server.log"; exit 2; }
kill -0 "$(cat "$RESULT_DIR/server.pid")" 2>/dev/null || { echo "BLOCKED: server not running"; exit 2; }

: > "$TEST_LOG"
echo "RUNNING: S2 DSV2-Lite memcache layerwise validation" | tee "$STATUS_FILE"

# ---------- 1. 请求序列:smoke + 长前缀 x2 ----------
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
e1 = time.time() - t0
print(f"first repeated-prefix ok elapsed={e1:.2f}s usage={json.dumps(r1.get('usage', {}))}")

t0 = time.time()
r2 = post("/v1/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": shared + "Question B: answer with the word beta only."}],
    "max_tokens": 16, "temperature": 0.0})
e2 = time.time() - t0
print(f"second repeated-prefix ok elapsed={e2:.2f}s usage={json.dumps(r2.get('usage', {}))}")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "request sequence failed"

sleep 3  # 等 metrics 刷新

# ---------- 2. layerwise 激活证据 ----------
grep -E "layerwise config: num_layers" "$LOG_FILE" | tail -2 | tee -a "$TEST_LOG" || fail "no layerwise config activation line"

# ---------- 3. hit_check hit_tokens>0 + load_gvas valid_gvas>0 ----------
HIT=$(grep "hit_check:" "$LOG_FILE" | grep -v "hit_tokens=0" | tail -3 || true)
echo "$HIT" | tee -a "$TEST_LOG"
echo "$HIT" | grep -Eq "hit_tokens=[1-9]" || fail "no hit_check line with hit_tokens>0"

GVAS=$(grep "load_gvas:" "$LOG_FILE" | grep -v "valid_gvas=0 " | tail -3 || true)
echo "$GVAS" | tee -a "$TEST_LOG"
echo "$GVAS" | grep -Eq "valid_gvas=[1-9]" || fail "no load_gvas line with valid_gvas>0"

# ---------- 4. vllm /metrics:external hit ----------
curl -s --max-time 10 "http://127.0.0.1:$SERVER_PORT/metrics" > "$RESULT_DIR/vllm_metrics.txt" || true
grep -E "external_prefix_cache_(queries|hits)" "$RESULT_DIR/vllm_metrics.txt" | grep -v "^#" | tail -6 | tee -a "$TEST_LOG"
grep -E "external_prefix_cache_hits_total" "$RESULT_DIR/vllm_metrics.txt" | grep -v "^#" | grep -Eq " [1-9]" \
  || echo "WARN: external_prefix_cache_hits_total not >0 (检查指标名)" | tee -a "$TEST_LOG"

# ---------- 5. MetaService :8000/metrics 三维证据链(memcache 1.2.0 指标名) ----------
curl -s --max-time 10 "http://127.0.0.1:8000/metrics" > "$RESULT_DIR/pool_metrics.txt" || fail "MetaService metrics unreachable"
mksum() { grep -E "^$1" "$RESULT_DIR/pool_metrics.txt" | awk '{s+=$2} END {print s+0}'; }
ALLOC=$(mksum "memcache_alloc_successes_total")
KEYS=$(mksum "memcache_stored_keys")
QUERY=$(mksum "memcache_query_successes_total")
QNF=$(mksum "memcache_query_not_found_total")
# 注:layerwise/GVA 路径数据面走 GVA 直读(device_sdma),不经 memcache get API,
# "取"证据由 §3 valid_gvas>0 + §4 vllm external hits 承担
echo "pool evidence: alloc_successes=$ALLOC stored_keys=$KEYS query_successes=$QUERY query_not_found=$QNF" | tee -a "$TEST_LOG"
[ "${ALLOC%.*}" -gt 0 ] || fail "存: memcache_alloc_successes_total == 0"
[ "${KEYS%.*}" -gt 0 ] || fail "存: memcache_stored_keys == 0"
[ "${QUERY%.*}" -gt 0 ] || fail "去重: memcache_query_successes_total == 0"

# ---------- 6. 致命错误扫描 ----------
if grep -Eiq 'Traceback \(most recent call last\)|Segmentation fault|core dumped|Store initialization failed|Configuration loading failed' "$LOG_FILE"; then
  grep -Ein 'Traceback|Segmentation fault|Store initialization failed' "$LOG_FILE" | tail -20 | tee -a "$TEST_LOG" >&2
  fail "server log contains fatal error patterns"
fi

echo "PASS: S2 memcache layerwise (DSV2-Lite) — activation + valid_gvas>0 + pool evidence chain" | tee -a "$STATUS_FILE"
