#!/usr/bin/env bash
# s1_test.sh — 场景1 判定:PD 分离请求 100% 成功 + 无 AttributeError(#14465 回归点)+ 链路证据
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s1_pd_multiconn"
PROXY_PORT="${PROXY_PORT:-9000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv2-lite-pd}"
P_LOG="$RESULT_DIR/prefill.log"
D_LOG="$RESULT_DIR/decode.log"
TEST_LOG="$RESULT_DIR/test.log"
STATUS_FILE="$RESULT_DIR/test_status.txt"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }

[ -f "$P_LOG" ] || { echo "BLOCKED: no prefill.log"; exit 2; }
[ -f "$D_LOG" ] || { echo "BLOCKED: no decode.log"; exit 2; }
kill -0 "$(cat "$RESULT_DIR/prefill.pid")" 2>/dev/null || { echo "BLOCKED: prefill not running"; exit 2; }
kill -0 "$(cat "$RESULT_DIR/decode.pid")" 2>/dev/null || { echo "BLOCKED: decode not running"; exit 2; }
kill -0 "$(cat "$RESULT_DIR/proxy.pid")" 2>/dev/null || { echo "BLOCKED: proxy not running"; exit 2; }

: > "$TEST_LOG"
echo "RUNNING: S1 MultiConnector PD disaggregation validation" | tee "$STATUS_FILE"

# ---------- 0. 回归核心:初始化链路无 AttributeError(#14465 失败特征) ----------
# 修复前特征:AscendMultiConnector.__init__ → _configure_layerwise_reuse_completion
# → set_external_slot_release_waiter 即 AttributeError
if grep -q "AscendMultiConnector" "$P_LOG"; then
  echo "prefill: AscendMultiConnector init path present in log" | tee -a "$TEST_LOG"
fi
for L in "$P_LOG" "$D_LOG"; do
  if grep -Eq "AttributeError|_configure_layerwise_reuse_completion.*Error" "$L"; then
    grep -En "AttributeError" "$L" | tail -10 | tee -a "$TEST_LOG" >&2
    fail "AttributeError found in $L (#14465 regression signature)"
  fi
done
echo "no AttributeError in prefill/decode logs (#14465 regression point clear)" | tee -a "$TEST_LOG"

# ---------- 1. 请求序列(经 proxy,GSM8K-lite 问题 + 共享长前缀) ----------
python3 - "$PROXY_PORT" "$SERVED_MODEL_NAME" <<'PY' 2>&1 | tee -a "$TEST_LOG"
import json, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

with urllib.request.urlopen(base + "/healthcheck", timeout=10) as r:
    print("healthcheck:", r.read().decode()[:200])

shared = ("You are validating disaggregated prefill/decode with a layerwise KV pool. "
          "Keep this shared prefix identical between requests. The numbered facts are: "
          "1 means alpha, 2 means beta, 3 means gamma. Repeat these facts internally "
          "before answering. ") * 80

questions = ["Question A: answer with the word alpha only.",
             "Question B: answer with the word beta only.",
             "Question C: answer with the word gamma only."]

# 尝试混入 GSM8K-lite 真实问题
gsm8k = "/mnt/share/c00814587/vllm-ascend_gsm8k-lite/gsm8k-lite.jsonl"
try:
    with open(gsm8k) as f:
        lines = [json.loads(x) for x in f.readlines()[:2]]
    for item in lines:
        q = item.get("question") or item.get("prompt") or ""
        if q:
            questions.append("GSM8K question: " + q.strip()[:400])
    print(f"loaded {len(lines)} gsm8k-lite questions")
except Exception as e:
    print(f"gsm8k-lite unavailable ({e}), synthetic questions only")

ok, total = 0, 0
for i, q in enumerate(questions):
    total += 1
    t0 = time.time()
    try:
        r = post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": shared + q}],
            "max_tokens": 32, "temperature": 0.0})
        content = r["choices"][0]["message"]["content"].strip()
        usage = r.get("usage", {})
        assert content, "empty content"
        ok += 1
        print(f"req{i} ok elapsed={time.time()-t0:.2f}s prompt_tokens={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')}")
    except Exception as e:
        print(f"req{i} FAILED: {e}")

print(f"success_rate={ok}/{total}")
assert ok == total and total > 0, f"request success rate {ok}/{total} < 100%"
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "request sequence failed (<100% success)"

sleep 3

# ---------- 2. 传输链路证据 ----------
# 2a. P 侧 MooncakeLayerwise 发送 + AscendStore layerwise save(load_gvas/save 标记)
grep -cE "load_gvas:" "$P_LOG" 2>/dev/null | xargs -I{} echo "prefill load_gvas lines: {}" | tee -a "$TEST_LOG" || true
grep -E "metaserver|Send request" "$P_LOG" | tail -2 | tee -a "$TEST_LOG" || true

# 2b. D 侧 layerwise recv 证据(从远端拉取 KV 层)
D_RECV=$(grep -cE "recv|Recving|remote_engine_id" "$D_LOG" 2>/dev/null || echo 0)
echo "decode recv-related lines: $D_RECV" | tee -a "$TEST_LOG"

# 2c. P 侧 pool 证据:MetaService :8000 metrics(memcache put 成功)
curl -s --max-time 10 "http://127.0.0.1:8000/metrics" > "$RESULT_DIR/metaservice_metrics.txt" || fail "metaservice metrics unreachable"
PUT=$(awk '/^memcache_.*put.*success/ {s+=$2} END {print s+0}' "$RESULT_DIR/metaservice_metrics.txt")
QUERY=$(awk '/^memcache_query_successes_total/ {s+=$2} END {print s+0}' "$RESULT_DIR/metaservice_metrics.txt")
echo "metaservice put_success_sum=$PUT query_success_sum=$QUERY" | tee -a "$TEST_LOG"
[ "${PUT%.*}" -gt 0 ] || echo "WARN: memcache put success == 0 (P 侧 pool save 未见)" | tee -a "$TEST_LOG"

# ---------- 3. 致命错误扫描(两侧) ----------
for L in "$P_LOG" "$D_LOG"; do
  if grep -Eiq 'Traceback \(most recent call last\)|Segmentation fault|core dumped' "$L"; then
    grep -Ein 'Traceback|Segmentation fault' "$L" | tail -10 | tee -a "$TEST_LOG" >&2
    fail "fatal error pattern in $L"
  fi
done

echo "PASS: S1 MultiConnector PD disaggregation — 100% success, no AttributeError" | tee -a "$STATUS_FILE"
