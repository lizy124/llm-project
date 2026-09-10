#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCENARIO_DIR/results}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/latest}"
SERVER_PORT="${SERVER_PORT:-8004}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-kvpool}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$SERVER_PORT}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"

LOG_FILE="$RESULT_DIR/server.log"
PID_FILE="$RESULT_DIR/server.pid"
TEST_LOG="$RESULT_DIR/test.log"
STATUS_FILE="$RESULT_DIR/test_status.txt"
SMOKE_RESPONSE="$RESULT_DIR/smoke_response.json"
PREFIX_RESPONSE_1="$RESULT_DIR/prefix_response_1.json"
PREFIX_RESPONSE_2="$RESULT_DIR/prefix_response_2.json"

fail_blocked() {
  echo "BLOCKED: $*" | tee -a "$STATUS_FILE" >&2
  exit 2
}

fail_failed() {
  echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2
  exit 1
}

pass_msg() {
  echo "PASS: $*" | tee -a "$STATUS_FILE"
}

[[ -d "$RESULT_DIR" ]] || fail_blocked "result directory not found: $RESULT_DIR"
[[ -f "$PID_FILE" ]] || fail_blocked "server pid file not found: $PID_FILE"
[[ -f "$LOG_FILE" ]] || fail_blocked "server log not found: $LOG_FILE"

PID="$(cat "$PID_FILE")"
kill -0 "$PID" 2>/dev/null || fail_blocked "server process is not running: pid=$PID"
command -v python >/dev/null 2>&1 || fail_blocked "python is not available"

: > "$TEST_LOG"
echo "RUNNING: Qwen3-32B PDMix non-layerwise KV Pool validation" | tee "$STATUS_FILE"

python - "$BASE_URL" "$SERVED_MODEL_NAME" "$REQUEST_TIMEOUT_S" "$SMOKE_RESPONSE" "$PREFIX_RESPONSE_1" "$PREFIX_RESPONSE_2" <<'PY' 2>&1 | tee -a "$TEST_LOG"
import json
import sys
import time
import urllib.error
import urllib.request

base_url, model, timeout_s, smoke_path, prefix1_path, prefix2_path = sys.argv[1:]
timeout_s = int(timeout_s)


def post_json(path, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {body}")
        return json.loads(body)


def extract_text(resp):
    choices = resp.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "message" in choice:
        return choice["message"].get("content") or ""
    return choice.get("text") or ""

# Readiness must still work when validation starts.
with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as resp:
    models = json.loads(resp.read().decode("utf-8"))
if "data" not in models:
    raise SystemExit("/v1/models response has no data field")
print("models endpoint ok")

smoke_payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Give one short sentence about prefix caching."}
    ],
    "max_tokens": 32,
    "temperature": 0.0,
}
smoke = post_json("/v1/chat/completions", smoke_payload, timeout_s)
with open(smoke_path, "w", encoding="utf-8") as f:
    json.dump(smoke, f, ensure_ascii=False, indent=2)
smoke_text = extract_text(smoke)
if not smoke_text.strip():
    raise SystemExit("smoke response is empty")
print("smoke request ok")

shared_prefix = (
    "You are validating KV cache pool reuse. "
    "Keep this shared prefix identical between requests. "
    "The numbered facts are: 1 means alpha, 2 means beta, 3 means gamma. "
    "Repeat these facts internally before answering. "
) * 80

payload_1 = {
    "model": model,
    "messages": [
        {"role": "user", "content": shared_prefix + "Question A: answer with the word alpha only."}
    ],
    "max_tokens": 16,
    "temperature": 0.0,
}
payload_2 = {
    "model": model,
    "messages": [
        {"role": "user", "content": shared_prefix + "Question B: answer with the word beta only."}
    ],
    "max_tokens": 16,
    "temperature": 0.0,
}

start = time.time()
resp1 = post_json("/v1/chat/completions", payload_1, timeout_s)
elapsed1 = time.time() - start
with open(prefix1_path, "w", encoding="utf-8") as f:
    json.dump(resp1, f, ensure_ascii=False, indent=2)
text1 = extract_text(resp1)
if not text1.strip():
    raise SystemExit("first repeated-prefix response is empty")
print(f"first repeated-prefix request ok elapsed={elapsed1:.2f}s")

start = time.time()
resp2 = post_json("/v1/chat/completions", payload_2, timeout_s)
elapsed2 = time.time() - start
with open(prefix2_path, "w", encoding="utf-8") as f:
    json.dump(resp2, f, ensure_ascii=False, indent=2)
text2 = extract_text(resp2)
if not text2.strip():
    raise SystemExit("second repeated-prefix response is empty")
print(f"second repeated-prefix request ok elapsed={elapsed2:.2f}s")

usage1 = resp1.get("usage", {})
usage2 = resp2.get("usage", {})
print("usage1", json.dumps(usage1, ensure_ascii=False))
print("usage2", json.dumps(usage2, ensure_ascii=False))
PY

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  fail_failed "OpenAI API request validation failed; see $TEST_LOG"
fi

if grep -Eiq 'Traceback \(most recent call last\)|RuntimeError:|Segmentation fault|core dumped|ERR[0-9]+|HCCL.*(error|failed)|NPU.*(error|failed)|Store initialization failed|Configuration loading failed' "$LOG_FILE"; then
  grep -Ein 'Traceback \(most recent call last\)|RuntimeError:|Segmentation fault|core dumped|ERR[0-9]+|HCCL.*(error|failed)|NPU.*(error|failed)|Store initialization failed|Configuration loading failed' "$LOG_FILE" | tail -80 | tee -a "$TEST_LOG" >&2
  fail_failed "server log contains fatal error patterns"
fi

if ! grep -Eiq 'AscendStoreConnector|Initializing Memcache store|KV pool|kv pool|start_load_kv|store KV cache|lookup_rpc_port' "$LOG_FILE"; then
  fail_failed "server log does not show AscendStore/Memcache KV Pool activity"
fi

pass_msg "Qwen3-32B PDMix non-layerwise KV Pool smoke and repeated-prefix validation passed"
echo "result_dir=$RESULT_DIR"
