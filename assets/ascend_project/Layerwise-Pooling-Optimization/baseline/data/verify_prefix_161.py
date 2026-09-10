#!/usr/bin/env python3
"""verify_prefix_161.py — DSV4-Flash prefix caching 鉴别实验.

三场景, 每步差分 /metrics:
  A  : prefix(4096, block 对齐) + suffix1(1024)   首灌, 期望 hits≈0
  A' : 与 A 完全相同文本                            鉴别: 哈希链/同文复用是否工作
  B  : 同 prefix + suffix2(不同)                   鉴别: 跨请求前缀复用是否工作

判读:
  A' hits≈0      -> 哈希/缓存注册层面有 bug (配置或代码问题)
  A' hits>0 且 B hits≈0 -> 同文可复用, 前缀跨请求不可复用 (滑窗释放语义, 需求级问题)
  A' hits>0 且 B hits>0 -> 前缀复用工作正常 (165 的问题另有原因)
"""
import json
import random
import re
import time
import urllib.request

API = "http://127.0.0.1:8004"
MODEL_NAME = "dsv4-flash"
MODEL_PATH = "/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp"

PREFIX_TOKENS = 65536  # 4x SWA window(16384): 读写对齐(scheduler_block_size=16384)均非零
SUFFIX_TOKENS = 256
SEED = 20260907
VOCAB = [f"w{i:03d}" for i in range(500)]


def post_completion(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": 1,
               "temperature": 0.0}
    req = urllib.request.Request(
        API + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.time() - t0


def parse_counters(text):
    out = {}
    for name in ("prefix_cache_queries", "prefix_cache_hits"):
        pattern = (r"^vllm:" + re.escape(name) + r"_total(\{[^}]*\})?\s+"
                   r"([0-9.eE+-]+)\s*$")
        m = re.search(pattern, text, re.M)
        if m:
            out[name] = float(m.group(2))
    return out


def snapshot():
    with urllib.request.urlopen(API + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = parse_counters(text)
    m = re.search(r"^vllm:kv_cache_usage_perc(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
                  text, re.M)
    if m:
        out["kv_usage"] = float(m.group(2))
    return out


def make_text(tok, rng, target):
    words = [rng.choice(VOCAB) for _ in range(target // 2 + 64)]
    text = " ".join(words)
    n = len(tok.encode(text))
    guard = 0
    while n != target and guard < 500:
        guard += 1
        if n > target:
            text = " ".join(text.split(" ")[:-1])
        else:
            text += " " + " ".join(rng.choice(VOCAB) for _ in range(4))
        n = len(tok.encode(text))
    return text, n


def main():
    with urllib.request.urlopen(API + "/v1/models", timeout=10) as r:
        assert b"data" in r.read()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    rng = random.Random(SEED)
    prefix, n_prefix = make_text(tok, rng, PREFIX_TOKENS)
    suffix1, n_s1 = make_text(tok, random.Random(SEED + 1), SUFFIX_TOKENS)
    suffix2, n_s2 = make_text(tok, random.Random(SEED + 2), SUFFIX_TOKENS)
    print(f"[gen] prefix={n_prefix} s1={n_s1} s2={n_s2} tokens", flush=True)

    scenarios = [
        ("A", prefix + " " + suffix1),
        ("A2", prefix + " " + suffix1),      # 与 A 完全相同
        ("B", prefix + " " + suffix2),       # 同前缀异后缀
    ]

    prev = snapshot()
    for name, prompt in scenarios:
        n_tok = len(tok.encode(prompt))
        dt = post_completion(prompt)
        time.sleep(2)  # 等引擎释放/保留结算完再采样
        cur = snapshot()
        q = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
        h = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
        rate = (h / q * 100) if q else 0.0
        print(f"[{name}] prompt={n_tok} tokens, latency={dt:.2f}s, "
              f"queries={q:.0f}, hits={h:.0f}, hit_rate={rate:.2f}%, "
              f"kv_usage_now={cur.get('kv_usage', -1)*100:.2f}%", flush=True)
        prev = cur

    # 二次确认: A 再发一遍 (B 之后), 看 B 的调度是否挤掉 A 的缓存
    dt = post_completion(scenarios[0][1])
    time.sleep(2)
    cur = snapshot()
    q = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
    h = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
    print(f"[A3] latency={dt:.2f}s, queries={q:.0f}, hits={h:.0f}", flush=True)

    print("EXP_DONE", flush=True)


if __name__ == "__main__":
    main()
