#!/usr/bin/env python3
"""verify_dp2_161.py — DP2×TP8 并发+命中验证 (port 8005).

核心问题:
  1. 两副本是否各有 ~279K KV (=> DP 真复制, 总容量翻倍)
  2. 无前缀亲和路由下, 并发同前缀请求, 平均命中率如何 (负载均衡会把同前缀请求打散到两副本,
     每副本首见该前缀时冷灌 -> 平均命中率折损)
"""
import json, random, re, time, urllib.request

API = "http://127.0.0.1:8005"
MODEL_NAME = "dsv4-flash"
MODEL_PATH = "/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp"
PREFIX_TOKENS = 65536
SEED = 20260907
VOCAB = [f"w{i:03d}" for i in range(500)]

def post(prompt, max_tokens=1):
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0}
    req = urllib.request.Request(API + "/v1/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.time() - t0

def snapshot():
    with urllib.request.urlopen(API + "/metrics", timeout=30) as r:
        txt = r.read().decode()
    out = {}
    for name in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(r"^vllm:" + name + r"_total(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M)
        if m: out[name] = float(m.group(2))
    m = re.search(r"^vllm:kv_cache_usage_perc(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M)
    if m: out["kv_usage"] = float(m.group(2))
    return out

def make_text(tok, rng, target):
    words = [rng.choice(VOCAB) for _ in range(target // 2 + 64)]
    text = " ".join(words); n = len(tok.encode(text)); guard = 0
    while n != target and guard < 500:
        guard += 1
        if n > target: text = " ".join(text.split(" ")[:-1])
        else: text += " " + " ".join(rng.choice(VOCAB) for _ in range(4))
        n = len(tok.encode(text))
    return text, n

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    rng = random.Random(SEED)
    prefix, _ = make_text(tok, rng, PREFIX_TOKENS)
    suffix, _ = make_text(tok, random.Random(SEED + 1), 256)
    prompt = prefix + " " + suffix

    # 串行首灌 (确认基础容量/命中能力)
    prev = snapshot()
    dt = post(prompt); time.sleep(2); cur = snapshot()
    q = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
    h = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
    print(f"[SEED] latency={dt:.2f}s hits={h:.0f} rate={(h/q*100 if q else 0):.1f}%", flush=True)

    # 并发 8 个同前缀 (模拟: 负载均衡打散到两副本)
    prev = cur
    import threading
    results = []
    def worker(res):
        res.append(post(prompt))
    threads = [threading.Thread(target=worker, args=(results,)) for _ in range(8)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    time.sleep(2); cur = snapshot()
    q = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
    h = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
    print(f"[CONC8] wall={wall:.2f}s mean_lat={sum(results)/len(results):.2f}s "
          f"hits={h:.0f} rate={(h/q*100 if q else 0):.1f}% kv_usage={cur.get('kv_usage',-1)*100:.1f}%", flush=True)
    print("EXP_DONE", flush=True)

if __name__ == "__main__":
    main()