#!/usr/bin/env python3
"""verify_multi_161.py — 单服务内多前缀 prefix caching 验证 (161, single-TP8, :8004).

可供 64K 与 128K 两档复用, 用 CLI 参数控制:
  --pfx-tokens 65536|131072   前缀长度 (64K / 128K 档)
  --n-prefix N                同时常驻的前缀数 (64K:4, 128K:尽量小防挤占)
  --conc C                    并发扫描档位 (逗号分隔, 每档 R 轮混合随机前缀)

流程 (每步差分 /metrics):
  1. 逐前缀串行: 冷灌 + 同文重发, 确认各前缀单独命中能力
  2. 并发混合扫描: 所有前缀随机轮流, 采总体 hit_rate / TTFT; 观察多前缀同时常驻下的挤占
判读:
  · 单前缀重发命中高  -> 单前缀前缀复用正常 (同 161 V1)
  · 并发混合总体命中低 -> 多前缀无法同时常驻 (容量/挤占)
  · 并发混合总体命中高 -> 多前缀可同时常驻, 多前缀形态构造成功
"""
import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "http://127.0.0.1:8004"
MODEL_NAME = "dsv4-flash"
MODEL_PATH = "/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp"
SUFFIX_TOKENS = 256
SEED = 20260908
VOCAB = [f"w{i:03d}" for i in range(500)]


def post_completion(prompt, timeout=1800):
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": 1,
               "temperature": 0.0}
    req = urllib.request.Request(
        API + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
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


def gen_prefix(tok, rng, pfx):
    """生成 pfx 长度的不同前缀(固定 seed 打散, 避免跨前缀文本重叠)。"""
    return make_text(tok, rng, pfx)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfx-tokens", type=int, default=65536)
    ap.add_argument("--n-prefix", type=int, default=4)
    ap.add_argument("--conc", default="8,16")
    ap.add_argument("--out", required=True, help="JSON 输出路径")
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()

    with urllib.request.urlopen(API + "/v1/models", timeout=10) as r:
        assert b"data" in r.read()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    n = args.n_prefix
    prefixes = [gen_prefix(tok, random.Random(SEED + i * 79), args.pfx_tokens)
                for i in range(n)]
    suffix, _ = make_text(tok, random.Random(SEED + 999), SUFFIX_TOKENS)
    prompts = [p + " " + suffix for p in prefixes]
    n_tok = len(tok.encode(prompts[0]))
    print(f"[gen] n_prefix={n} pfx_tokens={args.pfx_tokens} suffix={SUFFIX_TOKENS} "
          f"prompt_tokens≈{n_tok}", flush=True)

    result = {"cfg": vars(args), "prefix_serial": {}, "conc": {}}

    # 1) 逐前缀串行: 冷灌 + 同文重发
    prev = snapshot()
    for i, pr in enumerate(prompts):
        dt0 = post_completion(pr)
        time.sleep(2)
        cur = snapshot()
        q0 = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
        h0 = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
        dt1 = post_completion(pr)  # 同文重发
        time.sleep(2)
        prev = cur
        cur = snapshot()
        q1 = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
        h1 = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
        prev = cur
        result["prefix_serial"][f"p{i}"] = {
            "cold_lat": round(dt0, 2), "cold_hits": h0,
            "replay_lat": round(dt1, 2), "replay_hits": h1,
            "replay_rate": round((h1 / q1 * 100) if q1 else 0, 2),
        }
        print(f"[P{i}] cold={dt0:.2f}s(queries={q0:.0f},hits={h0:.0f}) | "
              f"replay={dt1:.2f}s(hits={h1:.0f},rate={(h1/q1*100 if q1 else 0):.1f}%) "
              f"kv_usage={cur.get('kv_usage',-1)*100:.1f}%", flush=True)

    # 2) 并发混合扫描: 随机轮流取前缀
    for conc in [int(c) for c in args.conc.split(",")]:
        prev = snapshot()
        lat = []
        jobs = [(random.Random(SEED + conc).choice(prompts), None)
                for _ in range(args.rounds * conc)]
        def worker(job):
            return post_completion(job[0])
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            lat = list(ex.map(worker, jobs))
        wall = time.time() - t0
        time.sleep(3)  # 等结算
        cur = snapshot()
        q = cur["prefix_cache_queries"] - prev["prefix_cache_queries"]
        h = cur["prefix_cache_hits"] - prev["prefix_cache_hits"]
        rate = (h / q * 100) if q else 0.0
        result["conc"][str(conc)] = {
            "wall": round(wall, 2), "lat_mean": round(sum(lat)/len(lat), 2),
            "queries": q, "hits": h, "hit_rate": round(rate, 2),
            "kv_usage": round(cur.get("kv_usage", -1) * 100, 2),
        }
        print(f"[CONC{conc}] wall={wall:.2f}s mean_lat={sum(lat)/len(lat):.2f}s "
              f"queries={q:.0f} hits={h:.0f} rate={rate:.2f}% "
              f"kv_usage={cur.get('kv_usage',-1)*100:.2f}%", flush=True)
        prev = cur

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"SAVED {args.out}", flush=True)
    print("EXP_DONE", flush=True)


if __name__ == "__main__":
    main()