#!/usr/bin/env python3
"""Mock 负载链路验证（缩小序列长度等比）：前缀池生成 -> 灌入 -> 并发扫描 -> TPS/TTFT/命中率采集.

等比缩小: 128K -> 16K（前缀 14336 = 112 blocks + 后缀 ~1536，目标命中率 ~90%）
运行位置: baseline_165 容器内；依赖 transformers（镜像自带）；服务须已就绪于 :8004
用法: python3 bench_mock.py --out /home/lizhongyang/lw_baseline/results/mock_round1

设计对齐 01_method.md §三/§四:
- 前缀 block 对齐（128 倍数），保证命中边界干净
- 命中率从 /metrics 前后差分采集（不是本地推算）
- 每请求 token 数用 tokenizer 精确计数（TPS 分子）
- 固定 seed 可复现
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "http://127.0.0.1:8004"
MODEL_NAME = "dsv4-flash"
MODEL_PATH = "/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp"

BLOCK = 128
P_PREFIX = 12                  # 前缀池规模（容量 63% 占用，mock 轮容量约束下取值）
PREFIX_TOKENS = 112 * BLOCK    # 14336
SUFFIX_WORDS = 700             # 后缀 ~1.5K token，无需对齐（本来就是 miss 部分）
LEVELS = [4, 8, 16, 32]
SEED = 20260906
VOCAB = [f"w{i:03d}" for i in range(500)]


# ---------------- HTTP ----------------

def post_json(path, payload, timeout=900):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def get_metrics_text():
    with urllib.request.urlopen(API + "/metrics", timeout=30) as r:
        return r.read().decode()


def parse_counters(text):
    """解析 prefix cache 计数器；兼容 vllm:name / vllm:name_total 两种导出、新旧命名、带 label 段。"""
    out = {}
    for name in ("gpu_prefix_cache_queries", "gpu_prefix_cache_hits",
                 "prefix_cache_queries", "prefix_cache_hits"):
        for variant in (f"vllm:{name}_total", f"vllm:{name}"):
            pattern = "^" + re.escape(variant) + r"(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$"
            m = re.search(pattern, text, re.M)
            if m:
                out[name] = float(m.group(2))
                break
    return out


def get_q(m):
    return m.get("gpu_prefix_cache_queries", m.get("prefix_cache_queries", 0.0))


def get_h(m):
    return m.get("gpu_prefix_cache_hits", m.get("prefix_cache_hits", 0.0))


# ---------------- 负载生成 ----------------

def make_prefix_text(tok, rng, target):
    """生成 target 个 token 的随机文本（block 对齐），逐词收敛。"""
    words = [rng.choice(VOCAB) for _ in range(target // 2 + 64)]
    text = " ".join(words)
    n = len(tok.encode(text))
    guard = 0
    while n != target and guard < 500:
        guard += 1
        if n > target:
            if n - target > 50:  # 差距大时批量删，减少重编码次数
                cut = (n - target) // 3
                text = " ".join(text.split(" ")[:-cut])
            else:
                text = " ".join(text.split(" ")[:-1])
        else:
            text += " " + " ".join(rng.choice(VOCAB) for _ in range(4))
        n = len(tok.encode(text))
    return text, n


def make_suffix(rng):
    return " ".join(rng.choice(VOCAB) for _ in range(SUFFIX_WORDS)) + f" tail{rng.randrange(1 << 30)}"


# ---------------- 请求 ----------------

def one_request(prompt, max_tokens=16, timeout=900):
    """流式请求，返回 (ttft_s, total_s)。TTFT = 首个非空 text chunk 时刻。"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    req = urllib.request.Request(
        API + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            if ttft is None and ch.get("text"):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return (ttft if ttft is not None else total), total


def warmup_one(prompt):
    post_json("/v1/completions",
              {"model": MODEL_NAME, "prompt": prompt,
               "max_tokens": 1, "temperature": 0.0})


# ---------------- 扫描 ----------------

def run_level(level, prefixes, tok, out_dir):
    tag_seed = SEED * 100000 + level * 1000
    R = max(40, 5 * level)
    prompts = []
    for i in range(R):
        rr = random.Random(tag_seed + i)
        prompts.append(prefixes[rr.randrange(len(prefixes))] + " " + make_suffix(rr))
    n_tokens = [len(tok.encode(p)) for p in prompts]

    m_before = parse_counters(get_metrics_text())
    results = [None] * R

    def work(i):
        results[i] = one_request(prompts[i])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as ex:
        list(ex.map(work, range(R)))
    wall = time.perf_counter() - t0
    m_after = parse_counters(get_metrics_text())

    ttfts = sorted(r[0] for r in results)
    total_tokens = sum(n_tokens)
    q = get_q(m_after) - get_q(m_before)
    h = get_h(m_after) - get_h(m_before)
    stats = {
        "level": level,
        "requests": R,
        "wall_s": round(wall, 2),
        "prompt_tokens": total_tokens,
        "input_tps": round(total_tokens / wall, 1),
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3),
        "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3),
        "ttft_p99_s": round(ttfts[min(len(ttfts) - 1, int(0.99 * len(ttfts)))], 3),
        "queries_delta": q,
        "hits_delta": h,
        "hit_rate": round(h / q, 4) if q else None,
        "ttft_all_s": [round(t, 3) for t in ttfts],
    }
    with open(f"{out_dir}/level_{level}.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 服务探活
    with urllib.request.urlopen(API + "/v1/models", timeout=10) as r:
        assert b"data" in r.read(), "server not ready"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ---- 1. 生成前缀池 ----
    rng = random.Random(SEED)
    print(f"[gen] generating {P_PREFIX} prefixes x {PREFIX_TOKENS} tokens ...", flush=True)
    prefixes, meta = [], []
    for i in range(P_PREFIX):
        text, n = make_prefix_text(tok, rng, PREFIX_TOKENS)
        prefixes.append(text)
        meta.append({"prefix_id": i, "tokens": n, "chars": len(text)})
        print(f"[gen] prefix {i}: {n} tokens", flush=True)
    with open(f"{args.out}/prefix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- 2. 灌前缀（warmup，同时完成引擎预热）----
    print("[warmup] seeding prefix cache ...", flush=True)
    m0 = parse_counters(get_metrics_text())
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(warmup_one, prefixes))
    m1 = parse_counters(get_metrics_text())
    wq, wh = get_q(m1) - get_q(m0), get_h(m1) - get_h(m0)
    print(f"[warmup] queries_delta={wq:.0f} hits_delta={wh:.0f} "
          f"(expect ~{P_PREFIX * PREFIX_TOKENS} / 0)", flush=True)
    with open(f"{args.out}/warmup.json", "w") as f:
        json.dump({"queries_delta": wq, "hits_delta": wh}, f, indent=2)

    # ---- 3. 并发扫描 ----
    all_stats = []
    for lv in LEVELS:
        print(f"[bench] level {lv} running ...", flush=True)
        s = run_level(lv, prefixes, tok, args.out)
        all_stats.append(s)
        print(f"[bench] level {lv}: TPS={s['input_tps']} "
              f"TTFT_mean={s['ttft_mean_s']}s hit_rate={s['hit_rate']}", flush=True)
        time.sleep(5)

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("==== SUMMARY ====", flush=True)
    for s in all_stats:
        print(json.dumps({k: v for k, v in s.items() if k != "ttft_all_s"}), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
