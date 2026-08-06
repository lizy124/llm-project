#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = (
    "You are validating an external KV cache path after an AscendStore backend module reorganization. "
    "Summarize the following repeated context in three concise bullet points. "
    + "AscendStore KV Pool should reuse a long shared prefix across requests. " * 80
)


def iter_sse_lines(response):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[len("data:"):].strip()


def extract_delta_text(obj):
    choices = obj.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if "text" in first and first["text"] is not None:
        return first["text"]
    delta = first.get("delta")
    if isinstance(delta, dict):
        return delta.get("content") or ""
    message = first.get("message")
    if isinstance(message, dict):
        return message.get("content") or ""
    return ""


def post_stream(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_chunk_time = None
    chunks = []
    usage = None
    finish_reason = None
    raw_events = []

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for event in iter_sse_lines(resp):
                now = time.perf_counter()
                raw_events.append(event)
                if event == "[DONE]":
                    break
                try:
                    obj = json.loads(event)
                except Exception:
                    continue
                if first_chunk_time is None:
                    first_chunk_time = now
                usage = obj.get("usage") or usage
                choices = obj.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                text = extract_delta_text(obj)
                if text:
                    chunks.append(text)
            total = time.perf_counter() - start
            ttft = None if first_chunk_time is None else first_chunk_time - start
            return status, "".join(chunks), usage, finish_reason, ttft, total, len(raw_events), None, raw_events[-5:]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        total = time.perf_counter() - start
        return exc.code, "", None, None, None, total, 0, f"{exc}: {body}", []
    except Exception as exc:
        total = time.perf_counter() - start
        return None, "", None, None, None, total, 0, repr(exc), []


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", default="manual_stream")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "stream_requests.jsonl"
    summary_path = run_dir / "stream_summary.json"

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = DEFAULT_PROMPT

    url = f"http://{args.host}:{args.port}/v1/completions"
    records = []

    with out_path.open("a", encoding="utf-8") as f:
        for idx in range(args.repeat):
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            status, text, usage, finish_reason, ttft, total, event_count, error, last_events = post_stream(
                url, payload, args.timeout
            )
            record = {
                "case": args.case,
                "index": idx + 1,
                "url": url,
                "model": args.model,
                "status": status,
                "ttft_sec": ttft,
                "total_sec": total,
                "event_count": event_count,
                "finish_reason": finish_reason,
                "error": error,
                "prompt_chars": len(prompt),
                "usage": usage,
                "text": text,
                "last_events": last_events,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(json.dumps({
                "index": idx + 1,
                "status": status,
                "ttft_sec": None if ttft is None else round(ttft, 4),
                "total_sec": round(total, 4),
                "event_count": event_count,
                "error": error,
                "usage": usage,
            }, ensure_ascii=False))

    ok = [r for r in records if r["status"] == 200 and not r["error"]]
    ttfts = [r["ttft_sec"] for r in ok if r["ttft_sec"] is not None]
    totals = [r["total_sec"] for r in ok]
    summary = {
        "case": args.case,
        "request_count": len(records),
        "success_count": len(ok),
        "failure_count": len(records) - len(ok),
        "ttft_sec": ttfts,
        "total_sec": totals,
        "ttft_avg_sec": sum(ttfts) / len(ttfts) if ttfts else None,
        "ttft_p50_sec": percentile(ttfts, 0.50),
        "ttft_p90_sec": percentile(ttfts, 0.90),
        "total_avg_sec": sum(totals) / len(totals) if totals else None,
        "usage": [r["usage"] for r in records],
        "same_text_as_first": [r["text"] == records[0]["text"] for r in records] if records else [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"STREAM_REQUESTS_JSONL={out_path}")
    print(f"STREAM_SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
