#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = (
    "You are validating an external KV cache path. "
    "Summarize the following repeated context in three concise bullet points. "
    + "AscendStore KV Pool should reuse a long shared prefix across requests. " * 80
)


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - start
            return resp.status, body, elapsed, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - start
        return exc.code, body, elapsed, str(exc)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, "", elapsed, repr(exc)


def extract_text(response):
    try:
        obj = json.loads(response)
    except Exception:
        return None, None
    choices = obj.get("choices") or []
    text = None
    if choices:
        first = choices[0]
        text = first.get("text")
        if text is None and isinstance(first.get("message"), dict):
            text = first["message"].get("content")
    return text, obj.get("usage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", default="manual")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "requests.jsonl"
    summary_path = run_dir / "summary.json"

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
            }
            status, body, elapsed, error = post_json(url, payload, args.timeout)
            text, usage = extract_text(body)
            record = {
                "case": args.case,
                "index": idx + 1,
                "url": url,
                "model": args.model,
                "status": status,
                "elapsed_sec": elapsed,
                "error": error,
                "prompt_chars": len(prompt),
                "usage": usage,
                "text": text,
                "raw_response": body,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(json.dumps({
                "index": idx + 1,
                "status": status,
                "elapsed_sec": round(elapsed, 4),
                "error": error,
                "usage": usage,
            }, ensure_ascii=False))

    ok = [r for r in records if r["status"] == 200 and not r["error"]]
    summary = {
        "case": args.case,
        "request_count": len(records),
        "success_count": len(ok),
        "failure_count": len(records) - len(ok),
        "elapsed_sec": [r["elapsed_sec"] for r in records],
        "usage": [r["usage"] for r in records],
        "same_text_as_first": [r["text"] == records[0]["text"] for r in records] if records else [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REQUESTS_JSONL={out_path}")
    print(f"SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
