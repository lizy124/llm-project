#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = (
    "Stream validation for AscendStore KV Pool. "
    "Return a short answer with the repeated prefix preserved. "
    + "The fence relocation must not change stream behavior. " * 60
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", default="stream")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "stream_requests.jsonl"
    summary_path = run_dir / "stream_summary.json"
    prompt = DEFAULT_PROMPT
    url = f"http://{args.host}:{args.port}/v1/completions"
    records = []

    with out_path.open("a", encoding="utf-8") as f:
        for idx in range(args.repeat):
            payload = json.dumps({
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.0,
                "stream": True,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            text = []
            error = None
            status = None
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    status = resp.status
                    for raw in resp:
                        chunk = raw.decode("utf-8", errors="replace").strip()
                        if not chunk or chunk == "data: [DONE]":
                            continue
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        try:
                            obj = json.loads(chunk)
                        except Exception:
                            continue
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("text")
                            if delta:
                                text.append(delta)
            except Exception as exc:
                error = repr(exc)
            elapsed = time.perf_counter() - started
            record = {
                "case": args.case,
                "index": idx + 1,
                "status": status,
                "elapsed_sec": elapsed,
                "error": error,
                "text": "".join(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(json.dumps({
                "index": idx + 1,
                "status": status,
                "elapsed_sec": round(elapsed, 4),
                "error": error,
                "text_len": len(record["text"]),
            }, ensure_ascii=False))

    ok = [r for r in records if r["status"] == 200 and not r["error"]]
    summary = {
        "case": args.case,
        "request_count": len(records),
        "success_count": len(ok),
        "failure_count": len(records) - len(ok),
        "elapsed_sec": [r["elapsed_sec"] for r in records],
        "same_text_as_first": [r["text"] == records[0]["text"] for r in records] if records else [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"STREAM_REQUESTS_JSONL={out_path}")
    print(f"STREAM_SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
