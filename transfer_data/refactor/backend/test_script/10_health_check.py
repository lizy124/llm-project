#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--timeout-sec', type=float, default=480.0)
    parser.add_argument('--interval-sec', type=float, default=2.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = run_dir / 'health_check.json'
    url = f'http://{args.host}:{args.port}/v1/models'
    start = time.perf_counter()
    attempts = 0
    last_error = ''
    status = None
    body = ''
    ok = False
    while time.perf_counter() - start < args.timeout_sec:
        attempts += 1
        try:
            with urllib.request.urlopen(url, timeout=min(args.interval_sec, 5.0)) as resp:
                status = resp.status
                body = resp.read().decode('utf-8', errors='replace')
                ok = status == 200
                if ok:
                    break
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(args.interval_sec)
    result = {
        'url': url,
        'ok': ok,
        'status': status,
        'attempts': attempts,
        'elapsed_sec': time.perf_counter() - start,
        'last_error': last_error,
        'body': body[:4000],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
