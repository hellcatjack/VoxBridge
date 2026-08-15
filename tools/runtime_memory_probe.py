#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from voxbridge.debug.runtime_memory import read_runtime_memory


TTS_STATUS_FIELDS = (
    "available",
    "listener_count",
    "queue_depth",
    "synthesis_active",
    "preparation_queue_depth",
    "preparation_active",
    "prepared_audio_count",
    "pending_audio_ms",
    "encoder_active",
    "producer_active",
    "last_error",
)


def fetch_http(url: str, timeout_sec: float) -> tuple[int | None, dict[str, Any] | None, str]:
    if not str(url or "").strip():
        return None, None, ""
    request = urllib.request.Request(str(url), headers={"Cache-Control": "no-store"})
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, float(timeout_sec))) as response:
            status = int(response.status)
            content_type = str(response.headers.get("Content-Type", ""))
            payload = None
            if "json" in content_type.lower():
                decoded = json.loads(response.read().decode("utf-8"))
                if isinstance(decoded, dict):
                    payload = decoded
            return status, payload, ""
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, type(exc).__name__
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, None, type(exc).__name__


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream VoxBridge cgroup/GTT metrics to JSONL")
    parser.add_argument("--cgroup", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=15.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--health-url", default="")
    parser.add_argument("--tts-status-url", default="")
    parser.add_argument("--http-timeout-sec", type=float, default=2.0)
    return parser.parse_args(argv)


def _status_sample(
    health_url: str,
    tts_status_url: str,
    timeout_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    health_status, _, health_error = fetch_http(health_url, timeout_sec)
    tts_status, tts_payload, tts_error = fetch_http(tts_status_url, timeout_sec)
    health = {"status": health_status, "error": health_error}
    tts: dict[str, Any] = {"status": tts_status, "error": tts_error}
    if tts_payload:
        tts.update({key: tts_payload[key] for key in TTS_STATUS_FIELDS if key in tts_payload})
    return health, tts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval_sec < 0 or args.duration_sec < 0 or args.sample_count < 0:
        print("interval, duration, and sample count must not be negative", file=sys.stderr)
        return 2

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    started = time.monotonic()
    samples = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a", encoding="utf-8", buffering=1) as output:
            while not stop_requested:
                try:
                    runtime = read_runtime_memory(args.cgroup, proc_root=args.proc_root)
                except FileNotFoundError as exc:
                    print(str(exc), file=sys.stderr)
                    return 2
                health, tts_status = _status_sample(
                    str(args.health_url),
                    str(args.tts_status_url),
                    float(args.http_timeout_sec),
                )
                row = {
                    "elapsed_ms": max(0, round((time.monotonic() - started) * 1000)),
                    "runtime": asdict(runtime),
                    "health": health,
                    "tts_status": tts_status,
                }
                output.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                output.flush()
                samples += 1
                if args.sample_count and samples >= args.sample_count:
                    break
                elapsed = time.monotonic() - started
                if args.duration_sec and elapsed >= args.duration_sec:
                    break
                if args.interval_sec:
                    remaining = args.duration_sec - elapsed if args.duration_sec else args.interval_sec
                    time.sleep(min(args.interval_sec, max(0.0, remaining)))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
