#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import websockets

from voxbridge.debug.subtitle_selfcheck import analyze_subtitle_events, summarize_result


def _read_pcm16_mono_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if channels != 1:
            raise ValueError(f"wav must be mono, got channels={channels}")
        if sample_width != 2:
            raise ValueError(f"wav must be 16-bit PCM, got sampwidth={sample_width}")
        if sample_rate != 16000:
            raise ValueError(f"wav must be 16kHz, got sample_rate={sample_rate}")
        return wf.readframes(wf.getnframes())


def _chunk_pcm16(raw: bytes, chunk_ms: int, sample_rate: int = 16000) -> List[bytes]:
    samples_per_chunk = max(1, int(sample_rate * (chunk_ms / 1000.0)))
    bytes_per_chunk = samples_per_chunk * 2
    chunks: List[bytes] = []
    for i in range(0, len(raw), bytes_per_chunk):
        seg = raw[i : i + bytes_per_chunk]
        if seg:
            chunks.append(seg)
    return chunks


async def _recv_loop(ws, events: List[Dict[str, Any]], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        events.append(msg)
        msg_type = str(msg.get("type", "")).lower()
        if msg_type in {"final", "error"}:
            stop.set()


async def _replay_wav(
    ws_url: str,
    wav_path: Path,
    chunk_ms: int,
    language: str,
    realtime_factor: float,
) -> List[Dict[str, Any]]:
    raw_pcm = _read_pcm16_mono_wav(wav_path)
    chunks = _chunk_pcm16(raw_pcm, chunk_ms=chunk_ms, sample_rate=16000)

    events: List[Dict[str, Any]] = []
    stop = asyncio.Event()

    async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
        ready = json.loads(await ws.recv())
        events.append(ready)
        if str(ready.get("type", "")).lower() != "ready":
            raise RuntimeError(f"unexpected first message: {ready}")

        await ws.send(json.dumps({"type": "start", "language": language}))
        recv_task = asyncio.create_task(_recv_loop(ws, events, stop))
        sleep_sec = max(0.0, (chunk_ms / 1000.0) / max(0.01, float(realtime_factor)))

        for chunk in chunks:
            if stop.is_set():
                break
            await ws.send(chunk)
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

        if not stop.is_set():
            await ws.send(json.dumps({"type": "finish"}))

        try:
            await asyncio.wait_for(stop.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass
        stop.set()
        await recv_task

    return events


def _load_events_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            events.append(json.loads(text))
    return events


def _save_events_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay WS ASR stream and self-check subtitle sentence stability.")
    p.add_argument("--ws-url", default="ws://127.0.0.1:8024/ws")
    p.add_argument("--wav", default="", help="16kHz/mono/16-bit PCM wav for replay")
    p.add_argument("--language", default="")
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--realtime-factor", type=float, default=1.0, help="1.0=realtime, 2.0=2x faster")
    p.add_argument("--events-jsonl", default="", help="save replayed events to jsonl; or load existing when --wav omitted")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events_path = Path(args.events_jsonl).expanduser() if args.events_jsonl else None

    if args.wav:
        wav_path = Path(args.wav).expanduser()
        events = asyncio.run(
            _replay_wav(
                ws_url=str(args.ws_url),
                wav_path=wav_path,
                chunk_ms=int(args.chunk_ms),
                language=str(args.language),
                realtime_factor=float(args.realtime_factor),
            )
        )
        if events_path is not None:
            _save_events_jsonl(events_path, events)
    else:
        if events_path is None:
            raise SystemExit("provide --wav for replay, or --events-jsonl to load existing events")
        events = _load_events_jsonl(events_path)

    result = analyze_subtitle_events(events)
    print(summarize_result(result))


if __name__ == "__main__":
    main()

