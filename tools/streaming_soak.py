#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import json
import os
import sys
import urllib.parse
import urllib.request
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import websockets


@dataclass(frozen=True)
class SoakConfig:
    ws_url: str
    wav_path: Path
    output_path: Path
    duration_sec: float
    chunk_ms: int = 320
    realtime_factor: float = 1.0
    language: str = ""
    translation_direction: str = "zh2en"
    context_terms: tuple[str, ...] = ()
    tts_enabled: bool = True
    tts_client_id: str = "soak-client"
    cookie_header: str = ""
    final_timeout_sec: float = 60.0


@dataclass(frozen=True)
class SoakSummary:
    audio_samples_sent: int
    event_count: int
    partial_count: int
    committed_count: int
    translation_count: int
    error_count: int
    final_received: bool


class EventJSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.count = 0
        self._file = None

    def __enter__(self) -> "EventJSONLWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def write(self, event: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("event writer is not open")
        self._file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()
        self.count += 1

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def build_start_message(
    *,
    language: str,
    translation_direction: str,
    context_terms: Sequence[str],
    tts_enabled: bool,
    tts_client_id: str,
) -> dict[str, Any]:
    return {
        "type": "start",
        "language": str(language or ""),
        "translation_direction": str(translation_direction or "zh2en"),
        "asr_context_terms": [str(term) for term in context_terms if str(term).strip()],
        "tts_enabled": bool(tts_enabled),
        "tts_client_id": str(tts_client_id),
    }


def login_cookie(login_url: str, username: str, password: str) -> str:
    if not login_url:
        return ""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    body = urllib.parse.urlencode(
        {"username": username, "password": password, "next": "/"}
    ).encode("utf-8")
    request = urllib.request.Request(login_url, data=body, method="POST")
    with opener.open(request, timeout=15) as response:
        if int(response.status) not in {200, 303}:
            raise RuntimeError(f"login failed with HTTP {response.status}")
    cookies = [f"{cookie.name}={cookie.value}" for cookie in jar]
    if not cookies:
        raise RuntimeError("login did not return an authentication cookie")
    return "; ".join(cookies)


def _validate_wav(wav: wave.Wave_read) -> None:
    if wav.getnchannels() != 1:
        raise ValueError("WAV must be mono")
    if wav.getsampwidth() != 2:
        raise ValueError("WAV must be 16-bit PCM")
    if wav.getframerate() != 16000:
        raise ValueError("WAV must use a 16000 Hz sample rate")


async def run_streaming_soak(
    config: SoakConfig,
    *,
    connector: Callable[..., Any] = websockets.connect,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SoakSummary:
    if config.duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if config.chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if config.realtime_factor <= 0:
        raise ValueError("realtime_factor must be positive")

    with wave.open(str(config.wav_path), "rb") as wav:
        _validate_wav(wav)
        target_samples = max(1, round(config.duration_sec * 16000))
        chunk_samples = max(1, round(config.chunk_ms * 16))
        headers = {"Cookie": config.cookie_header} if config.cookie_header else None
        connect_options: dict[str, Any] = {
            "max_size": 16 * 1024 * 1024,
            "ping_interval": 20,
            "ping_timeout": 20,
        }
        if headers is not None:
            connect_options["additional_headers"] = headers

        partial_count = 0
        committed_count = 0
        translation_count = 0
        error_count = 0
        final_received = asyncio.Event()
        error_received = asyncio.Event()
        audio_samples_sent = 0

        with EventJSONLWriter(config.output_path) as writer:
            async with connector(config.ws_url, **connect_options) as websocket:
                ready = json.loads(await websocket.recv())
                writer.write(ready)
                if str(ready.get("type", "")).lower() != "ready":
                    raise RuntimeError(f"unexpected first WebSocket message: {ready.get('type')}")
                await websocket.send(
                    json.dumps(
                        build_start_message(
                            language=config.language,
                            translation_direction=config.translation_direction,
                            context_terms=config.context_terms,
                            tts_enabled=config.tts_enabled,
                            tts_client_id=config.tts_client_id,
                        ),
                        ensure_ascii=False,
                    )
                )
                started = json.loads(await websocket.recv())
                writer.write(started)
                if str(started.get("type", "")).lower() != "started":
                    raise RuntimeError(f"stream start failed: {started.get('message', '')}")

                async def receive_events() -> None:
                    nonlocal partial_count, committed_count, translation_count, error_count
                    while True:
                        raw = await websocket.recv()
                        if isinstance(raw, bytes):
                            continue
                        event = json.loads(raw)
                        writer.write(event)
                        event_type = str(event.get("type", "")).lower()
                        if event_type == "partial":
                            partial_count += 1
                        elif event_type in {"sentence_committed", "sentence_updated"}:
                            committed_count += 1
                        elif event_type == "sentence_translation":
                            translation_count += 1
                        elif event_type == "error":
                            error_count += 1
                            error_received.set()
                            return
                        elif event_type == "final":
                            final_received.set()
                            return

                receiver = asyncio.create_task(receive_events())
                try:
                    while audio_samples_sent < target_samples and not error_received.is_set():
                        take = min(chunk_samples, target_samples - audio_samples_sent)
                        pcm = wav.readframes(take)
                        if not pcm:
                            wav.rewind()
                            continue
                        actual_samples = len(pcm) // 2
                        await websocket.send(pcm)
                        audio_samples_sent += actual_samples
                        await sleep((actual_samples / 16000.0) / config.realtime_factor)
                    if error_received.is_set():
                        raise RuntimeError("server returned an error during streaming")
                    await websocket.send(json.dumps({"type": "finish"}))
                    await asyncio.wait_for(
                        final_received.wait(),
                        timeout=max(1.0, float(config.final_timeout_sec)),
                    )
                    await receiver
                finally:
                    if not receiver.done():
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)

            return SoakSummary(
                audio_samples_sent=audio_samples_sent,
                event_count=writer.count,
                partial_count=partial_count,
                committed_count=committed_count,
                translation_count=translation_count,
                error_count=error_count,
                final_received=final_received.is_set(),
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded-memory VoxBridge stream soak")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8024/ws")
    parser.add_argument("--login-url", default="")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password-env", default="VOXBRIDGE_PASSWORD")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--language", default="")
    parser.add_argument("--translation-direction", choices=("zh2en", "en2zh"), default="zh2en")
    parser.add_argument("--context-term", action="append", default=[])
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-client-id", default="")
    parser.add_argument("--final-timeout-sec", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    password = os.environ.get(str(args.password_env), "")
    try:
        cookie = login_cookie(str(args.login_url), str(args.username), password)
        config = SoakConfig(
            ws_url=str(args.ws_url),
            wav_path=args.wav,
            output_path=args.events_jsonl,
            duration_sec=float(args.duration_sec),
            chunk_ms=int(args.chunk_ms),
            realtime_factor=float(args.realtime_factor),
            language=str(args.language),
            translation_direction=str(args.translation_direction),
            context_terms=tuple(str(term) for term in args.context_term),
            tts_enabled=not bool(args.no_tts),
            tts_client_id=(
                str(args.tts_client_id).strip() or f"soak-{uuid.uuid4().hex}"
            ),
            cookie_header=cookie,
            final_timeout_sec=float(args.final_timeout_sec),
        )
        summary = asyncio.run(run_streaming_soak(config))
    except (OSError, ValueError, RuntimeError, TimeoutError, asyncio.TimeoutError) as exc:
        print(f"soak failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
