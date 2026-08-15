from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import pytest

from tools.streaming_soak import (
    EventJSONLWriter,
    SoakConfig,
    build_start_message,
    run_streaming_soak,
)


def _write_wav(
    path: Path,
    *,
    frames: int = 1600,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames * channels * sample_width))


class _FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.incoming.put_nowait(json.dumps({"type": "ready"}))
        self.sent_binary_bytes = 0
        self.start_message: dict[str, object] | None = None
        self.finish_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def recv(self) -> str:
        return await self.incoming.get()

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            self.sent_binary_bytes += len(message)
            return
        payload = json.loads(message)
        if payload["type"] == "start":
            self.start_message = payload
            await self.incoming.put(json.dumps({"type": "started"}))
        elif payload["type"] == "finish":
            self.finish_count += 1
            await self.incoming.put(json.dumps({"type": "final", "text": "complete"}))


class _FakeConnector:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.kwargs: dict[str, object] = {}

    def __call__(self, _url: str, **kwargs):
        self.kwargs = kwargs
        return self.websocket


def test_start_message_carries_direction_context_and_tts_identity():
    message = build_start_message(
        language="Chinese",
        translation_direction="zh2en",
        context_terms=("尼希米记", "耶路撒冷"),
        tts_enabled=True,
        tts_client_id="soak-client-1234",
    )

    assert message == {
        "type": "start",
        "language": "Chinese",
        "translation_direction": "zh2en",
        "asr_context_terms": ["尼希米记", "耶路撒冷"],
        "tts_enabled": True,
        "tts_client_id": "soak-client-1234",
    }


def test_event_writer_appends_and_flushes_without_retaining_events(tmp_path):
    output = tmp_path / "events.jsonl"
    with EventJSONLWriter(output) as writer:
        writer.write({"type": "partial", "text": "one"})
        assert output.read_text(encoding="utf-8").endswith("\n")
        writer.write({"type": "final", "text": "two"})
        assert writer.count == 2
        assert not hasattr(writer, "events")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"type": "partial", "text": "one"},
        {"type": "final", "text": "two"},
    ]


def test_soak_streams_exact_duration_loops_wav_and_forwards_cookie(tmp_path):
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path, frames=1600)
    output = tmp_path / "events.jsonl"
    websocket = _FakeWebSocket()
    connector = _FakeConnector(websocket)
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    config = SoakConfig(
        ws_url="ws://127.0.0.1:8024/ws",
        wav_path=wav_path,
        output_path=output,
        duration_sec=0.25,
        chunk_ms=50,
        realtime_factor=1.0,
        language="Chinese",
        translation_direction="zh2en",
        context_terms=(),
        tts_enabled=False,
        tts_client_id="soak-client-1234",
        cookie_header="voxbridge_session=secret-cookie",
        final_timeout_sec=1.0,
    )

    summary = asyncio.run(
        run_streaming_soak(
            config,
            connector=connector,
            sleep=fake_sleep,
        )
    )

    assert websocket.sent_binary_bytes == int(16000 * 0.25) * 2
    assert websocket.finish_count == 1
    assert len(sleeps) == 5
    assert sleeps == [pytest.approx(0.05)] * 5
    assert connector.kwargs["additional_headers"] == {
        "Cookie": "voxbridge_session=secret-cookie"
    }
    assert summary.final_received is True
    assert summary.error_count == 0
    assert summary.audio_samples_sent == 4000
    assert summary.event_count == 3


@pytest.mark.parametrize(
    ("channels", "sample_width", "sample_rate", "message"),
    [
        (2, 2, 16000, "mono"),
        (1, 1, 16000, "16-bit"),
        (1, 2, 48000, "16000 Hz"),
    ],
)
def test_soak_rejects_incompatible_wav(
    tmp_path,
    channels,
    sample_width,
    sample_rate,
    message,
):
    wav_path = tmp_path / "bad.wav"
    _write_wav(
        wav_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
    )
    config = SoakConfig(
        ws_url="ws://127.0.0.1:8024/ws",
        wav_path=wav_path,
        output_path=tmp_path / "events.jsonl",
        duration_sec=0.1,
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(run_streaming_soak(config, connector=_FakeConnector(_FakeWebSocket())))
