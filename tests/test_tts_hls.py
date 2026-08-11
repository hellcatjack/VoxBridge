from __future__ import annotations

import asyncio
import io
import math
import shutil
import struct
import subprocess
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from voxbridge.tts.hls import (
    FFmpegHLSEncoder,
    HLSListenerCapacityExceeded,
    HLSListenerNotFound,
    HLSQueueFull,
    SharedHLSTTSPublisher,
    decode_mono_pcm16_wav,
)
from voxbridge.tts.jobs import TTSReadyItem


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeSynthesizer:
    def __init__(self, wav_bytes: bytes) -> None:
        self.wav_bytes = wav_bytes
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, target_language: str):
        self.calls.append((text, target_language))
        return SimpleNamespace(wav_bytes=self.wav_bytes, sample_rate=24000, duration_ms=100)


class FakeEncoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.start_count = 0
        self.close_count = 0
        self.appended: list[bytes] = []
        self.pending_audio_ms = 0

    async def start(self) -> None:
        self.start_count += 1
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "segment_000000001.ts").write_bytes(b"shared-segment")
        (self.root / "index.m3u8").write_text(
            "#EXTM3U\n#EXTINF:1.0,\nsegment_000000001.ts\n",
            encoding="utf-8",
        )

    async def append_pcm(self, pcm: bytes) -> None:
        self.appended.append(pcm)

    async def wait_ready(self, timeout: float = 5.0) -> None:
        del timeout

    def playlist_text(self) -> str:
        return (self.root / "index.m3u8").read_text(encoding="utf-8")

    def segment_path(self, name: str) -> Path:
        return self.root / name

    async def close(self) -> None:
        self.close_count += 1


def make_wav(*, duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    sample_count = round(sample_rate * duration_ms / 1000)
    samples = [
        round(1200 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(sample_count)
    ]
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def ready_item(
    order: int = 0,
    *,
    revision: int = 1,
    text: str | None = None,
) -> TTSReadyItem:
    return TTSReadyItem(
        sentence_id=f"sentence-{order}",
        revision=revision,
        source_order=order,
        target_language="English",
        text=text or f"Stable translation {order}.",
    )


async def wait_until(predicate, *, timeout: float = 1.0) -> None:
    async def _wait() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait(), timeout=timeout)


@pytest.mark.asyncio
async def test_shared_publisher_synthesizes_once_for_multiple_listeners(tmp_path):
    clock = FakeClock()
    synth = FakeSynthesizer(make_wav())
    encoders: list[FakeEncoder] = []

    def encoder_factory(root: Path) -> FakeEncoder:
        encoder = FakeEncoder(root)
        encoders.append(encoder)
        return encoder

    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=encoder_factory,
        root_dir=tmp_path,
        listener_ttl_sec=60,
        queue_size=8,
        sentence_pause_ms=300,
        clock=clock,
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.touch_listener("iphone-b", "owner-b")
        assert await publisher.publish(ready_item()) is True
        await publisher.wait_idle()

        assert synth.calls == [("Stable translation 0.", "English")]
        assert len(encoders) == 1
        assert encoders[0].start_count == 1
        assert len(encoders[0].appended) == 1
        assert len(encoders[0].appended[0]) == (2400 + 7200) * 2
        assert publisher.listener_count == 2
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_three_listeners_share_one_encoder_and_one_synthesis_per_item(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoders: list[FakeEncoder] = []

    def encoder_factory(root: Path) -> FakeEncoder:
        encoder = FakeEncoder(root)
        encoders.append(encoder)
        return encoder

    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=encoder_factory,
        root_dir=tmp_path,
        listener_ttl_sec=60,
        queue_size=16,
        clock=FakeClock(),
    )
    try:
        for index in range(3):
            await publisher.touch_listener(f"iphone-{index}", f"owner-{index}")
        for index in range(10):
            assert await publisher.publish(ready_item(index)) is True
        await publisher.wait_idle()

        assert synth.calls == [
            (f"Stable translation {index}.", "English") for index in range(10)
        ]
        assert len(encoders) == 1
        assert encoders[0].start_count == 1
        assert len(encoders[0].appended) == 10
        assert publisher.listener_count == 3

        assert await publisher.remove_listener("iphone-1", "owner-1") is True
        assert await publisher.publish(ready_item(10)) is True
        await publisher.wait_idle()
        assert len(synth.calls) == 11
        assert len(encoders) == 1
        assert encoders[0].close_count == 0
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_first_listener_drains_stable_translations_queued_before_join(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        queue_size=4,
        clock=FakeClock(),
    )
    try:
        assert await publisher.publish(ready_item(0)) is True
        assert await publisher.publish(ready_item(1)) is True
        assert synth.calls == []
        assert publisher.status.queue_depth == 2

        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.wait_idle()

        assert synth.calls == [
            ("Stable translation 0.", "English"),
            ("Stable translation 1.", "English"),
        ]
        assert len(encoder.appended) == 2
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_idle_backlog_is_bounded_to_most_recent_stable_translations(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        queue_size=2,
        clock=FakeClock(),
    )
    try:
        for order in range(3):
            assert await publisher.publish(ready_item(order)) is True
        assert publisher.status.queue_depth == 2

        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.wait_idle()

        assert synth.calls == [
            ("Stable translation 1.", "English"),
            ("Stable translation 2.", "English"),
        ]
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_status_counts_item_while_kokoro_synthesis_is_in_flight(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingSynthesizer(FakeSynthesizer):
        def synthesize(self, text: str, target_language: str):
            started.set()
            assert release.wait(timeout=2)
            return super().synthesize(text, target_language)

    publisher = SharedHLSTTSPublisher(
        synthesizer=BlockingSynthesizer(make_wav()),
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        await publisher.publish(ready_item())
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await asyncio.to_thread(started.wait, 1)

        assert publisher.status.synthesis_active is True
        assert publisher.status.queue_depth == 1

        release.set()
        await publisher.wait_idle()
        assert publisher.status.synthesis_active is False
        assert publisher.status.queue_depth == 0
    finally:
        release.set()
        await publisher.close()


@pytest.mark.asyncio
async def test_exact_revision_is_prepared_without_publishing_audio(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    item = ready_item(4, revision=2, text="Prepared exact revision.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")

        assert await publisher.prepare(item) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)

        assert synth.calls == [("Prepared exact revision.", "English")]
        assert encoder.appended == []
        assert publisher.status.preparation_active is False

        assert await publisher.publish(item) is True
        await publisher.wait_idle()

        assert synth.calls == [("Prepared exact revision.", "English")]
        assert len(encoder.appended) == 1
        assert publisher.status.prepared_audio_count == 0
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_new_revision_invalidates_prepared_audio_for_same_sentence(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    revision_one = ready_item(6, revision=1, text="Old translation.")
    revision_two = ready_item(6, revision=2, text="Corrected translation.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(revision_one) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)

        assert await publisher.prepare(revision_two) is True
        await wait_until(
            lambda: publisher.status.prepared_audio_count == 1
            and len(synth.calls) == 2
        )
        assert await publisher.publish(revision_two) is True
        await publisher.wait_idle()

        assert synth.calls == [
            ("Old translation.", "English"),
            ("Corrected translation.", "English"),
        ]
        assert len(encoder.appended) == 1
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_stable_release_supersedes_pending_preparation_without_duplicate_synthesis(
    tmp_path,
):
    worker_gate = asyncio.Event()
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        worker_start_gate=worker_gate,
        clock=FakeClock(),
    )
    item = ready_item(8, revision=3, text="Release takes priority.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        assert await publisher.publish(item) is True

        worker_gate.set()
        await publisher.wait_idle()

        assert synth.calls == [("Release takes priority.", "English")]
        assert len(encoder.appended) == 1
        assert publisher.status.preparation_queue_depth == 0
    finally:
        worker_gate.set()
        await publisher.close()


@pytest.mark.asyncio
async def test_stable_release_reuses_preparation_already_in_flight(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingSynthesizer(FakeSynthesizer):
        def synthesize(self, text: str, target_language: str):
            started.set()
            assert release.wait(timeout=2)
            return super().synthesize(text, target_language)

    synth = BlockingSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    item = ready_item(9, revision=2, text="Already being prepared.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        assert await asyncio.to_thread(started.wait, 1)

        assert await publisher.publish(item) is True
        release.set()
        await publisher.wait_idle()

        assert synth.calls == [("Already being prepared.", "English")]
        assert len(encoder.appended) == 1
        assert publisher.status.prepared_audio_count == 0
    finally:
        release.set()
        await publisher.close()


@pytest.mark.asyncio
async def test_revision_change_during_preparation_discards_stale_audio(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class FirstCallBlockingSynthesizer(FakeSynthesizer):
        def synthesize(self, text: str, target_language: str):
            if not self.calls:
                started.set()
                assert release.wait(timeout=2)
            return super().synthesize(text, target_language)

    synth = FirstCallBlockingSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    stale = ready_item(10, revision=1, text="Stale translation.")
    current = ready_item(10, revision=2, text="Current translation.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(stale) is True
        assert await asyncio.to_thread(started.wait, 1)
        assert await publisher.prepare(current) is True

        release.set()
        await wait_until(
            lambda: publisher.status.prepared_audio_count == 1
            and len(synth.calls) == 2
        )
        assert await publisher.publish(current) is True
        await publisher.wait_idle()

        assert synth.calls == [
            ("Stale translation.", "English"),
            ("Current translation.", "English"),
        ]
        assert len(encoder.appended) == 1
    finally:
        release.set()
        await publisher.close()


@pytest.mark.asyncio
async def test_preparation_is_skipped_without_an_active_listener(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        assert await publisher.prepare(ready_item()) is False
        assert synth.calls == []
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_status_reports_pcm_audio_waiting_for_real_time_encoder(tmp_path):
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 1750
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")

        assert publisher.status.pending_audio_ms == 1750
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_new_producer_session_can_discard_stale_idle_backlog(tmp_path):
    synth = FakeSynthesizer(make_wav())
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        await publisher.publish(ready_item(0))
        await publisher.publish(ready_item(1))

        assert await publisher.discard_idle_backlog() == 2
        assert publisher.status.queue_depth == 0

        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.wait_idle()
        assert synth.calls == []
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_shared_publisher_rejects_foreign_listener_owner(tmp_path):
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        with pytest.raises(HLSListenerNotFound):
            await publisher.touch_listener("iphone-a", "owner-b")
        with pytest.raises(HLSListenerNotFound):
            publisher.playlist_text("iphone-a", "owner-b")
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_shared_publisher_bounds_new_listener_capacity(tmp_path):
    clock = FakeClock()
    encoders: list[FakeEncoder] = []

    def encoder_factory(root: Path) -> FakeEncoder:
        encoder = FakeEncoder(root)
        encoders.append(encoder)
        return encoder

    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=encoder_factory,
        root_dir=tmp_path,
        max_listeners=2,
        clock=clock,
    )
    try:
        await publisher.touch_listener("listener-a", "public:listener-a")
        await publisher.touch_listener("listener-b", "public:listener-b")
        refreshed = await publisher.touch_listener(
            "listener-a",
            "public:listener-a",
        )

        with pytest.raises(HLSListenerCapacityExceeded):
            await publisher.touch_listener("listener-c", "public:listener-c")

        assert refreshed.expires_at == clock.value + 90.0
        assert publisher.listener_count == 2
        assert len(encoders) == 1
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_shared_publisher_expires_idle_lease_and_stops_encoder(tmp_path):
    clock = FakeClock()
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        listener_ttl_sec=10,
        clock=clock,
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        clock.value += 11
        assert await publisher.prune_expired() == 1
        assert publisher.listener_count == 0
        assert encoder.close_count == 1
        assert await publisher.publish(ready_item()) is True
        assert publisher.status.queue_depth == 1

        await publisher.touch_listener("iphone-b", "owner-b")
        await publisher.wait_idle()
        assert publisher.status.queue_depth == 0
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_shared_publisher_queue_is_bounded(tmp_path):
    encoder = FakeEncoder(tmp_path / "stream")
    blocker = asyncio.Event()

    class BlockingSynthesizer(FakeSynthesizer):
        def synthesize(self, text: str, target_language: str):
            del text, target_language
            raise AssertionError("worker should be suspended in this test")

    publisher = SharedHLSTTSPublisher(
        synthesizer=BlockingSynthesizer(make_wav()),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        queue_size=1,
        worker_start_gate=blocker,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.publish(ready_item(0)) is True
        with pytest.raises(HLSQueueFull):
            await publisher.publish(ready_item(1))
    finally:
        await publisher.close()


def test_decode_mono_pcm16_wav_rejects_incompatible_audio():
    with pytest.raises(ValueError, match="sample rate"):
        decode_mono_pcm16_wav(make_wav(sample_rate=16000), expected_rate=24000)


@pytest.mark.asyncio
async def test_ffmpeg_encoder_applies_backpressure_to_pending_pcm(tmp_path):
    encoder = FFmpegHLSEncoder(tmp_path / "live", pcm_queue_size=1)
    encoder._process = SimpleNamespace(returncode=None)
    await encoder.append_pcm(b"\x00\x00")

    blocked_append = asyncio.create_task(encoder.append_pcm(b"\x01\x00"))
    await asyncio.sleep(0)
    assert blocked_append.done() is False

    encoder._pcm_queue.get_nowait()
    encoder._pcm_queue.task_done()
    await asyncio.wait_for(blocked_append, timeout=1)
    encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_encoder_tracks_audio_until_writer_consumes_it(tmp_path):
    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=24000,
        frame_ms=20,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    pcm = bytes(round(24000 * 0.1) * 2)
    await encoder.append_pcm(pcm)

    assert encoder.pending_audio_ms == 100

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        async def consumed() -> None:
            while encoder.pending_audio_ms:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(consumed(), timeout=1)
        assert encoder.pending_audio_ms == 0
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_pending_audio_does_not_charge_frame_padding_to_next_clip(tmp_path):
    first_write = asyncio.Event()

    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data
            first_write.set()

        async def drain(self) -> None:
            return None

    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=24000,
        frame_ms=20,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    await encoder.append_pcm(bytes(100))
    await encoder.append_pcm(bytes(9600))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        await asyncio.wait_for(first_write.wait(), timeout=1)
        await asyncio.sleep(0)
        assert encoder.pending_audio_ms == 200
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_encoder_produces_shared_aac_hls_segment(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg tools are unavailable")
    root = tmp_path / "live"
    encoder = FFmpegHLSEncoder(
        root,
        sample_rate=24000,
        segment_sec=0.5,
        playlist_segments=6,
        frame_ms=50,
    )
    await encoder.start()
    try:
        pcm = decode_mono_pcm16_wav(make_wav(duration_ms=700), expected_rate=24000)
        await encoder.append_pcm(pcm)
        await encoder.wait_ready(timeout=5)
        playlist = encoder.playlist_text()
        segment_name = next(
            line.strip()
            for line in playlist.splitlines()
            if line.strip().endswith(".ts")
        )
        segment = encoder.segment_path(segment_name)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate",
                "-of",
                "default=nw=1",
                str(segment),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "codec_name=aac" in probe.stdout
        assert "sample_rate=24000" in probe.stdout
    finally:
        await encoder.close()
