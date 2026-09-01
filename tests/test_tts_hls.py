from __future__ import annotations

import asyncio
import io
import math
import shutil
import struct
import subprocess
import threading
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import voxbridge.tts.hls as hls_module
from voxbridge.tts.hls import (
    FFmpegHLSEncoder,
    HLSAppendReceipt,
    HLSListenerCapacityExceeded,
    HLSListenerNotFound,
    HLSQueueFull,
    SharedHLSTTSPublisher,
    decode_mono_pcm16_wav,
    parse_hls_live_edge_at_ms,
)
from voxbridge.tts.jobs import TTSReadyItem


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeSynthesizer:
    def __init__(self, wav_bytes: bytes, *, duration_ms: int = 100) -> None:
        self.wav_bytes = wav_bytes
        self.duration_ms = int(duration_ms)
        self.calls: list[tuple[str, str]] = []
        self.speed_calls: list[tuple[str, str, float | None]] = []

    def synthesize(
        self,
        text: str,
        target_language: str,
        *,
        speed: float | None = None,
    ):
        self.calls.append((text, target_language))
        self.speed_calls.append((text, target_language, speed))
        return SimpleNamespace(
            wav_bytes=self.wav_bytes,
            sample_rate=24000,
            duration_ms=self.duration_ms,
        )


class FakeEncoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.start_count = 0
        self.close_count = 0
        self.appended: list[bytes] = []
        self.receipts: list[HLSAppendReceipt] = []
        self.pending_audio_ms = 0
        self.next_start_at_ms = 100_000
        self.next_discardable_gap_before_ms = 0

    async def start(self) -> None:
        self.start_count += 1
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "segment_000000001.ts").write_bytes(b"shared-segment")
        (self.root / "index.m3u8").write_text(
            "#EXTM3U\n#EXTINF:1.0,\nsegment_000000001.ts\n",
            encoding="utf-8",
        )

    async def append_pcm(self, pcm: bytes) -> HLSAppendReceipt:
        self.appended.append(pcm)
        duration_ms = round(len(pcm) * 1000 / (24000 * 2))
        discardable_gap_before_ms = self.next_discardable_gap_before_ms
        self.next_discardable_gap_before_ms = 0
        self.next_start_at_ms += discardable_gap_before_ms
        receipt = HLSAppendReceipt(
            start_at_ms=self.next_start_at_ms,
            end_at_ms=self.next_start_at_ms + duration_ms,
            discardable_gap_before_ms=discardable_gap_before_ms,
        )
        self.receipts.append(receipt)
        self.next_start_at_ms = receipt.end_at_ms
        return receipt

    async def wait_ready(self, timeout: float = 5.0) -> None:
        del timeout

    def playlist_text(self) -> str:
        return (self.root / "index.m3u8").read_text(encoding="utf-8")

    def live_edge_at_ms(self) -> int:
        return self.next_start_at_ms

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


@pytest.mark.parametrize(
    ("backlog_ms", "expected"),
    [
        (0, 1.0),
        (9_999, 1.0),
        (10_000, 1.2),
        (29_999, 1.2),
        (30_000, 1.4),
        (39_999, 1.4),
        (40_000, 1.5),
    ],
)
def test_global_tts_multiplier_boundaries(backlog_ms, expected):
    assert hls_module.select_global_tts_multiplier(backlog_ms) == expected


@pytest.mark.asyncio
async def test_speech_epoch_skips_idle_debt_and_survives_first_listener_exit(
    tmp_path,
):
    synth = FakeSynthesizer(make_wav())
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    try:
        await publisher.publish(ready_item(0))
        await publisher.publish(ready_item(1))
        assert publisher.status.speech_epoch_id == ""
        assert publisher.status.translated_audio_backlog_ms == 0
        assert publisher.status.translated_audio_backlog_count == 0

        await publisher.touch_listener("iphone-a", "owner-a")
        epoch = publisher.status.speech_epoch_id
        assert epoch.startswith("epoch-")
        await publisher.touch_listener("chrome-b", "owner-b")
        await publisher.wait_idle()
        assert synth.calls == [("Stable translation 1.", "English")]
        assert synth.speed_calls == [("Stable translation 1.", "English", 1.05)]

        await publisher.remove_listener("iphone-a", "owner-a")
        assert publisher.status.speech_epoch_id == epoch
        assert publisher.status.listener_count == 1

        await publisher.remove_listener("chrome-b", "owner-b")
        assert publisher.status.speech_epoch_id == ""
        assert publisher.status.global_speed_multiplier == 1.0
        assert publisher.status.translated_audio_backlog_ms == 0
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_worker_applies_global_multiplier_as_absolute_kokoro_speed(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 40_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.publish(ready_item(text="Accelerated together."))
        await publisher.wait_idle()

        assert synth.speed_calls == [
            ("Accelerated together.", "English", pytest.approx(1.575))
        ]
        assert publisher.status.global_speed_multiplier == 1.5
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_release_reuses_speed_selected_when_audio_was_prepared(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 10_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    item = ready_item(text="Keep the selected accelerated voice.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)
        assert synth.speed_calls[-1][2] == pytest.approx(1.26)

        encoder.pending_audio_ms = 0
        assert await publisher.publish(item) is True
        await publisher.wait_idle()

        assert [call[2] for call in synth.speed_calls] == pytest.approx([1.26])
        assert publisher.status.global_speed_multiplier == 1.2
        assert len(encoder.appended) == 1
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_invalid_effective_speed_falls_back_without_stopping_epoch(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 40_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.5,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.publish(ready_item(text="Stay in range."))
        await publisher.wait_idle()

        assert synth.speed_calls[-1][2] == pytest.approx(1.5)
        assert publisher.status.global_speed_multiplier == 1.0
        assert publisher.status.speech_epoch_id.startswith("epoch-")
    finally:
        await publisher.close()


def test_fast_audio_observation_is_normalized_to_baseline_duration(tmp_path):
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        sentence_pause_ms=0,
        clock=FakeClock(),
    )
    item = ready_item(text="abcdefghij")

    publisher._observe_item_audio_ms(item, 1_000, displayed_multiplier=1.5)

    assert publisher._estimate_item_audio_ms(item) == 1_650


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
async def test_first_listener_starts_from_latest_translation_queued_before_join(tmp_path):
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

        assert synth.calls == [("Stable translation 1.", "English")]
        assert len(encoder.appended) == 1
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

        assert synth.calls == [("Stable translation 2.", "English")]
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_status_counts_item_while_kokoro_synthesis_is_in_flight(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingSynthesizer(FakeSynthesizer):
        def synthesize(
            self, text: str, target_language: str, *, speed: float | None = None
        ):
            started.set()
            assert release.wait(timeout=2)
            return super().synthesize(text, target_language, speed=speed)

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
async def test_caption_cue_is_created_only_when_stable_audio_is_published(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250), duration_ms=250)
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    item = ready_item(11, revision=2, text="Prepared exact revision.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)

        assert publisher.caption_snapshot("iphone-a", "owner-a").cues == ()

        assert await publisher.publish(item) is True
        await publisher.wait_idle()
        snapshot = publisher.caption_snapshot("iphone-a", "owner-a")

        assert snapshot.live_edge_at_ms == 100_550
        assert len(snapshot.cues) == 1
        cue = snapshot.cues[0]
        assert cue.text == "Prepared exact revision."
        assert cue.start_at_ms == 100_000
        assert cue.end_at_ms == 100_250
        assert cue.cue_id
        assert encoder.receipts == [
            HLSAppendReceipt(start_at_ms=100_000, end_at_ms=100_550)
        ]
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_caption_cue_excludes_synthesized_edge_silence(tmp_path):
    sample_rate = 24000
    leading_samples = round(sample_rate * 0.08)
    speech_samples = round(sample_rate * 0.25)
    trailing_samples = round(sample_rate * 0.10)
    samples = (
        [0] * leading_samples
        + [
            round(1200 * math.sin(2 * math.pi * 440 * index / sample_rate))
            for index in range(speech_samples)
        ]
        + [0] * trailing_samples
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    synth = FakeSynthesizer(output.getvalue(), duration_ms=430)
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.publish(ready_item(12)) is True
        await publisher.wait_idle()

        cue = publisher.caption_snapshot("iphone-a", "owner-a").cues[0]
        assert cue.start_at_ms == 100_080
        assert cue.end_at_ms == 100_330
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_caption_cue_marks_only_wait_generated_carrier(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=100), duration_ms=100)
    encoder = FakeEncoder(tmp_path / "stream")
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        sentence_pause_ms=300,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.publish(ready_item(20, text="First.")) is True
        await publisher.wait_idle()

        encoder.next_discardable_gap_before_ms = 1_000
        assert await publisher.publish(ready_item(21, text="Second.")) is True
        await publisher.wait_idle()

        first, second = publisher.caption_snapshot(
            "iphone-a", "owner-a"
        ).cues
        assert first.discardable_gap_before_ms == 0
        assert first.resume_at_ms is None
        assert second.discardable_gap_before_ms == 1_000
        assert second.start_at_ms - second.resume_at_ms == 300
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_caption_history_is_bounded_and_cleared_with_listener_epoch(tmp_path):
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
        queue_size=300,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        for order in range(257):
            assert await publisher.publish(ready_item(order)) is True
        await publisher.wait_idle()

        snapshot = publisher.caption_snapshot("iphone-a", "owner-a")
        assert len(snapshot.cues) == 256
        assert snapshot.cues[0].text == "Stable translation 1."
        assert snapshot.cues[-1].text == "Stable translation 256."
        with pytest.raises(HLSListenerNotFound):
            publisher.caption_snapshot("iphone-a", "owner-b")

        assert await publisher.remove_listener("iphone-a", "owner-a") is True
        await publisher.touch_listener("iphone-b", "owner-b")
        assert publisher.caption_snapshot("iphone-b", "owner-b").cues == ()
        assert len(encoders) == 2
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
        def synthesize(
            self, text: str, target_language: str, *, speed: float | None = None
        ):
            started.set()
            assert release.wait(timeout=2)
            return super().synthesize(text, target_language, speed=speed)

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
        def synthesize(
            self, text: str, target_language: str, *, speed: float | None = None
        ):
            if not self.calls:
                started.set()
                assert release.wait(timeout=2)
            return super().synthesize(text, target_language, speed=speed)

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
        assert publisher.status.translated_audio_backlog_ms == 1750
        assert publisher.status.translated_audio_backlog_count == 0
        assert publisher.status.translated_audio_backlog_estimated is False
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_status_includes_prepared_translation_before_hls_publish(tmp_path):
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 1750
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(
            make_wav(duration_ms=250),
            duration_ms=250,
        ),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
        sentence_pause_ms=300,
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(
            ready_item(text="Successfully translated but not released.")
        ) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)

        status = publisher.status
        assert status.pending_audio_ms == 1750
        assert status.translated_audio_backlog_ms == 2300
        assert status.translated_audio_backlog_count == 1
        assert status.translated_audio_backlog_estimated is False
    finally:
        await publisher.close()


@pytest.mark.asyncio
async def test_status_counts_prepared_and_release_queue_overlap_once(tmp_path):
    worker_gate = asyncio.Event()
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 1750
    publisher = SharedHLSTTSPublisher(
        synthesizer=FakeSynthesizer(make_wav()),
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        clock=FakeClock(),
        worker_start_gate=worker_gate,
    )
    item = ready_item(text="One successfully translated sentence.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        assert await publisher.publish(item) is True

        status = publisher.status
        assert status.translated_audio_backlog_count == 1
        assert status.translated_audio_backlog_ms > status.pending_audio_ms
        assert status.translated_audio_backlog_estimated is True
    finally:
        worker_gate.set()
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
        def synthesize(
            self, text: str, target_language: str, *, speed: float | None = None
        ):
            del text, target_language, speed
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
async def test_ffmpeg_append_receipts_follow_media_timeline_fifo(tmp_path):
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
    )
    encoder._process = SimpleNamespace(returncode=None)
    encoder._timeline_origin_at_ms = 100_000

    first = await encoder.append_pcm(bytes(8000 * 2))
    second = await encoder.append_pcm(bytes(4000 * 2))

    assert first == HLSAppendReceipt(start_at_ms=100_128, end_at_ms=101_128)
    assert second == HLSAppendReceipt(start_at_ms=101_128, end_at_ms=101_628)
    encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_append_receipt_reports_only_previously_submitted_carrier(tmp_path):
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
    )
    encoder._process = SimpleNamespace(returncode=None)
    encoder._timeline_origin_at_ms = 100_000
    encoder._scheduled_end_pcm_bytes = 32_000
    encoder._submitted_pcm_bytes = 48_000

    first = await encoder.append_pcm(bytes(8000 * 2))
    second = await encoder.append_pcm(bytes(8000 * 2))

    assert first.discardable_gap_before_ms == 1000
    assert second.discardable_gap_before_ms == 0
    encoder._process = None


def test_hls_live_edge_uses_last_complete_program_date_time_segment():
    playlist = """#EXTM3U
#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:00.000-04:00
#EXTINF:1.024,
segment_000000001.ts
#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:01.024-04:00
#EXTINF:0.512,
segment_000000002.ts
"""

    assert parse_hls_live_edge_at_ms(playlist) == 1_786_456_801_536


def test_hls_live_edge_accepts_ffmpeg_program_time_after_extinf():
    playlist = """#EXTM3U
#EXTINF:1.024000,
#EXT-X-PROGRAM-DATE-TIME:2026-08-11T00:29:06.816-0400
segment_000000000.ts
"""

    assert parse_hls_live_edge_at_ms(playlist) == 1_786_422_547_840


@pytest.mark.parametrize(
    "playlist",
    [
        "",
        "#EXTM3U\n#EXTINF:1.0,\nsegment_000000001.ts\n",
        (
            "#EXTM3U\n"
            "#EXT-X-PROGRAM-DATE-TIME:not-a-date\n"
            "#EXTINF:1.0,\nsegment_000000001.ts\n"
        ),
        (
            "#EXTM3U\n"
            "#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:00\n"
            "#EXTINF:1.0,\nsegment_000000001.ts\n"
        ),
        (
            "#EXTM3U\n"
            "#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:00-04:00\n"
            "#EXTINF:1.0,\n"
        ),
    ],
)
def test_hls_live_edge_rejects_incomplete_or_invalid_playlist(playlist):
    assert parse_hls_live_edge_at_ms(playlist) is None


@pytest.mark.asyncio
async def test_ffmpeg_encoder_applies_backpressure_to_pending_pcm(tmp_path):
    encoder = FFmpegHLSEncoder(tmp_path / "live", pcm_queue_size=1)
    encoder._process = SimpleNamespace(returncode=None)
    encoder._timeline_origin_at_ms = 100_000
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
    encoder._timeline_origin_at_ms = 100_000
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
async def test_ffmpeg_encoder_limits_two_x_burst_to_two_seconds(
    tmp_path,
    monkeypatch,
):
    real_sleep = asyncio.sleep
    delays: list[float] = []
    wrote_twenty_one_frames = asyncio.Event()

    async def record_delay(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 21:
            wrote_twenty_one_frames.set()
        await real_sleep(0)

    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    monkeypatch.setattr(asyncio, "sleep", record_delay)
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
        frame_ms=100,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    encoder._timeline_origin_at_ms = 100_000
    await encoder.append_pcm(bytes(round(8000 * 3.0) * 2))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        await asyncio.wait_for(wrote_twenty_one_frames.wait(), timeout=1)

        assert delays[:20] == pytest.approx([0.05] * 20)
        assert delays[20] == pytest.approx(0.1)
        assert encoder.pending_audio_ms > 0
        assert writer.done() is False
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_encoder_does_not_reset_burst_between_adjacent_clips(
    tmp_path,
    monkeypatch,
):
    real_sleep = asyncio.sleep
    delays: list[float] = []
    wrote_twenty_one_frames = asyncio.Event()

    async def record_delay(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 21:
            wrote_twenty_one_frames.set()
        await real_sleep(0)

    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    monkeypatch.setattr(asyncio, "sleep", record_delay)
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
        frame_ms=100,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    encoder._timeline_origin_at_ms = 100_000
    await encoder.append_pcm(bytes(round(8000 * 1.5) * 2))
    await encoder.append_pcm(bytes(round(8000 * 1.5) * 2))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        await asyncio.wait_for(wrote_twenty_one_frames.wait(), timeout=1)

        assert delays[:20] == pytest.approx([0.05] * 20)
        assert delays[20] == pytest.approx(0.1)
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None


@pytest.mark.asyncio
async def test_ffmpeg_encoder_resets_burst_after_genuine_queue_starvation(
    tmp_path,
    monkeypatch,
):
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    async def skip_tail_flush(*, frame_sec: float) -> None:
        del frame_sec

    monkeypatch.setattr(asyncio, "sleep", record_delay)
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
        frame_ms=100,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    encoder._timeline_origin_at_ms = 100_000
    encoder._flush_tail_until_visible = skip_tail_flush
    await encoder.append_pcm(bytes(round(8000 * 2.1) * 2))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        while encoder.pending_audio_ms:
            await real_sleep(0)
        await real_sleep(0)
        delay_count_before_resume = len(delays)

        await encoder.append_pcm(bytes(round(8000 * 0.1) * 2))
        while len(delays) == delay_count_before_resume:
            await real_sleep(0)

        assert delays[20] == pytest.approx(0.1)
        assert delays[delay_count_before_resume] == pytest.approx(0.05)
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
    encoder._timeline_origin_at_ms = 100_000
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
        await asyncio.sleep(1.0)
        playlist = encoder.playlist_text()
        probe = None
        for segment_name in (
            line.strip()
            for line in playlist.splitlines()
            if line.strip().endswith(".ts")
        ):
            candidate = subprocess.run(
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
                    str(encoder.segment_path(segment_name)),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if candidate.returncode == 0 and "codec_name=aac" in candidate.stdout:
                probe = candidate
                break
        assert probe is not None
        assert "codec_name=aac" in probe.stdout
        assert "sample_rate=24000" in probe.stdout
    finally:
        await encoder.close()


@pytest.mark.asyncio
async def test_ffmpeg_encoder_bootstrap_is_decodable_and_does_not_keep_advancing(
    tmp_path,
):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg tools are unavailable")
    root = tmp_path / "idle-live"
    encoder = FFmpegHLSEncoder(
        root,
        sample_rate=24000,
        segment_sec=0.5,
        playlist_segments=6,
        frame_ms=50,
    )
    await encoder.start()
    try:
        await encoder.wait_ready(timeout=5)
        await asyncio.sleep(0.7)
        playlist = encoder.playlist_text()
        bootstrap_segments = [
            line.strip()
            for line in playlist.splitlines()
            if line.strip().endswith(".ts")
        ]
        assert bootstrap_segments
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
                str(encoder.segment_path(bootstrap_segments[0])),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        assert "codec_name=aac" in probe.stdout
        assert "sample_rate=24000" in probe.stdout

        while encoder._bootstrap_pcm_bytes_remaining:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        idle_playlist = encoder.playlist_text()
        idle_edge = parse_hls_live_edge_at_ms(idle_playlist)

        await asyncio.sleep(0.8)

        assert encoder.playlist_text() == idle_playlist
        assert parse_hls_live_edge_at_ms(encoder.playlist_text()) == idle_edge
        assert encoder.pending_audio_ms == 0
    finally:
        await encoder.close()


@pytest.mark.asyncio
async def test_ffmpeg_encoder_resumes_after_idle_with_bounded_tail_padding(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")
    encoder = FFmpegHLSEncoder(
        tmp_path / "speech-only-live",
        sample_rate=24000,
        segment_sec=0.5,
        playlist_segments=8,
        frame_ms=50,
    )
    tone = decode_mono_pcm16_wav(
        make_wav(duration_ms=1200),
        expected_rate=24000,
    )
    await encoder.start()
    try:
        await encoder.wait_ready(timeout=5)
        while encoder._bootstrap_pcm_bytes_remaining:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        bootstrap_edge = parse_hls_live_edge_at_ms(encoder.playlist_text())
        assert bootstrap_edge is not None

        first = await encoder.append_pcm(tone)
        await wait_until(lambda: encoder.pending_audio_ms == 0, timeout=3)
        await wait_until(
            lambda: (
                parse_hls_live_edge_at_ms(encoder.playlist_text()) or 0
            ) > bootstrap_edge,
            timeout=3,
        )
        await asyncio.sleep(0.3)
        first_idle_playlist = encoder.playlist_text()
        first_idle_edge = parse_hls_live_edge_at_ms(first_idle_playlist)

        await asyncio.sleep(0.8)
        assert encoder.playlist_text() == first_idle_playlist

        second = await encoder.append_pcm(tone)
        tail_padding_ms = second.start_at_ms - first.end_at_ms
        assert 0 <= tail_padding_ms <= 600
        await wait_until(lambda: encoder.pending_audio_ms == 0, timeout=3)
        await wait_until(
            lambda: (
                parse_hls_live_edge_at_ms(encoder.playlist_text()) or 0
            ) > (first_idle_edge or 0),
            timeout=3,
        )
    finally:
        await encoder.close()


@pytest.mark.asyncio
async def test_ffmpeg_encoder_publishes_complete_latest_clip_before_pausing(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")
    encoder = FFmpegHLSEncoder(
        tmp_path / "complete-latest-clip",
        sample_rate=24000,
        segment_sec=1.0,
        frame_ms=100,
    )
    tone = decode_mono_pcm16_wav(
        make_wav(duration_ms=2500),
        expected_rate=24000,
    )
    await encoder.start()
    try:
        await encoder.wait_ready(timeout=5)
        while encoder._bootstrap_pcm_bytes_remaining:
            await asyncio.sleep(0.02)

        receipt = await encoder.append_pcm(tone)
        await wait_until(lambda: encoder.pending_audio_ms == 0, timeout=3)
        await wait_until(
            lambda: (
                parse_hls_live_edge_at_ms(encoder.playlist_text()) or 0
            ) >= receipt.end_at_ms,
            timeout=3,
        )
        completed_playlist = encoder.playlist_text()

        await asyncio.sleep(1.2)

        assert encoder.playlist_text() == completed_playlist
    finally:
        await encoder.close()


@pytest.mark.asyncio
async def test_ffmpeg_append_receipt_matches_decoded_hls_audio_timeline(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")
    root = tmp_path / "live-sync"
    encoder = FFmpegHLSEncoder(
        root,
        sample_rate=24000,
        segment_sec=1.0,
        frame_ms=100,
    )
    tone_pcm = struct.pack(
        "<28800h",
        *[
            round(12000 * math.sin(2 * math.pi * 997 * index / 24000))
            for index in range(28800)
        ],
    )

    await encoder.start()
    try:
        await encoder.wait_ready(timeout=5)
        await asyncio.sleep(0.2)
        receipt = await encoder.append_pcm(tone_pcm)
        await asyncio.sleep(3.0)
    finally:
        await encoder.close()

    playlist = encoder.playlist_path.read_text(encoding="utf-8")
    ended_playlist = root / "ended.m3u8"
    ended_playlist.write_text(playlist + "#EXT-X-ENDLIST\n", encoding="utf-8")
    decoded_path = root / "decoded.pcm"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(ended_playlist),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-f",
            "s16le",
            "-y",
            str(decoded_path),
        ],
        check=True,
    )
    first_program_line = next(
        line
        for line in playlist.splitlines()
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:")
    )
    timeline_origin_ms = round(
        datetime.fromisoformat(first_program_line.split(":", 1)[1]).timestamp()
        * 1000
    )
    decoded = decoded_path.read_bytes()
    samples = struct.unpack(f"<{len(decoded) // 2}h", decoded)
    window_samples = 240
    audible_windows = []
    for offset in range(0, len(samples) - window_samples + 1, window_samples):
        window = samples[offset : offset + window_samples]
        rms = math.sqrt(sum(value * value for value in window) / len(window))
        if rms >= 1000:
            audible_windows.append(offset // window_samples)
    assert audible_windows
    actual_audio_start_ms = timeline_origin_ms + audible_windows[0] * 10

    assert abs(receipt.start_at_ms - actual_audio_start_ms) <= 20
