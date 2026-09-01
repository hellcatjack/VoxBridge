from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import shutil
import sys
import time
import uuid
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from .jobs import TTSReadyItem


logger = logging.getLogger(__name__)

ACTIVE_PCM_BURST_RATE = 2.0
ACTIVE_PCM_BURST_MEDIA_SEC = 2.0
TAIL_PUBLISH_RATE = 2.0
DEFAULT_ENGLISH_AUDIO_MS_PER_CHAR = 65.0
DEFAULT_CHINESE_AUDIO_MS_PER_CHAR = 180.0
MIN_ESTIMATED_SENTENCE_AUDIO_MS = 500
DURATION_ESTIMATE_ALPHA = 0.25
ItemKey = tuple[str, int, str, str]


def select_global_tts_multiplier(backlog_ms: int) -> float:
    """Return the shared Auto multiplier for conservative unpublished speech."""

    value = max(0, int(backlog_ms))
    if value >= 40_000:
        return 1.5
    if value >= 30_000:
        return 1.4
    if value >= 10_000:
        return 1.2
    return 1.0


class HLSError(Exception):
    """Base error for the shared translated-speech stream."""


class HLSUnavailable(HLSError):
    """Raised when the shared encoder cannot serve media."""


class HLSListenerNotFound(HLSError):
    """Raised for an absent, expired, or foreign listener lease."""


class HLSListenerCapacityExceeded(HLSError):
    """Raised when a new listener would exceed the active lease limit."""


class HLSQueueFull(HLSError):
    """Raised instead of growing the pending speech queue without bound."""


@dataclass(frozen=True, slots=True)
class HLSListenerLease:
    listener_id: str
    owner_key: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class HLSAppendReceipt:
    start_at_ms: int
    end_at_ms: int
    discardable_gap_before_ms: int = 0


@dataclass(frozen=True, slots=True)
class HLSCaptionCue:
    cue_id: str
    start_at_ms: int
    end_at_ms: int
    text: str
    discardable_gap_before_ms: int = 0
    resume_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class HLSCaptionSnapshot:
    live_edge_at_ms: int | None
    cues: tuple[HLSCaptionCue, ...]


@dataclass(frozen=True, slots=True)
class HLSStreamStatus:
    available: bool
    listener_count: int
    queue_depth: int
    synthesis_active: bool
    preparation_queue_depth: int
    preparation_active: bool
    prepared_audio_count: int
    pending_audio_ms: int
    translated_audio_backlog_ms: int
    translated_audio_backlog_count: int
    translated_audio_backlog_estimated: bool
    speech_epoch_id: str
    global_speed_mode: str
    global_speed_multiplier: float
    tts_effective_speed: float
    encoder_active: bool
    last_error: str


class HLSEncoder(Protocol):
    root: Path

    @property
    def pending_audio_ms(self) -> int: ...

    async def start(self) -> None: ...

    async def append_pcm(self, pcm: bytes) -> HLSAppendReceipt | None: ...

    async def wait_ready(self, timeout: float = 5.0) -> None: ...

    def playlist_text(self) -> str: ...

    def live_edge_at_ms(self) -> int | None: ...

    def segment_path(self, name: str) -> Path: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _PreparedAudio:
    pcm: bytes
    audio_ms: int
    cue_start_offset_ms: int
    cue_end_offset_ms: int
    synthesis_ms: int
    prepared_at: float
    displayed_multiplier: float
    effective_speed: float


def decode_mono_pcm16_wav(wav_bytes: bytes, *, expected_rate: int) -> bytes:
    """Return validated mono signed-16 PCM from a synthesized WAV."""

    try:
        with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as source:
            channels = int(source.getnchannels())
            sample_width = int(source.getsampwidth())
            sample_rate = int(source.getframerate())
            compression = str(source.getcomptype())
            frames = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as exc:
        raise ValueError("invalid synthesized WAV") from exc
    if compression != "NONE":
        raise ValueError("synthesized WAV must contain uncompressed PCM")
    if channels != 1:
        raise ValueError("synthesized WAV must be mono")
    if sample_width != 2:
        raise ValueError("synthesized WAV must contain 16-bit PCM")
    if sample_rate != int(expected_rate):
        raise ValueError(
            f"synthesized WAV sample rate must be {int(expected_rate)}, got {sample_rate}"
        )
    if not frames:
        raise ValueError("synthesized WAV contains no audio")
    return frames


def _pcm_activity_bounds_ms(pcm: bytes, *, sample_rate: int) -> tuple[int, int]:
    """Find synthesized speech edges without interpreting its language or text."""

    data = bytes(pcm)
    duration_ms = max(1, round(len(data) * 1000 / (sample_rate * 2)))
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    window_samples = max(1, round(sample_rate * 0.01))
    window_energy: list[float] = []
    for start in range(0, len(samples), window_samples):
        window = samples[start : start + window_samples]
        if len(window) < window_samples:
            break
        window_energy.append(
            sum(int(value) * int(value) for value in window) / len(window)
        )
    if not window_energy:
        return 0, duration_ms
    peak_rms = max(window_energy) ** 0.5
    threshold_rms = max(32.0, peak_rms * 0.03)
    threshold_energy = threshold_rms * threshold_rms
    active = [
        index
        for index, energy in enumerate(window_energy)
        if energy >= threshold_energy
    ]
    if not active:
        return 0, duration_ms
    start_ms = round(active[0] * window_samples * 1000 / sample_rate)
    end_ms = min(
        duration_ms,
        round((active[-1] + 1) * window_samples * 1000 / sample_rate),
    )
    return start_ms, max(start_ms + 1, end_ms)


def _parse_hls_timeline_bounds_at_ms(
    playlist: str,
) -> tuple[int, int] | None:
    program_time: datetime | None = None
    duration_sec: float | None = None
    first_start_ms: int | None = None
    last_end_ms: int | None = None
    for raw_line in str(playlist or "").splitlines():
        line = raw_line.strip()
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            value = line.split(":", 1)[1].strip()
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                program_time = None
            else:
                program_time = parsed if parsed.tzinfo is not None else None
            continue
        if line.startswith("#EXTINF:"):
            value = line.split(":", 1)[1].split(",", 1)[0].strip()
            try:
                parsed_duration = float(value)
            except ValueError:
                duration_sec = None
            else:
                duration_sec = parsed_duration if parsed_duration > 0 else None
            continue
        if not line or line.startswith("#"):
            continue
        if program_time is not None and duration_sec is not None:
            start_ms = round(program_time.timestamp() * 1000.0)
            if first_start_ms is None:
                first_start_ms = start_ms
            last_end_ms = round(
                (program_time.timestamp() + duration_sec) * 1000.0
            )
        program_time = None
        duration_sec = None
    if first_start_ms is None or last_end_ms is None:
        return None
    return first_start_ms, last_end_ms


def parse_hls_timeline_origin_at_ms(playlist: str) -> int | None:
    """Return the wall-clock start of the first complete HLS media segment."""

    bounds = _parse_hls_timeline_bounds_at_ms(playlist)
    return bounds[0] if bounds is not None else None


def parse_hls_live_edge_at_ms(playlist: str) -> int | None:
    """Return the wall-clock end of the last complete HLS media segment."""

    bounds = _parse_hls_timeline_bounds_at_ms(playlist)
    return bounds[1] if bounds is not None else None


class FFmpegHLSEncoder:
    """Encode one real-time mono PCM timeline into shared audio-only HLS."""

    def __init__(
        self,
        root: Path,
        *,
        sample_rate: int = 24000,
        segment_sec: float = 1.0,
        playlist_segments: int = 1200,
        bitrate: str = "64k",
        ffmpeg_path: str = "ffmpeg",
        frame_ms: int = 100,
        pcm_queue_size: int = 8,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = max(8000, int(sample_rate))
        self.segment_sec = max(0.25, float(segment_sec))
        self.playlist_segments = max(3, int(playlist_segments))
        self.bitrate = str(bitrate or "64k")
        self.ffmpeg_path = str(ffmpeg_path or "ffmpeg")
        self.frame_ms = max(20, int(frame_ms))
        self._frame_bytes = max(
            1,
            round(self.sample_rate * self.frame_ms / 1000),
        ) * 2
        frame_samples = self._frame_bytes // 2
        idle_carrier = array(
            "h",
            (
                round(2 * math.sin(2 * math.pi * 1000 * index / self.sample_rate))
                for index in range(frame_samples)
            ),
        )
        if sys.byteorder != "little":
            idle_carrier.byteswap()
        # Exact digital silence is optimized into table-only MPEG-TS segments by
        # FFmpeg's AAC encoder. This -84 dBFS carrier keeps idle HLS decodable.
        self._idle_carrier_pcm = idle_carrier.tobytes()
        self._pcm_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=max(1, int(pcm_queue_size))
        )
        self._process: asyncio.subprocess.Process | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()
        self._pending_pcm_bytes = 0
        self._submitted_pcm_bytes = 0
        self._scheduled_end_pcm_bytes = 0
        self._bootstrap_pcm_bytes_total = 0
        self._bootstrap_pcm_bytes_remaining = 0
        self._tail_flush_target_at_ms: int | None = None
        self._timeline_origin_at_ms: int | None = None
        self._closed = False

    @property
    def pending_audio_ms(self) -> int:
        bytes_per_second = self.sample_rate * 2
        return max(0, round(self._pending_pcm_bytes * 1000 / bytes_per_second))

    @property
    def playlist_path(self) -> Path:
        return self.root / "index.m3u8"

    def _required_bootstrap_pcm_bytes(self) -> int:
        frame_sec = self._frame_bytes / (self.sample_rate * 2)
        aac_frame_sec = 1024 / self.sample_rate
        # Cross one HLS boundary and leave enough PCM for the writer and AAC
        # encoders to emit the packet that finalizes the first segment.
        bootstrap_sec = self.segment_sec + frame_sec + aac_frame_sec
        frame_count = max(1, math.ceil(bootstrap_sec / frame_sec))
        return frame_count * self._frame_bytes

    async def start(self) -> None:
        if self._process is not None:
            return
        if shutil.which(self.ffmpeg_path) is None:
            raise HLSUnavailable("FFmpeg executable is unavailable")
        self.root.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            self.bitrate,
            "-f",
            "hls",
            "-hls_time",
            f"{self.segment_sec:g}",
            "-hls_list_size",
            str(self.playlist_segments),
            "-hls_flags",
            "delete_segments+omit_endlist+independent_segments+program_date_time",
            "-hls_segment_filename",
            str(self.root / "segment_%09d.ts"),
            str(self.playlist_path),
        ]
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise HLSUnavailable("failed to start FFmpeg") from exc
        self._bootstrap_pcm_bytes_total = self._required_bootstrap_pcm_bytes()
        self._bootstrap_pcm_bytes_remaining = self._bootstrap_pcm_bytes_total
        self._scheduled_end_pcm_bytes = max(
            self._scheduled_end_pcm_bytes,
            self._bootstrap_pcm_bytes_total,
        )
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def append_pcm(self, pcm: bytes) -> HLSAppendReceipt:
        if self._closed or self._process is None:
            raise HLSUnavailable("HLS encoder is not running")
        data = bytes(pcm)
        if not data or len(data) % 2:
            raise ValueError("PCM payload must contain complete signed-16 samples")
        if self._timeline_origin_at_ms is None:
            await self.wait_ready(timeout=5.0)
        timeline_origin_at_ms = self._timeline_origin_at_ms
        if timeline_origin_at_ms is None:
            raise HLSUnavailable("HLS media timeline is unavailable")
        await self._pcm_queue.put(data)
        bytes_per_second = self.sample_rate * 2
        discardable_gap_bytes = max(
            0,
            self._submitted_pcm_bytes - self._scheduled_end_pcm_bytes,
        )
        start_pcm_bytes = max(
            self._submitted_pcm_bytes,
            self._scheduled_end_pcm_bytes,
        )
        padded_bytes = (
            (len(data) + self._frame_bytes - 1) // self._frame_bytes
        ) * self._frame_bytes
        self._scheduled_end_pcm_bytes = start_pcm_bytes + padded_bytes
        self._pending_pcm_bytes += len(data)
        # MPEG-TS AAC exposes one 1024-sample encoder frame before new PCM is audible.
        aac_priming_bytes = 1024 * 2
        start_at_ms = timeline_origin_at_ms + round(
            (start_pcm_bytes + aac_priming_bytes) * 1000.0 / bytes_per_second
        )
        end_at_ms = round(
            start_at_ms + len(data) * 1000.0 / bytes_per_second
        )
        if (
            self._tail_flush_target_at_ms is None
            or end_at_ms > self._tail_flush_target_at_ms
        ):
            self._tail_flush_target_at_ms = end_at_ms
        discardable_gap_before_ms = round(
            discardable_gap_bytes * 1000.0 / bytes_per_second
        )
        return HLSAppendReceipt(
            start_at_ms=start_at_ms,
            end_at_ms=end_at_ms,
            discardable_gap_before_ms=discardable_gap_before_ms,
        )

    async def _flush_tail_until_visible(self, *, frame_sec: float) -> None:
        target_at_ms = self._tail_flush_target_at_ms
        if target_at_ms is None:
            return
        flush_pcm_bytes_remaining = self._required_bootstrap_pcm_bytes()
        bytes_per_second = self.sample_rate * 2
        flush_media_sec = flush_pcm_bytes_remaining / bytes_per_second
        deadline = asyncio.get_running_loop().time() + max(
            1.0,
            flush_media_sec / TAIL_PUBLISH_RATE + 0.75,
        )
        while self._pcm_queue.empty():
            live_edge_at_ms = self.live_edge_at_ms()
            if live_edge_at_ms is not None and live_edge_at_ms >= target_at_ms:
                if self._tail_flush_target_at_ms == target_at_ms:
                    self._tail_flush_target_at_ms = None
                return
            if flush_pcm_bytes_remaining <= 0:
                if asyncio.get_running_loop().time() >= deadline:
                    logger.warning(
                        "shared HLS tail did not become visible target_at_ms=%d live_edge_at_ms=%s",
                        target_at_ms,
                        live_edge_at_ms,
                    )
                    if self._tail_flush_target_at_ms == target_at_ms:
                        self._tail_flush_target_at_ms = None
                    return
                await asyncio.sleep(0.02)
                continue
            process = self._process
            if (
                process is None
                or process.returncode is not None
                or process.stdin is None
            ):
                raise HLSUnavailable("FFmpeg exited while flushing HLS tail")
            process.stdin.write(self._idle_carrier_pcm)
            self._submitted_pcm_bytes += len(self._idle_carrier_pcm)
            flush_pcm_bytes_remaining = max(
                0,
                flush_pcm_bytes_remaining - len(self._idle_carrier_pcm),
            )
            await process.stdin.drain()
            await asyncio.sleep(frame_sec / TAIL_PUBLISH_RATE)

    async def _writer_loop(self) -> None:
        frame_bytes = self._frame_bytes
        frame_samples = frame_bytes // 2
        frame_sec = frame_samples / self.sample_rate
        active = b""
        active_burst_bytes_remaining = round(
            self.sample_rate * 2 * ACTIVE_PCM_BURST_MEDIA_SEC
        )
        try:
            while True:
                bootstrapping = self._bootstrap_pcm_bytes_remaining > 0
                if bootstrapping:
                    chunk = self._idle_carrier_pcm
                    self._bootstrap_pcm_bytes_remaining = max(
                        0,
                        self._bootstrap_pcm_bytes_remaining - frame_bytes,
                    )
                    writing_audio = False
                    consumed_bytes = 0
                else:
                    if not active:
                        try:
                            active = self._pcm_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            await self._flush_tail_until_visible(frame_sec=frame_sec)
                            active = await self._pcm_queue.get()
                            active_burst_bytes_remaining = round(
                                self.sample_rate * 2 * ACTIVE_PCM_BURST_MEDIA_SEC
                            )
                    writing_audio = True
                    chunk = active[:frame_bytes]
                    consumed_bytes = len(chunk)
                    active = active[len(chunk) :]
                    if len(chunk) < frame_bytes:
                        chunk += bytes(frame_bytes - len(chunk))
                process = self._process
                if process is None or process.returncode is not None or process.stdin is None:
                    raise HLSUnavailable("FFmpeg exited while streaming")
                process.stdin.write(chunk)
                self._submitted_pcm_bytes += len(chunk)
                await process.stdin.drain()
                if writing_audio:
                    self._pending_pcm_bytes = max(
                        0,
                        self._pending_pcm_bytes - consumed_bytes,
                    )
                    if not active:
                        self._pcm_queue.task_done()
                publish_rate = 1.0
                if writing_audio and active_burst_bytes_remaining > 0:
                    publish_rate = ACTIVE_PCM_BURST_RATE
                    active_burst_bytes_remaining = max(
                        0,
                        active_burst_bytes_remaining - len(chunk),
                    )
                await asyncio.sleep(frame_sec / publish_rate)
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionResetError, HLSUnavailable) as exc:
            logger.warning("shared HLS encoder stopped: %s", type(exc).__name__)

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                chunk = await process.stderr.read(1024)
                if not chunk:
                    return
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > 4096:
                    del self._stderr_tail[:-4096]
        except asyncio.CancelledError:
            raise

    async def wait_ready(self, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + max(0.1, float(timeout))
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise HLSUnavailable("FFmpeg exited before producing a playlist")
            if self.playlist_path.is_file():
                text = self.playlist_path.read_text(encoding="utf-8", errors="replace")
                timeline_origin_at_ms = parse_hls_timeline_origin_at_ms(text)
                if timeline_origin_at_ms is not None:
                    if self._timeline_origin_at_ms is None:
                        self._timeline_origin_at_ms = timeline_origin_at_ms
                    return
            await asyncio.sleep(0.05)
        raise HLSUnavailable("HLS playlist was not ready in time")

    def playlist_text(self) -> str:
        try:
            return self.playlist_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HLSUnavailable("HLS playlist is unavailable") from exc

    def live_edge_at_ms(self) -> int | None:
        try:
            return parse_hls_live_edge_at_ms(self.playlist_text())
        except HLSUnavailable:
            return None

    def segment_path(self, name: str) -> Path:
        value = str(name or "")
        if not value.startswith("segment_") or not value.endswith(".ts"):
            raise HLSUnavailable("invalid HLS segment")
        if Path(value).name != value:
            raise HLSUnavailable("invalid HLS segment")
        path = self.root / value
        if not path.is_file():
            raise HLSUnavailable("HLS segment is unavailable")
        return path

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_pcm_bytes = 0
        self._submitted_pcm_bytes = 0
        self._scheduled_end_pcm_bytes = 0
        self._bootstrap_pcm_bytes_total = 0
        self._bootstrap_pcm_bytes_remaining = 0
        self._tail_flush_target_at_ms = None
        self._timeline_origin_at_ms = None
        writer = self._writer_task
        self._writer_task = None
        if writer is not None:
            writer.cancel()
            try:
                await writer
            except asyncio.CancelledError:
                pass
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass


class SharedHLSTTSPublisher:
    """Own one TTS/encoder pipeline and fan its HLS files out to leases."""

    def __init__(
        self,
        *,
        synthesizer: object,
        root_dir: Path,
        encoder_factory: Callable[[Path], HLSEncoder] | None = None,
        listener_ttl_sec: float = 90.0,
        max_listeners: int = 128,
        queue_size: int = 128,
        preparation_cache_size: int = 8,
        caption_history_size: int = 256,
        sample_rate: int = 24000,
        sentence_pause_ms: int = 300,
        baseline_tts_speed: float = 1.05,
        auto_speed_enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        worker_start_gate: asyncio.Event | None = None,
    ) -> None:
        if listener_ttl_sec <= 0:
            raise ValueError("listener_ttl_sec must be positive")
        if max_listeners <= 0:
            raise ValueError("max_listeners must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if preparation_cache_size <= 0:
            raise ValueError("preparation_cache_size must be positive")
        if caption_history_size <= 0:
            raise ValueError("caption_history_size must be positive")
        if not math.isfinite(float(baseline_tts_speed)) or not (
            0.5 <= float(baseline_tts_speed) <= 2.0
        ):
            raise ValueError("baseline_tts_speed must be between 0.5 and 2.0")
        self._synthesizer = synthesizer
        self._root_dir = Path(root_dir)
        self._encoder_factory = encoder_factory or (lambda root: FFmpegHLSEncoder(root))
        self._listener_ttl_sec = float(listener_ttl_sec)
        self._max_listeners = int(max_listeners)
        self._sample_rate = max(8000, int(sample_rate))
        self._sentence_pause_ms = max(0, int(sentence_pause_ms))
        self._baseline_tts_speed = float(baseline_tts_speed)
        self._auto_speed_enabled = bool(auto_speed_enabled)
        self._clock = clock
        self._worker_start_gate = worker_start_gate
        self._leases: dict[str, HLSListenerLease] = {}
        self._queue: asyncio.Queue[TTSReadyItem] = asyncio.Queue(maxsize=int(queue_size))
        self._preparation_cache_size = int(preparation_cache_size)
        self._caption_cues: deque[HLSCaptionCue] = deque(
            maxlen=int(caption_history_size)
        )
        self._preparation_pending: dict[ItemKey, TTSReadyItem] = {}
        self._prepared_audio: dict[ItemKey, _PreparedAudio] = {}
        self._known_items: dict[ItemKey, TTSReadyItem] = {}
        self._audio_ms_per_char: dict[str, float] = {}
        self._latest_key_by_sentence: dict[str, ItemKey] = {}
        self._work_available = asyncio.Event()
        self._encoder: HLSEncoder | None = None
        self._active_root: Path | None = None
        self._worker: asyncio.Task[None] | None = None
        self._reaper: asyncio.Task[None] | None = None
        self._inflight_item: TTSReadyItem | None = None
        self._inflight_key: ItemKey | None = None
        self._inflight_kind = ""
        self._lock = asyncio.Lock()
        self._closed = False
        self._last_error = ""
        self._idle_backlog_dropped = 0
        self._speech_epoch_id = ""
        self._global_speed_multiplier = 1.0
        self._first_epoch_item_key: ItemKey | None = None

    @property
    def listener_count(self) -> int:
        return len(self._leases)

    @property
    def status(self) -> HLSStreamStatus:
        pending_audio_ms, backlog_ms, backlog_count, backlog_estimated = (
            self._backlog_snapshot()
        )
        multiplier = self._global_speed_multiplier if self._speech_epoch_id else 1.0
        return HLSStreamStatus(
            available=not self._closed and self._synthesizer is not None,
            listener_count=len(self._leases),
            queue_depth=self._queue.qsize() + int(self._inflight_kind == "release"),
            synthesis_active=self._inflight_item is not None,
            preparation_queue_depth=len(self._preparation_pending),
            preparation_active=self._inflight_kind == "prepare",
            prepared_audio_count=len(self._prepared_audio),
            pending_audio_ms=pending_audio_ms,
            translated_audio_backlog_ms=backlog_ms,
            translated_audio_backlog_count=backlog_count,
            translated_audio_backlog_estimated=backlog_estimated,
            speech_epoch_id=self._speech_epoch_id,
            global_speed_mode="auto" if self._auto_speed_enabled else "fixed",
            global_speed_multiplier=multiplier,
            tts_effective_speed=self._baseline_tts_speed * multiplier,
            encoder_active=self._encoder is not None,
            last_error=self._last_error,
        )

    def _backlog_snapshot(self) -> tuple[int, int, int, bool]:
        if not self._leases or self._encoder is None or not self._speech_epoch_id:
            return 0, 0, 0, False
        pending_audio_ms = max(
            0,
            int(getattr(self._encoder, "pending_audio_ms", 0) or 0),
        )
        future_audio_ms = 0
        backlog_estimated = False
        for key, item in self._known_items.items():
            prepared = self._prepared_audio.get(key)
            if prepared is not None:
                future_audio_ms += int(prepared.audio_ms)
            else:
                future_audio_ms += self._estimate_item_audio_ms(item)
                backlog_estimated = True
        return (
            pending_audio_ms,
            pending_audio_ms + future_audio_ms,
            len(self._known_items),
            backlog_estimated,
        )

    @staticmethod
    def _require_identity(value: str, name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{name} must be a non-empty string")
        return normalized

    @staticmethod
    def _item_key(item: TTSReadyItem) -> ItemKey:
        text_hash = hashlib.sha256(str(item.text).encode("utf-8")).hexdigest()
        return (
            str(item.sentence_id),
            int(item.revision),
            str(item.target_language),
            text_hash,
        )

    @staticmethod
    def _language_key(language: str) -> str:
        normalized = str(language or "").strip().lower()
        if "chinese" in normalized or "中文" in normalized:
            return "chinese"
        return "english"

    def _estimate_item_audio_ms(self, item: TTSReadyItem) -> int:
        language = self._language_key(item.target_language)
        default_ms_per_char = (
            DEFAULT_CHINESE_AUDIO_MS_PER_CHAR
            if language == "chinese"
            else DEFAULT_ENGLISH_AUDIO_MS_PER_CHAR
        )
        observed_ms_per_char = self._audio_ms_per_char.get(language, 0.0)
        ms_per_char = max(default_ms_per_char, observed_ms_per_char * 1.10)
        text_chars = max(1, len(str(item.text or "").strip()))
        return max(
            MIN_ESTIMATED_SENTENCE_AUDIO_MS,
            round(text_chars * ms_per_char) + self._sentence_pause_ms,
        )

    def _observe_item_audio_ms(
        self,
        item: TTSReadyItem,
        audio_ms: int,
        *,
        displayed_multiplier: float,
    ) -> None:
        text_chars = len(str(item.text or "").strip())
        if text_chars < 1:
            return
        language = self._language_key(item.target_language)
        speech_audio_ms = max(1, int(audio_ms) - self._sentence_pause_ms)
        baseline_speech_ms = speech_audio_ms * max(1.0, displayed_multiplier)
        observed = baseline_speech_ms / text_chars
        previous = self._audio_ms_per_char.get(language)
        self._audio_ms_per_char[language] = (
            observed
            if previous is None
            else previous * (1.0 - DURATION_ESTIMATE_ALPHA)
            + observed * DURATION_ESTIMATE_ALPHA
        )

    def _select_synthesis_speed_locked(
        self,
        key: ItemKey,
        kind: str,
    ) -> tuple[float, float, int]:
        _, backlog_ms, _, _ = self._backlog_snapshot()
        force_join_baseline = (
            kind == "release" and key == self._first_epoch_item_key
        )
        multiplier = 1.0
        if self._auto_speed_enabled and not force_join_baseline:
            multiplier = select_global_tts_multiplier(backlog_ms)
        effective_speed = self._baseline_tts_speed * multiplier
        if not math.isfinite(effective_speed) or not 0.5 <= effective_speed <= 2.0:
            logger.warning(
                "shared HLS TTS speed fallback epoch=%s backlog_ms=%d multiplier=%.1f effective_speed=%.3f",
                self._speech_epoch_id,
                backlog_ms,
                multiplier,
                effective_speed,
            )
            self._last_error = "TTSSpeedRangeError"
            multiplier = 1.0
            effective_speed = self._baseline_tts_speed
        self._global_speed_multiplier = multiplier
        if force_join_baseline:
            self._first_epoch_item_key = None
        return multiplier, effective_speed, backlog_ms

    def _select_latest_item_locked(
        self,
        item: TTSReadyItem,
        key: ItemKey,
    ) -> None:
        sentence_id = str(item.sentence_id)
        self._latest_key_by_sentence[sentence_id] = key
        for prepared_key in list(self._preparation_pending):
            if prepared_key[0] == sentence_id and prepared_key != key:
                del self._preparation_pending[prepared_key]
        for prepared_key in list(self._prepared_audio):
            if prepared_key[0] == sentence_id and prepared_key != key:
                del self._prepared_audio[prepared_key]
        for known_key in list(self._known_items):
            if known_key[0] == sentence_id and known_key != key:
                del self._known_items[known_key]

    def _retain_latest_idle_item_locked(self) -> tuple[int, ItemKey | None]:
        """Collapse pre-listener speech to the current live sentence."""

        queued: list[TTSReadyItem] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            queued.append(item)
        if not queued:
            return 0, None

        latest = queued[-1]
        self._queue.put_nowait(latest)
        stale = queued[:-1]
        for item in stale:
            key = self._item_key(item)
            sentence_id = str(item.sentence_id)
            if self._latest_key_by_sentence.get(sentence_id) == key:
                self._latest_key_by_sentence.pop(sentence_id, None)
            self._preparation_pending.pop(key, None)
            self._prepared_audio.pop(key, None)
            self._known_items.pop(key, None)

        dropped = len(stale)
        if dropped:
            self._idle_backlog_dropped += dropped
            logger.info(
                "shared HLS live join skipped stale backlog skipped=%d retained=%d total_skipped=%d",
                dropped,
                self._queue.qsize(),
                self._idle_backlog_dropped,
            )
        return dropped, self._item_key(latest)

    async def touch_listener(self, listener_id: str, owner_key: str) -> HLSListenerLease:
        listener = self._require_identity(listener_id, "listener_id")
        owner = self._require_identity(owner_key, "owner_key")
        await self.prune_expired()
        async with self._lock:
            if self._closed or self._synthesizer is None:
                raise HLSUnavailable("shared HLS TTS is unavailable")
            existing = self._leases.get(listener)
            if existing is not None and existing.owner_key != owner:
                raise HLSListenerNotFound("listener lease is unavailable")
            if existing is None and len(self._leases) >= self._max_listeners:
                raise HLSListenerCapacityExceeded("listener capacity reached")
            if self._encoder is None:
                _, retained_key = self._retain_latest_idle_item_locked()
                root = self._root_dir / f"epoch-{uuid.uuid4().hex}"
                encoder = self._encoder_factory(root)
                try:
                    await encoder.start()
                except Exception:
                    shutil.rmtree(root, ignore_errors=True)
                    raise
                self._active_root = root
                self._speech_epoch_id = root.name
                self._global_speed_multiplier = 1.0
                self._first_epoch_item_key = retained_key
                self._encoder = encoder
                self._worker = asyncio.create_task(self._worker_loop())
                self._reaper = asyncio.create_task(self._reaper_loop())
                if self._queue.qsize() or self._preparation_pending:
                    self._work_available.set()
            lease = HLSListenerLease(
                listener_id=listener,
                owner_key=owner,
                expires_at=self._clock() + self._listener_ttl_sec,
            )
            self._leases[listener] = lease
            return lease

    def _require_lease(self, listener_id: str, owner_key: str) -> HLSListenerLease:
        listener = self._require_identity(listener_id, "listener_id")
        owner = self._require_identity(owner_key, "owner_key")
        lease = self._leases.get(listener)
        if (
            lease is None
            or lease.owner_key != owner
            or lease.expires_at <= self._clock()
        ):
            raise HLSListenerNotFound("listener lease is unavailable")
        return lease

    async def publish(self, item: TTSReadyItem) -> bool:
        await self.prune_expired()
        async with self._lock:
            if self._closed or self._synthesizer is None:
                return False
            key = self._item_key(item)
            self._select_latest_item_locked(item, key)
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull as exc:
                if self._leases and self._encoder is not None:
                    raise HLSQueueFull("shared HLS TTS queue is full") from exc
                dropped_item = self._queue.get_nowait()
                self._queue.task_done()
                self._known_items.pop(self._item_key(dropped_item), None)
                self._queue.put_nowait(item)
                self._idle_backlog_dropped += 1
                if self._idle_backlog_dropped == 1 or self._idle_backlog_dropped % 32 == 0:
                    logger.info(
                        "shared HLS idle backlog dropped oldest total=%d retained=%d",
                        self._idle_backlog_dropped,
                        self._queue.qsize(),
                    )
            self._known_items[key] = item
            self._work_available.set()
            return True

    async def prepare(self, item: TTSReadyItem) -> bool:
        """Prepare an exact translation revision without publishing its audio."""

        await self.prune_expired()
        async with self._lock:
            if (
                self._closed
                or self._synthesizer is None
                or not self._leases
                or self._encoder is None
            ):
                return False
            key = self._item_key(item)
            self._select_latest_item_locked(item, key)
            self._known_items[key] = item
            if (
                key in self._prepared_audio
                or key in self._preparation_pending
                or key == self._inflight_key
            ):
                return True
            if len(self._preparation_pending) >= self._preparation_cache_size:
                return False
            self._preparation_pending[key] = item
            self._work_available.set()
            return True

    async def wait_idle(self) -> None:
        await self._queue.join()

    async def discard_idle_backlog(self) -> int:
        """Drop retained speech only when no listener epoch is active."""

        async with self._lock:
            if self._leases or self._encoder is not None or self._inflight_item is not None:
                return 0
            dropped = 0
            while True:
                try:
                    dropped_item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    self._queue.task_done()
                    self._known_items.pop(self._item_key(dropped_item), None)
                    dropped += 1
            return dropped

    async def wait_ready(self, listener_id: str, owner_key: str, timeout: float = 5.0) -> None:
        self._require_lease(listener_id, owner_key)
        encoder = self._encoder
        if encoder is None:
            raise HLSUnavailable("shared HLS encoder is unavailable")
        await encoder.wait_ready(timeout)

    def playlist_text(self, listener_id: str, owner_key: str) -> str:
        self._require_lease(listener_id, owner_key)
        encoder = self._encoder
        if encoder is None:
            raise HLSUnavailable("shared HLS encoder is unavailable")
        return encoder.playlist_text()

    def segment_path(self, listener_id: str, owner_key: str, name: str) -> Path:
        self._require_lease(listener_id, owner_key)
        encoder = self._encoder
        if encoder is None:
            raise HLSUnavailable("shared HLS encoder is unavailable")
        return encoder.segment_path(name)

    def caption_snapshot(
        self,
        listener_id: str,
        owner_key: str,
    ) -> HLSCaptionSnapshot:
        self._require_lease(listener_id, owner_key)
        encoder = self._encoder
        live_edge_at_ms: int | None = None
        if encoder is not None:
            get_live_edge = getattr(encoder, "live_edge_at_ms", None)
            if callable(get_live_edge):
                live_edge_at_ms = get_live_edge()
        return HLSCaptionSnapshot(
            live_edge_at_ms=live_edge_at_ms,
            cues=tuple(self._caption_cues),
        )

    async def remove_listener(self, listener_id: str, owner_key: str) -> bool:
        listener = self._require_identity(listener_id, "listener_id")
        owner = self._require_identity(owner_key, "owner_key")
        should_stop = False
        async with self._lock:
            existing = self._leases.get(listener)
            if existing is None or existing.owner_key != owner:
                return False
            del self._leases[listener]
            should_stop = not self._leases
        if should_stop:
            await self._stop_stream()
        return True

    async def prune_expired(self) -> int:
        now = self._clock()
        removed = 0
        should_stop = False
        async with self._lock:
            for listener_id, lease in list(self._leases.items()):
                if lease.expires_at <= now:
                    del self._leases[listener_id]
                    removed += 1
            should_stop = removed > 0 and not self._leases
        if should_stop:
            await self._stop_stream()
        return removed

    async def _worker_loop(self) -> None:
        if self._worker_start_gate is not None:
            await self._worker_start_gate.wait()
        while True:
            await self._work_available.wait()
            item: TTSReadyItem | None = None
            key: ItemKey | None = None
            prepared: _PreparedAudio | None = None
            kind = ""
            displayed_multiplier = 1.0
            effective_speed = self._baseline_tts_speed
            decision_backlog_ms = 0
            speed_source = "decision"
            async with self._lock:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    item = None
                if item is not None:
                    kind = "release"
                    key = self._item_key(item)
                    self._select_latest_item_locked(item, key)
                    self._preparation_pending.pop(key, None)
                    prepared = self._prepared_audio.pop(key, None)
                elif self._preparation_pending:
                    key, item = min(
                        self._preparation_pending.items(),
                        key=lambda pair: int(pair[1].source_order),
                    )
                    del self._preparation_pending[key]
                    kind = "prepare"
                else:
                    self._work_available.clear()
                    continue
                self._inflight_item = item
                self._inflight_key = key
                self._inflight_kind = kind
                if prepared is not None:
                    _, decision_backlog_ms, _, _ = self._backlog_snapshot()
                    displayed_multiplier = prepared.displayed_multiplier
                    effective_speed = prepared.effective_speed
                    self._global_speed_multiplier = displayed_multiplier
                    speed_source = "prepared"
                    if key == self._first_epoch_item_key:
                        self._first_epoch_item_key = None
                else:
                    displayed_multiplier, effective_speed, decision_backlog_ms = (
                        self._select_synthesis_speed_locked(key, kind)
                    )
            try:
                encoder = self._encoder
                if encoder is None or item is None or key is None:
                    continue
                cache_hit = prepared is not None
                if prepared is None:
                    synthesis_started = time.monotonic()
                    logger.info(
                        "shared HLS TTS %s started epoch=%s source_order=%d revision=%d text_chars=%d stable_pending=%d prepare_pending=%d backlog_ms=%d multiplier=%.1f effective_speed=%.3f speed_source=%s",
                        "preparation" if kind == "prepare" else "synthesis",
                        self._speech_epoch_id,
                        int(item.source_order),
                        int(item.revision),
                        len(str(item.text)),
                        self.status.queue_depth,
                        self.status.preparation_queue_depth,
                        decision_backlog_ms,
                        displayed_multiplier,
                        effective_speed,
                        speed_source,
                    )
                    audio = await asyncio.to_thread(
                        self._synthesizer.synthesize,
                        item.text,
                        item.target_language,
                        speed=effective_speed,
                    )
                    pcm = decode_mono_pcm16_wav(
                        audio.wav_bytes,
                        expected_rate=self._sample_rate,
                    )
                    cue_start_offset_ms, cue_end_offset_ms = _pcm_activity_bounds_ms(
                        pcm,
                        sample_rate=self._sample_rate,
                    )
                    pause_samples = round(
                        self._sample_rate * self._sentence_pause_ms / 1000
                    )
                    if pause_samples > 0:
                        pcm += bytes(pause_samples * 2)
                    synthesis_ms = round((time.monotonic() - synthesis_started) * 1000)
                    audio_ms = max(
                        1,
                        round(len(pcm) * 1000 / (self._sample_rate * 2)),
                    )
                    prepared = _PreparedAudio(
                        pcm=pcm,
                        audio_ms=audio_ms,
                        cue_start_offset_ms=cue_start_offset_ms,
                        cue_end_offset_ms=cue_end_offset_ms,
                        synthesis_ms=synthesis_ms,
                        prepared_at=self._clock(),
                        displayed_multiplier=displayed_multiplier,
                        effective_speed=effective_speed,
                    )
                    self._observe_item_audio_ms(
                        item,
                        audio_ms,
                        displayed_multiplier=displayed_multiplier,
                    )

                if kind == "prepare":
                    cached = False
                    async with self._lock:
                        if (
                            self._encoder is encoder
                            and self._latest_key_by_sentence.get(str(item.sentence_id)) == key
                        ):
                            while len(self._prepared_audio) >= self._preparation_cache_size:
                                oldest_key = next(iter(self._prepared_audio))
                                del self._prepared_audio[oldest_key]
                            self._prepared_audio[key] = prepared
                            cached = True
                    self._last_error = ""
                    logger.info(
                        "shared HLS TTS preparation completed epoch=%s source_order=%d revision=%d synthesis_ms=%d audio_ms=%d rtf=%.3f cached=%s prepared=%d backlog_ms=%d multiplier=%.1f effective_speed=%.3f",
                        self._speech_epoch_id,
                        int(item.source_order),
                        int(item.revision),
                        int(prepared.synthesis_ms),
                        int(prepared.audio_ms),
                        prepared.synthesis_ms / prepared.audio_ms,
                        str(cached).lower(),
                        len(self._prepared_audio),
                        decision_backlog_ms,
                        displayed_multiplier,
                        effective_speed,
                    )
                    continue

                receipt = await encoder.append_pcm(prepared.pcm)
                published_discardable_gap_ms = 0
                async with self._lock:
                    self._known_items.pop(key, None)
                if isinstance(receipt, HLSAppendReceipt):
                    cue_start_at_ms = (
                        int(receipt.start_at_ms)
                        + int(prepared.cue_start_offset_ms)
                    )
                    cue_end_at_ms = min(
                        int(receipt.end_at_ms),
                        int(receipt.start_at_ms)
                        + int(prepared.cue_end_offset_ms),
                    )
                    if cue_end_at_ms > cue_start_at_ms:
                        previous = (
                            self._caption_cues[-1]
                            if self._caption_cues
                            else None
                        )
                        actual_gap_ms = (
                            0
                            if previous is None
                            else max(
                                0,
                                cue_start_at_ms - previous.end_at_ms,
                            )
                        )
                        discardable_gap_ms = min(
                            actual_gap_ms,
                            max(
                                0,
                                int(receipt.discardable_gap_before_ms),
                            ),
                        )
                        published_discardable_gap_ms = discardable_gap_ms
                        natural_gap_ms = max(
                            0,
                            actual_gap_ms - discardable_gap_ms,
                        )
                        resume_at_ms = (
                            cue_start_at_ms - natural_gap_ms
                            if previous is not None
                            and discardable_gap_ms > 0
                            else None
                        )
                        epoch = self._active_root.name if self._active_root else ""
                        cue_key = (
                            f"{epoch}:{item.sentence_id}:{int(item.revision)}:"
                            f"{cue_start_at_ms}"
                        )
                        self._caption_cues.append(
                            HLSCaptionCue(
                                cue_id=hashlib.sha256(
                                    cue_key.encode("utf-8")
                                ).hexdigest()[:16],
                                start_at_ms=cue_start_at_ms,
                                end_at_ms=cue_end_at_ms,
                                text=str(item.text),
                                discardable_gap_before_ms=discardable_gap_ms,
                                resume_at_ms=resume_at_ms,
                            )
                        )
                self._last_error = ""
                preparation_age_ms = max(
                    0,
                    round((self._clock() - prepared.prepared_at) * 1000),
                )
                logger.info(
                    "shared HLS TTS audio published epoch=%s source_order=%d revision=%d cache_hit=%s synthesis_ms=%d preparation_age_ms=%d audio_ms=%d pending=%d pending_audio_ms=%d backlog_ms=%d multiplier=%.1f effective_speed=%.3f speed_source=%s discardable_gap_before_ms=%d",
                    self._speech_epoch_id,
                    int(item.source_order),
                    int(item.revision),
                    str(cache_hit).lower(),
                    int(prepared.synthesis_ms),
                    preparation_age_ms,
                    int(prepared.audio_ms),
                    self._queue.qsize(),
                    int(getattr(encoder, "pending_audio_ms", 0) or 0),
                    decision_backlog_ms,
                    displayed_multiplier,
                    effective_speed,
                    speed_source,
                    published_discardable_gap_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = type(exc).__name__
                if kind == "release" and key is not None:
                    async with self._lock:
                        self._known_items.pop(key, None)
                logger.warning(
                    "shared HLS TTS item failed source_order=%d error=%s",
                    int(item.source_order),
                    type(exc).__name__,
                )
            finally:
                async with self._lock:
                    self._inflight_item = None
                    self._inflight_key = None
                    self._inflight_kind = ""
                    if self._queue.qsize() or self._preparation_pending:
                        self._work_available.set()
                    else:
                        self._work_available.clear()
                if kind == "release":
                    self._queue.task_done()

    async def _reaper_loop(self) -> None:
        interval = max(1.0, min(self._listener_ttl_sec / 3.0, 15.0))
        try:
            while True:
                await asyncio.sleep(interval)
                await self.prune_expired()
                if not self._leases:
                    return
        except asyncio.CancelledError:
            raise

    async def _stop_stream(self) -> None:
        async with self._lock:
            if self._leases:
                return
            worker = self._worker
            reaper = self._reaper
            encoder = self._encoder
            root = self._active_root
            self._worker = None
            self._reaper = None
            self._encoder = None
            self._active_root = None
            self._speech_epoch_id = ""
            self._global_speed_multiplier = 1.0
            self._first_epoch_item_key = None
            while True:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._queue.task_done()
            self._preparation_pending.clear()
            self._prepared_audio.clear()
            self._known_items.clear()
            self._latest_key_by_sentence.clear()
            self._caption_cues.clear()
            self._work_available.clear()
        current = asyncio.current_task()
        for task in (worker, reaper):
            if task is not None and task is not current:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if encoder is not None:
            await encoder.close()
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._leases.clear()
        await self._stop_stream()
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()


__all__ = [
    "FFmpegHLSEncoder",
    "HLSAppendReceipt",
    "HLSCaptionCue",
    "HLSCaptionSnapshot",
    "HLSError",
    "HLSListenerCapacityExceeded",
    "HLSListenerLease",
    "HLSListenerNotFound",
    "HLSQueueFull",
    "HLSStreamStatus",
    "HLSUnavailable",
    "SharedHLSTTSPublisher",
    "decode_mono_pcm16_wav",
    "parse_hls_live_edge_at_ms",
    "parse_hls_timeline_origin_at_ms",
    "select_global_tts_multiplier",
]
