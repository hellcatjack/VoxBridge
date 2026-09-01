from types import SimpleNamespace
import asyncio
import hashlib
import io
import inspect
import json
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import voxbridge.cli.demo_streaming_ws as demo_streaming_ws
from voxbridge.cli.demo_streaming_ws import _create_app, _hash_auth_password
from voxbridge.tts.hls import HLSAppendReceipt
from voxbridge.tts.jobs import RevisionStableTTSBuffer, TTSReadyItem
from voxbridge.tts.kokoro_onnx import SynthesizedAudio
from voxbridge.streaming.vad_support import SileroShadowObserver


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(str(text or ""))


class _FakeASR:
    def __init__(self):
        self.init_calls = []
        self.finish_calls = 0
        self.transcribe_calls = []
        self.processor = SimpleNamespace(tokenizer=_FakeTokenizer())

    def init_streaming_state(self, **kwargs):
        self.init_calls.append(kwargs)
        return SimpleNamespace(
            language="",
            text="",
            kwargs=kwargs,
            force_language=kwargs.get("language"),
            audio_accum=np.zeros((0,), dtype=np.float32),
        )

    def streaming_transcribe(self, wav, state):
        assert isinstance(wav, np.ndarray)
        state.audio_accum = np.concatenate([state.audio_accum, np.asarray(wav, dtype=np.float32)])
        state.language = "Chinese"
        state.text = state.text + "partial"
        return state

    def finish_streaming_transcribe(self, state):
        self.finish_calls += 1
        state.language = state.language or "Chinese"
        state.text = state.text + "|final"
        return state

    def transcribe(self, audio, context="", language=None):
        self.transcribe_calls.append(
            {
                "audio": audio,
                "context": context,
                "language": language,
                "sampling_max_tokens": getattr(
                    getattr(self, "sampling_params", None),
                    "max_tokens",
                    None,
                ),
            }
        )

        return [
            SimpleNamespace(
                language=str(getattr(self, "transcribe_language", "Chinese") or "Chinese"),
                text=str(getattr(self, "transcribe_text", "pseudo") or "pseudo"),
            )
        ]


class _FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text: str, source_language: str = None, target_language: str = None):
        src = str(source_language or "")
        tgt = str(target_language or "")
        self.calls.append((str(text or ""), src, tgt))
        return f"[{src}->{tgt}] {text}"


class _FakeTTSSynthesizer:
    def __init__(self):
        self.calls = []
        self.speed_calls = []

    def synthesize(
        self,
        text: str,
        target_language: str,
        *,
        speed: float | None = None,
    ):
        self.calls.append((text, target_language))
        self.speed_calls.append((text, target_language, speed))
        return SynthesizedAudio(b"RIFF-fake-wav", sample_rate=24000, duration_ms=750)


class _FakeHLSEncoder:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.started = 0
        self.closed = 0
        self.appended = []
        self.pending_audio_ms = 1750
        self.next_start_at_ms = 100_000
        self.next_discardable_gap_before_ms = 0

    async def start(self):
        self.started += 1
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "segment_000000001.ts").write_bytes(b"shared-aac-segment")
        (self.root / "index.m3u8").write_text(
            "#EXTM3U\n#EXTINF:1.0,\nsegment_000000001.ts\n",
            encoding="utf-8",
        )

    async def append_pcm(self, pcm):
        data = bytes(pcm)
        self.appended.append(data)
        duration_ms = round(len(data) * 1000 / (24000 * 2))
        discardable_gap_before_ms = self.next_discardable_gap_before_ms
        self.next_discardable_gap_before_ms = 0
        self.next_start_at_ms += discardable_gap_before_ms
        receipt = HLSAppendReceipt(
            start_at_ms=self.next_start_at_ms,
            end_at_ms=self.next_start_at_ms + duration_ms,
            discardable_gap_before_ms=discardable_gap_before_ms,
        )
        self.next_start_at_ms = receipt.end_at_ms
        return receipt

    async def wait_ready(self, timeout=5.0):
        del timeout

    def playlist_text(self):
        return (self.root / "index.m3u8").read_text(encoding="utf-8")

    def live_edge_at_ms(self):
        return self.next_start_at_ms

    def segment_path(self, name):
        return self.root / name

    async def close(self):
        self.closed += 1


def _silent_wav_bytes(duration_ms: int = 250) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(bytes(round(24000 * duration_ms / 1000) * 2))
    return output.getvalue()


def _args():
    return SimpleNamespace(
        backend="vllm",
        force_language=None,
        translation_source_language="Chinese",
        translation_target_language="English",
        max_new_tokens=32,
        audio_queue_size=32,
        client_chunk_ms=320,
        max_connections=4,
        unfixed_chunk_num=4,
        unfixed_token_num=5,
        chunk_size_sec=1.0,
        min_audio_sec=1.0,
        decode_interval_sec=1.0,
        idle_timeout_sec=30,
        max_frame_samples=32000,
        tts_revision_stable_sec=0.0,
        tts_latest_revision_grace_sec=0.0,
    )


def _receive_until_type(ws, expected_type: str, max_steps: int = 40):
    seen = []
    for _ in range(max_steps):
        msg = ws.receive_json()
        if msg.get("type") == expected_type:
            return msg
        seen.append(msg.get("type"))
    pytest.fail(f"did not receive {expected_type}, seen={seen}")


def _login_tts_owner(client: TestClient) -> str:
    login = client.post(
        "/login",
        data={"username": "admin", "password": "secret"},
        follow_redirects=False,
    )
    assert login.status_code in {302, 303, 307}
    token = client.cookies.get("voxbridge_session")
    assert token
    return hashlib.sha256(f"auth:{token}".encode()).hexdigest()


def test_ws_ready_partial_final_flow():
    app = _create_app(_args(), _FakeASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000
        assert ready["translation_direction"] == "zh2en"

        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        partial = ws.receive_json()
        assert partial["type"] == "partial"
        assert partial["language"] == "Chinese"
        assert "partial" in partial["text"]

        ws.send_text('{"type":"finish"}')
        final = _receive_until_type(ws, "final")
        assert final["type"] == "final"
        assert final["language"] == "Chinese"
        assert final["text"]


def test_ws_client_silence_span_never_enters_asr(monkeypatch, tmp_path):
    class CountingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_calls = 0

        def streaming_transcribe(self, wav, state):
            self.streaming_calls += 1
            return super().streaming_transcribe(wav, state)

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    trace_path = tmp_path / "client-silence.jsonl"
    args = _args()
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = CountingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": ["尼希米记", "耶路撒冷"],
            }
        )
        _receive_until_type(ws, "started")
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 1000,
                "capture_sample_index": 16000,
            }
        )
        ws.send_json({"type": "ping"})
        _receive_until_type(ws, "pong")
        time.sleep(0.1)

    assert fake_asr.streaming_calls == 0
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    silence_rows = [row for row in rows if row.get("event") == "client_silence_applied"]
    assert len(silence_rows) == 1
    assert silence_rows[0]["duration_ms"] == 1000
    assert silence_rows[0]["context_guard_active"] is True


def test_ws_client_silence_span_drives_backend_vad_cut(monkeypatch):
    class CountingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_calls = 0

        def streaming_transcribe(self, wav, state):
            self.streaming_calls += 1
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "Chinese"
            state.text = "这是一句已经完成的测试。"
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    args = _args()
    args.final_redecode_on_stop = False
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    fake_asr = CountingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        speech = np.full(6400, 20_000, dtype="<i2").tobytes()
        ws.send_bytes(speech)
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.01)

    assert fake_asr.streaming_calls == 1
    assert fake_asr.finish_calls == 1
    assert len(fake_asr.init_calls) == 2


def test_ws_vad_cut_preserves_deferred_prefix_when_next_segment_has_no_overlap(
    monkeypatch,
    tmp_path,
):
    deferred = (
        "但又听到别人不经意的一句批评，"
        "心里面呢就会开始，啊，啊，有这个怨气，慢慢的开始。"
    )
    following = "后来大家才逐渐明白这件事情的原因。"

    class DeferredAcrossVadCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            return state

        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = deferred if state.segment_no == 1 else following
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    trace_path = tmp_path / "vad-deferred-prefix.jsonl"
    args = _args()
    args.final_redecode_on_stop = False
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 1.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.unfixed_token_num = 8
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    translator = _FakeTranslator()
    fake_asr = DeferredAcrossVadCutASR()
    events = []

    with TestClient(_create_app(args, fake_asr, translator=translator)).websocket_connect("/ws") as ws:
        ws.receive_json()
        speech = np.full(6400, 20_000, dtype="<i2").tobytes()
        ws.send_bytes(speech)
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        deadline = time.time() + 2.0
        while len(fake_asr.init_calls) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(fake_asr.init_calls) == 2

        for _ in range(2):
            ws.send_bytes(speech)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

    committed = [
        str(message.get("text", "")).strip()
        for message in events
        if message.get("type") == "sentence_committed"
    ]
    translated_sources = [call[0] for call in translator.calls]
    assert deferred in "".join(committed)
    assert deferred in "".join(translated_sources)
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "pending_prefix_vad_boundary_preserved" for row in trace_rows)
    assert not any(row.get("event") == "pending_prefix_drop_no_overlap" for row in trace_rows)


def test_ws_segment_final_redecode_repairs_tail_before_translation_and_tts(monkeypatch):
    class TailRepairASR(_FakeASR):
        streaming_text = "这是实时识别测试，句末不能。"
        canonical_text = "这是实时识别测试，句末不能缺失。"

        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = self.streaming_text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = self.streaming_text
            return state

        def transcribe(self, audio, context="", language=None):
            super().transcribe(audio, context=context, language=language)
            return [SimpleNamespace(language="Chinese", text=self.canonical_text)]

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    args = _args()
    args.final_redecode_on_stop = False
    args.segment_final_redecode = True
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.tts_revision_stable_sec = 0.0
    args.tts_latest_revision_grace_sec = 0.0
    fake_asr = TailRepairASR()
    translator = _FakeTranslator()
    app = _create_app(
        args,
        fake_asr,
        translator=translator,
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "language": "Chinese"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.full(6400, 20_000, dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        committed = _poll_ws_with_ping(
            ws,
            events,
            lambda message: message.get("type") == "sentence_committed",
        )
        translated = _poll_ws_with_ping(
            ws,
            events,
            lambda message: message.get("type") == "sentence_translation",
        )

    jobs = [
        event
        for event in _drain_listener_events(listener)
        if event.get("type") == "tts_job"
    ]
    assert fake_asr.finish_calls == 1
    assert len(fake_asr.transcribe_calls) == 1
    assert committed["text"] == fake_asr.canonical_text
    assert fake_asr.canonical_text in translated["translation"]
    assert translator.calls == [(fake_asr.canonical_text, "Chinese", "English")]
    assert len(jobs) == 1
    assert jobs[0]["sentence_id"] == committed["sentence_id"]
    assert jobs[0]["revision"] == translated["revision"]


def test_ws_segment_redecode_commits_new_terminal_sentence_after_resegmentation(monkeypatch, tmp_path):
    first = "The first sufficiently long sentence is already complete."
    second = "The second sufficiently long sentence is already complete."
    streaming_third = "The third sufficiently long sentence was wrote before the cut."
    corrected_third = "The third sufficiently long sentence was written before the cut."
    terminal = "The terminal sentence remains visible."

    class ResegmentedTailASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "English"
            state.text = f"{first} {second} {streaming_third} {terminal}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

        def transcribe(self, audio, context="", language=None):
            super().transcribe(audio, context=context, language=language)
            return [
                SimpleNamespace(
                    language="English",
                    text=f"{first} {second} {corrected_third} {terminal}",
                )
            ]

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    args = _args()
    args.force_language = "English"
    args.final_redecode_on_stop = False
    args.segment_final_redecode = True
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.early_translation_stable_hits = 99
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "resegmented-terminal.jsonl")
    fake_asr = ResegmentedTailASR()
    events = []

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "language": "English"})
        _receive_until_type(ws, "started")
        frame = np.full(6400, 20_000, dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(frame)
            _poll_ws_with_ping(ws, events, lambda message: message.get("type") == "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        time.sleep(0.2)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 200,
                "capture_sample_index": 24000,
            }
        )
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_committed"
                and message.get("text") == terminal
            ),
        )

    committed = [
        str(message.get("text") or "")
        for message in events
        if message.get("type") == "sentence_committed"
    ]
    assert terminal in committed
    assert committed.count(terminal) == 1


@pytest.mark.parametrize(
    "redecode_case",
    (
        "completed_unit_regression",
        "unsafe_divergence",
        "context_echo",
        "effective_completed_unit_regression",
    ),
)
def test_ws_segment_redecode_rejects_unsafe_candidate_and_releases_fallback(
    monkeypatch,
    tmp_path,
    redecode_case,
):
    first = "The first sufficiently long sentence is already complete."
    second = "The second sufficiently long sentence is already complete."
    third = "The third sufficiently long sentence is already complete."
    fourth = "The fourth sufficiently long sentence is already complete."
    terminal = "The terminal sentence must remain visible."
    streaming_text = f"{first} {second} {third} {fourth} {terminal}"
    merged_correction = (
        "The first sufficiently long sentence is already complete, and the second "
        f"sufficiently long sentence is already complete. {third} {fourth} {terminal}"
    )
    divergent_correction = (
        "Unrelated travelers carefully cataloged every lighthouse beside the northern coast. "
        "Several musicians rehearsed a difficult symphony inside the restored theater. "
        "Gardeners planted winter vegetables throughout the newly prepared community plots. "
        "Engineers inspected each bridge before the regional transportation meeting began. "
        "Astronomers photographed distant galaxies during an exceptionally clear evening."
    )
    context_echo_terms = [
        "UnrelatedTravelersCatalogedLighthouses",
        "MusiciansRehearsedSymphonies",
        "GardenersPlantedVegetables",
        "EngineersInspectedBridges",
        "AstronomersPhotographedGalaxies",
    ]
    context_echo_correction = ". ".join(context_echo_terms) + "."
    accepted_correction = streaming_text.replace(
        terminal,
        "The terminal sentence will remain clearly visible.",
    )
    corrected_text = (
        merged_correction
        if redecode_case == "completed_unit_regression"
        else (
            context_echo_correction
            if redecode_case == "context_echo"
            else (
                accepted_correction
                if redecode_case == "effective_completed_unit_regression"
                else divergent_correction
            )
        )
    )
    context_terms = (
        context_echo_terms
        if redecode_case == "context_echo"
        else []
    )

    class RegressiveResegmentationASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "English"
            state.text = streaming_text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

        def transcribe(self, audio, context="", language=None):
            super().transcribe(audio, context=context, language=language)
            return [SimpleNamespace(language="English", text=corrected_text)]

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    if redecode_case == "effective_completed_unit_regression":
        monkeypatch.setattr(
            demo_streaming_ws,
            "_effective_completed_unit_counts",
            lambda *args, **kwargs: (2, 1),
        )
    trace_path = tmp_path / f"rejected-{redecode_case}.jsonl"
    args = _args()
    args.force_language = "English"
    args.final_redecode_on_stop = False
    args.segment_final_redecode = True
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.early_translation_stable_hits = 99
    args.asr_context_max_chars = 1000
    args.tts_revision_stable_sec = 0.0
    args.tts_latest_revision_grace_sec = 0.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = RegressiveResegmentationASR()
    app = _create_app(
        args,
        fake_asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": context_terms,
            }
        )
        _receive_until_type(ws, "started")
        frame = np.full(6400, 20_000, dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(frame)
            _poll_ws_with_ping(ws, events, lambda message: message.get("type") == "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        time.sleep(0.2)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 200,
                "capture_sample_index": 24000,
            }
        )
        terminal_commit = _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_committed"
                and message.get("text") == terminal
            ),
        )
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_translation"
                and message.get("sentence_id") == terminal_commit["sentence_id"]
            ),
        )

        broadcast_events = []
        for _ in range(40):
            broadcast_events.extend(_drain_listener_events(listener))
            if any(
                event.get("type") == "tts_job"
                and event.get("sentence_id") == terminal_commit["sentence_id"]
                for event in broadcast_events
            ):
                break
            time.sleep(0.01)

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(fake_asr.transcribe_calls) == 1
    assert any(
        row.get("event") == "segment_final_redecode_skipped"
        and row.get("reason") == redecode_case
        for row in rows
    )
    assert any(row.get("event") == "tts_source_sealed" for row in rows)
    assert not any(row.get("event") == "tts_source_seal_deferred" for row in rows)
    assert any(
        event.get("type") == "tts_job"
        and event.get("sentence_id") == terminal_commit["sentence_id"]
        for event in broadcast_events
    )
    committed = [
        str(message.get("text") or "")
        for message in events
        if message.get("type") == "sentence_committed"
    ]
    assert terminal in committed
    assert committed.count(terminal) == 1


def test_ws_mid_speech_hard_cut_skips_blocking_segment_redecode(monkeypatch, tmp_path):
    class HardCutASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = "A live sentence is still growing without a pause"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    trace_path = tmp_path / "hard-cut-fast-rotation.jsonl"
    args = _args()
    args.segment_final_redecode = True
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.vad_silence_sec = 30.0
    args.vad_force_cut_sec = 30.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = HardCutASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        frame = np.full(6400, 20_000, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.02)
        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert fake_asr.finish_calls >= 2
    assert fake_asr.transcribe_calls == []
    assert any(
        row.get("event") == "segment_final_redecode_skipped"
        and row.get("reason") == "live_hard_cut"
        for row in rows
    )


def test_ws_hard_cut_before_vad_endpoint_keeps_mid_speech_tail_without_redecode(
    monkeypatch, tmp_path
):
    class HardCutASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "English"
            state.text = "A live news sentence is still growing across the hard limit"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    trace_path = tmp_path / "hard-cut-low-energy-tail.jsonl"
    args = _args()
    args.segment_final_redecode = True
    args.segment_hard_cut_sec = 1.3
    args.segment_overlap_sec = 0.0
    args.vad_silence_sec = 30.0
    args.vad_force_cut_sec = 30.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = HardCutASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        speech = np.full(6400, 20_000, dtype="<i2").tobytes()
        low_energy_tail = np.zeros(6400, dtype="<i2").tobytes()
        ws.send_bytes(speech)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(low_energy_tail)
        _receive_until_type(ws, "partial")
        time.sleep(0.3)
        ws.send_bytes(low_energy_tail)
        _receive_until_type(ws, "partial")
        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.02)
        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert fake_asr.finish_calls >= 2
    assert fake_asr.transcribe_calls == []
    assert any(
        row.get("event") == "segment_cut_decision"
        and row.get("reason") == "hard_cut"
        and int(row.get("silence_ms", 0)) >= 80
        for row in rows
    )
    assert any(
        row.get("event") == "segment_finalize_done"
        and row.get("reason") == "hard_cut"
        and row.get("hard_cut_mid_speech") is True
        for row in rows
    )
    assert any(
        row.get("event") == "segment_final_redecode_skipped"
        and row.get("reason") == "live_hard_cut"
        for row in rows
    )


def test_ws_default_stop_keeps_visible_history_without_full_redecode():
    fake_asr = _FakeASR()
    app = _create_app(_args(), fake_asr)
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        events.append(_receive_until_type(ws, "partial"))
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    assert fake_asr.finish_calls == 1
    assert fake_asr.transcribe_calls == []
    assert not any(event.get("type") == "processing" for event in events)
    assert not any(event.get("type") == "sentence_reset" for event in events)


def test_ws_segment_final_redecode_failure_does_not_seal_tts(monkeypatch, tmp_path):
    class FailedRedecodeASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = "这是尚未通过最终校验的流式句子。"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

        def transcribe(self, audio, context="", language=None):
            super().transcribe(audio, context=context, language=language)
            raise RuntimeError("segment final decode failed")

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    trace_path = tmp_path / "segment-final-redecode-failed.jsonl"
    args = _args()
    args.final_redecode_on_stop = False
    args.segment_final_redecode = True
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.tts_revision_stable_sec = 0.0
    args.tts_latest_revision_grace_sec = 0.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = FailedRedecodeASR()
    app = _create_app(
        args,
        fake_asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "language": "Chinese"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.full(6400, 20_000, dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: message.get("type") == "sentence_translation",
        )
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "segment_final_redecode_failed" for row in trace_rows)
    assert any(row.get("event") == "tts_source_seal_deferred" for row in trace_rows)
    assert not any(row.get("event") == "tts_source_sealed" for row in trace_rows)


def test_ws_flushes_skipped_audio_tail_before_client_silence_cut(monkeypatch):
    class RecordingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_wavs = []

        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32).copy()
            self.streaming_wavs.append(samples)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = "句末弱音应该被完整识别。"
            return state

    decisions = iter((False, True))
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: next(decisions),
    )
    args = _args()
    args.final_redecode_on_stop = False
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 0.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.silent_decode_pre_roll_sec = 0.4
    args.silero_vad_shadow = False
    fake_asr = RecordingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        speech = np.full(6400, 20_000, dtype="<i2")
        quiet_tail = np.resize(np.array([200, -200], dtype="<i2"), 1600)
        ws.send_bytes(speech.tobytes())
        _receive_until_type(ws, "partial")
        ws.send_bytes(quiet_tail.tobytes())
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 22400,
            }
        )
        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.01)

    assert fake_asr.finish_calls == 1
    assert len(fake_asr.streaming_wavs) == 2
    np.testing.assert_allclose(
        fake_asr.streaming_wavs[-1],
        quiet_tail.astype(np.float32) / 32768.0,
    )


def test_ws_client_silence_discards_idle_noise_preroll(monkeypatch):
    class RecordingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_wavs = []

        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32).copy()
            self.streaming_wavs.append(samples)
            return super().streaming_transcribe(samples, state)

    decisions = iter((True, False))
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: next(decisions),
    )
    args = _args()
    args.silent_decode_pre_roll_sec = 0.4
    args.silero_vad_shadow = False
    fake_asr = RecordingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        idle_noise = np.resize(np.array([100, -100], dtype="<i2"), 1600)
        speech = np.resize(np.array([8000, -8000], dtype="<i2"), 3200)
        ws.send_bytes(idle_noise.tobytes())
        time.sleep(0.05)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 1000,
                "capture_sample_index": 17600,
            }
        )
        time.sleep(0.05)
        ws.send_bytes(speech.tobytes())
        _receive_until_type(ws, "partial")

    assert len(fake_asr.streaming_wavs) == 1
    np.testing.assert_allclose(
        fake_asr.streaming_wavs[0],
        speech.astype(np.float32) / 32768.0,
    )


def test_ws_client_silence_quarantines_two_term_context_hallucination(
    monkeypatch,
):
    hallucination = "民数记 尼希米记。"

    class ContextHallucinationASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            self.decode_count += 1
            state.language = "Chinese"
            state.text = hallucination
            state._raw_decoded = hallucination
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = hallucination
            state._raw_decoded = hallucination
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.0,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.final_redecode_on_stop = False
    fake_asr = ContextHallucinationASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": ["民数记", "尼希米记"],
            }
        )
        _receive_until_type(ws, "started")
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 1000,
                "capture_sample_index": 16000,
            }
        )
        ws.send_bytes(np.full(3200, 5000, dtype="<i2").tobytes())
        deadline = time.time() + 1.0
        while fake_asr.decode_count < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert fake_asr.decode_count == 1
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    assert not [
        event
        for event in events
        if event.get("type") == "partial" and hallucination in str(event.get("text", ""))
    ]
    final = next(event for event in events if event.get("type") == "final")
    assert hallucination not in str(final.get("text", ""))
    assert hallucination not in str(final.get("committed_text", ""))


def test_ws_context_long_silence_never_decodes_or_publishes_hotwords(monkeypatch):
    class CountingContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def streaming_transcribe(self, wav, state):
            self.decode_count += 1
            state.language = "Chinese"
            state.text = "尼希米 城墙 羊门。"
            return state

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    fake_asr = CountingContextASR()
    translator = _FakeTranslator()
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.final_redecode_on_stop = False
    events = []

    with TestClient(_create_app(args, fake_asr, translator=translator)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": ["尼希米", "城墙", "羊门"],
            }
        )
        _receive_until_type(ws, "started")
        for seconds in (3, 5, 10):
            ws.send_json(
                {
                    "type": "audio_silence",
                    "duration_ms": seconds * 1000,
                    "capture_sample_index": seconds * 16000,
                }
            )
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    assert fake_asr.decode_count == 0
    assert fake_asr.finish_calls == 0
    assert translator.calls == []
    assert not [
        event
        for event in events
        if event.get("type")
        in {"partial", "sentence_committed", "sentence_translation", "tts_job"}
    ]
    final = next(event for event in events if event.get("type") == "final")
    assert str(final.get("text", "") or "") == ""
    assert str(final.get("committed_text", "") or "") == ""


def test_ws_context_guard_keeps_aligned_tail_repair_from_spoken_segment(
    monkeypatch,
    tmp_path,
):
    streaming_text = "有人默默地最早来预备场。"
    repaired_text = "有人默默地最早来预备场地。"

    class ContextTailRepairASR(_FakeASR):
        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            return state

        def streaming_transcribe(self, wav, state):
            samples = np.asarray(wav, dtype=np.float32)
            state.audio_accum = np.concatenate((state.audio_accum, samples))
            state.language = "Chinese"
            state.text = streaming_text
            state._raw_decoded = streaming_text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = repaired_text
            state._raw_decoded = repaired_text
            return state

        def transcribe(self, audio, context="", language=None):
            super().transcribe(audio, context=context, language=language)
            return [SimpleNamespace(language="Chinese", text=repaired_text)]

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.0,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    trace_path = tmp_path / "context-tail-repair.jsonl"
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.segment_final_redecode = True
    args.final_redecode_on_stop = False
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.vad_silence_sec = 0.7
    args.vad_force_cut_sec = 1.8
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    translator = _FakeTranslator()
    fake_asr = ContextTailRepairASR()
    events = []

    with TestClient(_create_app(args, fake_asr, translator=translator)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": ["尼希米", "城墙", "羊门"],
            }
        )
        _receive_until_type(ws, "started")
        ws.send_bytes(np.full(6400, 20_000, dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        ws.send_json(
            {
                "type": "audio_silence",
                "duration_ms": 900,
                "capture_sample_index": 20800,
            }
        )
        committed = _poll_ws_with_ping(
            ws,
            events,
            lambda message: message.get("type") == "sentence_committed",
        )
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: message.get("type") == "sentence_translation",
        )

    assert committed["text"] == repaired_text
    assert translator.calls == [(repaired_text, "Chinese", "English")]
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "asr_context_final_tail_repair_trusted"
        for row in trace_rows
    )
    assert not any(
        row.get("event") == "asr_context_silent_segment_discarded"
        for row in trace_rows
    )


def test_ws_replays_skipped_audio_once_when_decode_resumes(monkeypatch):
    class RecordingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_wavs = []

        def streaming_transcribe(self, wav, state):
            self.streaming_wavs.append(np.asarray(wav, dtype=np.float32).copy())
            return super().streaming_transcribe(wav, state)

    decisions = iter((True, False))
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: next(decisions),
    )
    args = _args()
    args.silent_decode_pre_roll_sec = 0.4
    args.silero_vad_shadow = False
    fake_asr = RecordingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        first = np.array([100, 200, 300], dtype="<i2")
        second = np.array([400, 500, 600], dtype="<i2")
        ws.send_bytes(first.tobytes())
        time.sleep(0.05)
        ws.send_bytes(second.tobytes())
        _receive_until_type(ws, "partial")

    assert len(fake_asr.streaming_wavs) == 1
    expected = np.concatenate((first, second)).astype(np.float32) / 32768.0
    np.testing.assert_allclose(fake_asr.streaming_wavs[0], expected)


def test_ws_quarantines_context_fragment_after_silent_resume(monkeypatch, tmp_path):
    context_terms = ["尼希米", "城墙", "羊门", "粪门", "祭司", "圣经"]
    context_fragment = "所以说，城墙、羊门、粪门。"
    natural_text = "整本圣经的作用和要求正在这里继续说明。"

    class ResumeContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            self.decode_count += 1
            state.language = "Chinese"
            state.text = context_fragment if self.decode_count <= 2 else natural_text
            state._raw_decoded = state.text
            return state

    decisions = iter((True, True, True, False, False, False))
    decision_count = 0

    def decide(**kwargs):
        nonlocal decision_count
        del kwargs
        decision_count += 1
        return next(decisions)

    monkeypatch.setattr(demo_streaming_ws, "_should_skip_stream_decode", decide)
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.0,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.silent_decode_pre_roll_sec = 0.4
    args.chunk_size_sec = 0.1
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "resume-context-trace.jsonl")
    fake_asr = ResumeContextASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": context_terms,
            }
        )
        _receive_until_type(ws, "started")
        frame = np.resize(np.array([100, 200, 300], dtype="<i2"), 640).tobytes()
        for expected_count in range(1, 4):
            ws.send_bytes(frame)
            deadline = time.time() + 1.0
            while decision_count < expected_count and time.time() < deadline:
                time.sleep(0.01)
            assert decision_count >= expected_count

        ws.send_bytes(frame)
        deadline = time.time() + 1.0
        while fake_asr.decode_count < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert fake_asr.decode_count >= 1

        time.sleep(args.chunk_size_sec + 0.05)
        ws.send_bytes(frame)
        deadline = time.time() + 1.0
        while fake_asr.decode_count < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert fake_asr.decode_count >= 2

        ws.send_bytes(frame)
        partial = _receive_until_type(ws, "partial")

    assert partial["text"] == natural_text
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quarantined = [
        row for row in trace_rows if row.get("event") == "asr_context_resume_partial_quarantined"
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["text_chars"] == len(context_fragment)
    assert context_fragment not in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8")


def test_ws_discards_context_fragment_if_hard_cut_finalizes_silent_resume(
    monkeypatch,
    tmp_path,
):
    context_terms = ["尼希米", "城墙", "羊门", "粪门", "祭司", "圣经"]
    context_fragment = "所以说，城墙、羊门、粪门。"
    natural_text = "这是此前已经听到的正常语音。"

    class FinalizeContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            self.decode_count += 1
            state.language = "Chinese"
            state.text = natural_text if self.decode_count == 1 else context_fragment
            state._raw_decoded = state.text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = context_fragment
            state._raw_decoded = state.text
            return state

    decisions = iter((False, True, True, True, False))
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: next(decisions),
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.0,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.silent_decode_pre_roll_sec = 0.4
    args.chunk_size_sec = 0.1
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "resume-context-finalize-trace.jsonl")
    fake_asr = FinalizeContextASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": context_terms,
            }
        )
        _receive_until_type(ws, "started")
        frame = np.resize(np.array([100, 200, 300], dtype="<i2"), 640).tobytes()
        ws.send_bytes(frame)
        assert _receive_until_type(ws, "partial")["text"] == natural_text
        for _ in range(3):
            ws.send_bytes(frame)
            time.sleep(0.02)
        time.sleep(1.1)
        ws.send_bytes(frame)

        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert fake_asr.finish_calls >= 1
        ws.send_json({"type": "ping"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("type") == "pong":
                break

    committed = [
        str(event.get("text", "") or "").strip()
        for event in events
        if event.get("type") == "sentence_committed"
    ]
    assert context_fragment not in committed
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    discarded = [
        row for row in trace_rows if row.get("event") == "asr_context_silent_segment_discarded"
    ]
    assert len(discarded) == 1
    assert discarded[0]["text_hash8"] == hashlib.md5(context_fragment.encode("utf-8")).hexdigest()[:8]
    assert discarded[0]["fallback_snapshot_used"] is True
    assert discarded[0]["fallback_snapshot_hash8"] == hashlib.md5(
        natural_text.encode("utf-8")
    ).hexdigest()[:8]
    finalized = [row for row in trace_rows if row.get("event") == "segment_finalize_done"]
    assert len(finalized) == 1
    assert finalized[0]["final_text_chars"] == len(natural_text)


def test_ws_discards_context_fragment_if_stop_finalizes_silent_resume(
    monkeypatch,
    tmp_path,
):
    context_terms = ["尼希米", "城墙", "羊门", "粪门", "祭司", "圣经"]
    context_fragment = "所以说，城墙、羊门、粪门。"

    class StopContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            self.decode_count += 1
            state.language = "Chinese"
            state.text = context_fragment
            state._raw_decoded = state.text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = context_fragment
            state._raw_decoded = state.text
            return state

    decisions = iter((True, True, True, False))
    decision_count = 0

    def decide(**kwargs):
        nonlocal decision_count
        del kwargs
        decision_count += 1
        return next(decisions)

    monkeypatch.setattr(demo_streaming_ws, "_should_skip_stream_decode", decide)
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.0,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "streaming"
    args.silent_decode_pre_roll_sec = 0.4
    args.chunk_size_sec = 0.1
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "resume-context-stop-trace.jsonl")
    fake_asr = StopContextASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": context_terms,
            }
        )
        _receive_until_type(ws, "started")
        frame = np.resize(np.array([100, 200, 300], dtype="<i2"), 640).tobytes()
        for expected_count in range(1, 4):
            ws.send_bytes(frame)
            deadline = time.time() + 1.0
            while decision_count < expected_count and time.time() < deadline:
                time.sleep(0.01)
            assert decision_count >= expected_count
        ws.send_bytes(frame)
        deadline = time.time() + 1.0
        while fake_asr.decode_count < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert fake_asr.decode_count >= 1

        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    final = next(event for event in events if event.get("type") == "final")
    assert final["text"] == ""
    assert context_fragment not in str(final.get("committed_text", "") or "")
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    discarded = [
        row for row in trace_rows if row.get("event") == "asr_context_silent_segment_discarded"
    ]
    assert len(discarded) == 1
    assert discarded[0]["reason"] == "stop"
    assert discarded[0]["fallback_snapshot_used"] is False


def test_ws_silero_shadow_writes_observations_without_controlling_asr(monkeypatch, tmp_path):
    trace_path = tmp_path / "silero-shadow.jsonl"
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.8,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.silero_vad_shadow = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    args.subtitle_trace_log_partial_every = 20

    with TestClient(_create_app(args, _FakeASR())).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.full(512, 20_000, dtype="<i2").tobytes())
        partial = _receive_until_type(ws, "partial")

    assert partial["type"] == "partial"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    by_event = {row["event"]: row for row in rows}
    assert by_event["silero_shadow_ready"]["control_mode"] == "observe_only"
    assert by_event["silero_shadow_observation"]["probability"] == 0.8
    assert by_event["silero_shadow_observation"]["control_mode"] == "observe_only"


def test_ws_silero_rescue_decodes_quiet_speech_rejected_by_energy_gate(
    monkeypatch,
    tmp_path,
):
    class CountingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_calls = 0

        def streaming_transcribe(self, wav, state):
            self.streaming_calls += 1
            return super().streaming_transcribe(wav, state)

    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: 0.9,
            frame_samples=512,
            threshold=threshold,
        ),
    )
    trace_path = tmp_path / "silero-rescue.jsonl"
    args = _args()
    args.silero_vad_shadow = True
    args.silero_vad_rescue = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = CountingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.full(512, 320, dtype="<i2").tobytes())
        deadline = time.time() + 1.0
        while fake_asr.streaming_calls < 1 and time.time() < deadline:
            time.sleep(0.01)

    assert fake_asr.streaming_calls == 1
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rescued = [row for row in rows if row.get("event") == "silero_decode_rescue"]
    assert len(rescued) == 1
    assert rescued[0]["mean_probability"] == 0.9
    assert rescued[0]["control_mode"] == "decode_rescue"


def test_ws_silero_rescue_keeps_speech_at_start_of_batch_when_batch_ends_silent(
    monkeypatch,
    tmp_path,
):
    class CountingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_calls = 0

        def streaming_transcribe(self, wav, state):
            self.streaming_calls += 1
            return super().streaming_transcribe(wav, state)

    probabilities = iter([0.95] * 120 + [0.05] * 68)
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: next(probabilities),
            frame_samples=512,
            threshold=threshold,
        ),
    )
    trace_path = tmp_path / "silero-rescue-speech-then-silence.jsonl"
    args = _args()
    args.max_frame_samples = 200_000
    args.silero_vad_shadow = True
    args.silero_vad_rescue = True
    args.silero_vad_shadow_threshold = 0.5
    args.silero_vad_shadow_log_sec = 1.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    fake_asr = CountingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.full(188 * 512, 320, dtype="<i2").tobytes())
        deadline = time.time() + 1.0
        while fake_asr.streaming_calls < 1 and time.time() < deadline:
            time.sleep(0.01)

    assert fake_asr.streaming_calls == 1
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rescued = [row for row in rows if row.get("event") == "silero_decode_rescue"]
    assert len(rescued) == 1
    assert rescued[0]["probability"] == 0.05
    assert 0.5 < rescued[0]["mean_probability"] < 0.8


def test_ws_silero_decode_rescue_preserves_energy_vad_silence_endpoint(monkeypatch):
    class CountingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.streaming_calls = 0

        def streaming_transcribe(self, wav, state):
            self.streaming_calls += 1
            state.audio_accum = np.concatenate(
                (state.audio_accum, np.asarray(wav, dtype=np.float32))
            )
            state.language = "Chinese"
            state.text = "这是一句已经完成的测试。"
            return state

    decisions = iter((False, True))
    probabilities = iter([0.95] * 12 + [0.95] * 20 + [0.05] * 11)
    monkeypatch.setattr(
        demo_streaming_ws,
        "_should_skip_stream_decode",
        lambda **kwargs: next(decisions),
    )
    monkeypatch.setattr(
        demo_streaming_ws,
        "create_silero_onnx_observer",
        lambda threshold: SileroShadowObserver(
            runner=lambda frame: next(probabilities),
            frame_samples=512,
            threshold=threshold,
        ),
    )
    args = _args()
    args.final_redecode_on_stop = False
    args.vad_silence_sec = 0.3
    args.vad_force_cut_sec = 0.4
    args.vad_min_slice_sec = 0.5
    args.vad_min_active_sec = 0.2
    args.segment_overlap_sec = 0.0
    args.silero_vad_shadow = True
    args.silero_vad_rescue = True
    args.silero_vad_shadow_threshold = 0.5
    fake_asr = CountingASR()

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.full(6400, 20_000, dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        time.sleep(0.55)
        quiet_tail = np.resize(np.array([120, -120], dtype="<i2"), 16_000)
        ws.send_bytes(quiet_tail.tobytes())
        deadline = time.time() + 2.0
        while fake_asr.finish_calls < 1 and time.time() < deadline:
            time.sleep(0.01)

    assert fake_asr.streaming_calls == 2
    assert fake_asr.finish_calls == 1


def test_ws_silero_shadow_load_failure_keeps_asr_available(monkeypatch, tmp_path):
    trace_path = tmp_path / "silero-shadow-load-failure.jsonl"

    def fail_to_load(*, threshold):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(demo_streaming_ws, "create_silero_onnx_observer", fail_to_load)
    args = _args()
    args.silero_vad_shadow = True
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)

    with TestClient(_create_app(args, _FakeASR())).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(np.full(512, 20_000, dtype="<i2").tobytes())
        partial = _receive_until_type(ws, "partial")

    assert partial["type"] == "partial"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    unavailable = [row for row in rows if row["event"] == "silero_shadow_unavailable"]
    assert unavailable
    assert unavailable[0]["phase"] == "load"
    assert unavailable[0]["error_type"] == "RuntimeError"


def test_auth_disabled_keeps_http_and_websocket_access_open():
    app = _create_app(_args(), _FakeASR())
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "语音识别与翻译" in resp.text

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"


def test_index_renders_backend_context_limits_without_placeholders():
    args = _args()
    args.asr_context_max_terms = 7
    args.asr_context_max_chars = 48
    response = TestClient(_create_app(args, _FakeASR())).get("/")

    assert response.status_code == 200
    assert "const ASR_CONTEXT_MAX_TERMS = 7;" in response.text
    assert "const ASR_CONTEXT_MAX_CHARS = 48;" in response.text
    assert "__ASR_CONTEXT_MAX_" not in response.text


def test_auth_enabled_redirects_index_and_rejects_ws_without_session():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")

    app = _create_app(args, _FakeASR())
    client = TestClient(app)

    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in {302, 303, 307}
    assert resp.headers["location"] == "/login"

    with client.websocket_connect("/ws") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["message"] == "unauthorized"


def test_auth_login_sets_cookie_allows_access_and_logout_blocks_again():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    args.auth_session_ttl_sec = 3600

    app = _create_app(args, _FakeASR())
    client = TestClient(app)

    bad = client.post("/login", data={"username": "admin", "password": "wrong"}, follow_redirects=False)
    assert bad.status_code == 401
    assert "voxbridge_session" not in bad.headers.get("set-cookie", "")

    login = client.post("/login", data={"username": "admin", "password": "secret"}, follow_redirects=False)
    assert login.status_code in {302, 303, 307}
    assert login.headers["location"] == "/"
    cookie = login.headers["set-cookie"]
    assert "voxbridge_session=" in cookie
    assert "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()

    assert client.get("/").status_code == 200
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "ready"

    logout = client.post("/logout", follow_redirects=False)
    assert logout.status_code in {302, 303, 307}
    assert logout.headers["location"] == "/login"
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"


def test_listener_page_is_public_while_main_page_remains_protected():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    client = TestClient(app)

    page = client.get("/listen", follow_redirects=False)
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    hls_js = client.get("/listen/assets/hls.min.js", follow_redirects=False)
    assert hls_js.status_code == 200
    assert hls_js.headers["content-type"].startswith("text/javascript")
    assert hls_js.headers["cache-control"] == "public, max-age=86400"
    assert hls_js.headers["x-content-type-options"] == "nosniff"
    assert len(hls_js.content) > 500_000
    protected = client.get("/", follow_redirects=False)
    assert protected.status_code in {302, 303, 307}
    assert protected.headers["location"] == "/login"


def test_public_listener_qr_svg_is_cacheable_and_script_free():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    client = TestClient(_create_app(args, _FakeASR()))

    response = client.get("/listen/qr.svg", follow_redirects=False)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "<svg" in response.text
    assert len(response.content) > 1000
    assert "<script" not in response.text.lower()
    assert re.search(r'(?:href|src)=["\']https?://', response.text, re.I) is None


@pytest.mark.parametrize(
    "unsafe_next",
    ["https://attacker.example/listen", "//attacker.example/listen", "/\\attacker"],
)
def test_login_rejects_external_next_redirect(unsafe_next):
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    client = TestClient(_create_app(args, _FakeASR()))

    login = client.post(
        "/login",
        data={"username": "admin", "password": "secret", "next": unsafe_next},
        follow_redirects=False,
    )

    assert login.status_code in {302, 303, 307}
    assert login.headers["location"] == "/"


def test_tts_listener_websocket_authentication_and_ready_state():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())

    unauthenticated = TestClient(app)
    with unauthenticated.websocket_connect("/ws/tts") as ws:
        error = ws.receive_json()
        assert error == {"type": "error", "message": "unauthorized"}

    authenticated = TestClient(app)
    _login_tts_owner(authenticated)
    with authenticated.websocket_connect("/ws/tts") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "tts_listener_ready"
        assert ready["listener_id"]
        assert ready["tts_available"] is True
        assert ready["producer_active"] is False
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_listener_websocket_does_not_consume_asr_connection_quota():
    args = _args()
    args.max_connections = 1
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    client = TestClient(app)

    with client.websocket_connect("/ws") as asr_ws:
        assert asr_ws.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws/tts") as listener_ws:
            assert listener_ws.receive_json()["type"] == "tts_listener_ready"


def test_shared_hls_playlist_and_segments_are_public_capabilities(tmp_path):
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    args.tts_hls_root_dir = str(tmp_path)
    encoders = []

    def encoder_factory(root):
        encoder = _FakeHLSEncoder(root)
        encoders.append(encoder)
        return encoder

    args.tts_hls_encoder_factory = encoder_factory
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    first = TestClient(app)
    second = TestClient(app)
    listener_id = "iphone-a-12345678"
    playlist_url = f"/api/tts/live/{listener_id}/index.m3u8"

    playlist = first.get(playlist_url)
    shared_capability = second.get(playlist_url)

    assert playlist.status_code == 200
    assert playlist.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    assert playlist.headers["cache-control"] == "no-store"
    scoped_segment = f"/api/tts/live/{listener_id}/segments/segment_000000001.ts"
    assert scoped_segment in playlist.text
    assert shared_capability.status_code == 200
    segment = second.get(scoped_segment)
    assert segment.status_code == 200
    assert segment.content == b"shared-aac-segment"
    assert segment.headers["content-type"].startswith("video/mp2t")
    assert first.get(
        f"/api/tts/live/{listener_id}/segments/not-a-segment.txt"
    ).status_code == 404
    assert second.get("/api/tts/live/status").status_code == 200
    assert len(encoders) == 1


def test_public_hls_caption_feed_requires_matching_listener_lease(tmp_path):
    class ValidHLSSynthesizer:
        def synthesize(self, text, target_language, *, speed=None):
            del text, target_language, speed
            return SynthesizedAudio(
                _silent_wav_bytes(250),
                sample_rate=24000,
                duration_ms=250,
            )

    encoders = []

    def encoder_factory(root):
        encoder = _FakeHLSEncoder(root)
        encoders.append(encoder)
        return encoder

    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    args.tts_hls_root_dir = str(tmp_path)
    args.tts_hls_encoder_factory = encoder_factory
    app = _create_app(
        args,
        _FakeASR(),
        tts_synthesizer=ValidHLSSynthesizer(),
    )
    listener_id = "iphone-caption-12345678"
    item = TTSReadyItem(
        sentence_id="caption-source-1",
        revision=2,
        source_order=0,
        target_language="English",
        text="The stable translated sentence being spoken.",
    )
    second_item = TTSReadyItem(
        sentence_id="caption-source-2",
        revision=1,
        source_order=1,
        target_language="English",
        text="The next stable translated sentence.",
    )

    with TestClient(app) as client:
        playlist = client.get(f"/api/tts/live/{listener_id}/index.m3u8")
        assert playlist.status_code == 200
        client.portal.call(app.state.tts_hls.publish, item)
        client.portal.call(app.state.tts_hls.wait_idle)
        encoders[0].next_discardable_gap_before_ms = 1_000
        client.portal.call(app.state.tts_hls.publish, second_item)
        client.portal.call(app.state.tts_hls.wait_idle)

        captions = client.get(f"/api/tts/live/{listener_id}/captions")
        unknown = client.get(
            "/api/tts/live/iphone-caption-unknown-12345678/captions"
        )

    assert captions.status_code == 200
    assert captions.headers["cache-control"] == "no-store"
    payload = captions.json()
    assert payload["live_edge_at_ms"] == 102_100
    assert payload["cues"] == [
        {
            "cue_id": payload["cues"][0]["cue_id"],
            "start_at_ms": 100_000,
            "end_at_ms": 100_250,
            "text": "The stable translated sentence being spoken.",
            "discardable_gap_before_ms": 0,
            "resume_at_ms": None,
        },
        {
            "cue_id": payload["cues"][1]["cue_id"],
            "start_at_ms": 101_550,
            "end_at_ms": 101_800,
            "text": "The next stable translated sentence.",
            "discardable_gap_before_ms": 1_000,
            "resume_at_ms": 101_250,
        },
    ]
    assert re.fullmatch(r"[0-9a-f]{16}", payload["cues"][0]["cue_id"])
    assert re.fullmatch(r"[0-9a-f]{16}", payload["cues"][1]["cue_id"])
    assert unknown.status_code == 404


def test_public_hls_capacity_rejects_only_new_listener_ids(tmp_path):
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    args.tts_hls_root_dir = str(tmp_path)
    args.tts_hls_max_listeners = 1
    args.tts_hls_encoder_factory = lambda root: _FakeHLSEncoder(root)
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    client = TestClient(app)
    first = "/api/tts/live/public-a-12345678/index.m3u8"
    second = "/api/tts/live/public-b-12345678/index.m3u8"

    assert client.get(first).status_code == 200
    assert client.get(first).status_code == 200
    rejected = client.get(second)

    assert rejected.status_code == 429
    assert rejected.json()["detail"] == "HLS listener capacity reached"
    assert app.state.tts_hls.listener_count == 1
    assert client.get(first).status_code == 200


def test_removing_one_hls_listener_does_not_stop_shared_encoder(tmp_path):
    args = _args()
    args.tts_hls_root_dir = str(tmp_path)
    encoders = []

    def encoder_factory(root):
        encoder = _FakeHLSEncoder(root)
        encoders.append(encoder)
        return encoder

    args.tts_hls_encoder_factory = encoder_factory
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    with TestClient(app) as client:
        idle = client.get("/api/tts/live/status").json()
        assert idle["speech_epoch_id"] == ""
        assert idle["translated_audio_backlog_ms"] == 0
        assert idle["translated_audio_backlog_count"] == 0

        first = "/api/tts/live/iphone-a-12345678/index.m3u8"
        second = "/api/tts/live/iphone-b-12345678/index.m3u8"
        assert client.get(first).status_code == 200
        assert client.get(second).status_code == 200
        assert app.state.tts_hls.listener_count == 2

        removed = client.delete("/api/tts/live/iphone-a-12345678")

        assert removed.json() == {"ok": True}
        assert app.state.tts_hls.listener_count == 1
        assert encoders[0].closed == 0
        assert client.get(second).status_code == 200
        status = client.get("/api/tts/live/status").json()
        assert status["listener_count"] == 1
        assert status["encoder_active"] is True
        assert status["synthesis_active"] is False
        assert status["preparation_queue_depth"] == 0
        assert status["preparation_active"] is False
        assert status["prepared_audio_count"] == 0
        assert status["pending_audio_ms"] == 1750
        assert status["translated_audio_backlog_ms"] == 1750
        assert status["translated_audio_backlog_count"] == 0
        assert status["translated_audio_backlog_estimated"] is False
        assert status["speech_epoch_id"].startswith("epoch-")
        assert status["global_speed_mode"] == "auto"
        assert status["global_speed_multiplier"] == 1.0
        assert status["tts_effective_speed"] == pytest.approx(1.05)

    assert encoders[0].closed == 1


def test_shared_hls_uses_configured_baseline_and_global_auto_rollback(tmp_path):
    args = _args()
    args.tts_hls_root_dir = str(tmp_path)
    args.tts_speed = 1.1
    args.disable_tts_global_auto_speed = True
    args.tts_hls_encoder_factory = lambda root: _FakeHLSEncoder(root)

    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())

    status = app.state.tts_hls.status
    assert status.global_speed_mode == "fixed"
    assert status.global_speed_multiplier == 1.0
    assert status.tts_effective_speed == pytest.approx(1.1)


def test_broadcast_audio_is_shared_across_authenticated_listeners():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    synth = _FakeTTSSynthesizer()
    app = _create_app(args, _FakeASR(), tts_synthesizer=synth)
    first_client = TestClient(app)
    second_client = TestClient(app)
    foreign_client = TestClient(app)
    first_owner = _login_tts_owner(first_client)
    second_owner = _login_tts_owner(second_client)
    _login_tts_owner(foreign_client)
    first = app.state.tts_broadcast.register(first_owner)
    second = app.state.tts_broadcast.register(second_owner)
    job = app.state.tts_broadcast.publish(
        TTSReadyItem("s1", 1, 0, "English", "Stable translation.")
    )
    endpoint = f"/api/tts/broadcast/jobs/{job.job_id}/audio"

    first_audio = first_client.post(
        endpoint,
        headers={"X-TTS-Listener-ID": first.listener_id},
    )
    second_audio = second_client.post(
        endpoint,
        headers={"X-TTS-Listener-ID": second.listener_id},
    )
    foreign_audio = foreign_client.post(
        endpoint,
        headers={"X-TTS-Listener-ID": first.listener_id},
    )

    assert first_audio.status_code == 200
    assert first_audio.content == b"RIFF-fake-wav"
    assert first_audio.headers["cache-control"] == "no-store"
    assert first_audio.headers["x-tts-sample-rate"] == "24000"
    assert first_audio.headers["x-tts-duration-ms"] == "750"
    assert second_audio.content == first_audio.content
    assert synth.calls == [("Stable translation.", "English")]
    assert foreign_audio.status_code == 404

    assert app.state.tts_broadcast.acknowledge(job.job_id, first.listener_id, first_owner)
    assert app.state.tts_broadcast.job_count == 1
    assert app.state.tts_broadcast.acknowledge(job.job_id, second.listener_id, second_owner)
    assert app.state.tts_broadcast.job_count == 0


def test_broadcast_audio_requires_listener_header_and_available_synthesizer():
    app = _create_app(_args(), _FakeASR())
    client = TestClient(app)
    listener = app.state.tts_broadcast.register("anonymous")
    job = app.state.tts_broadcast.publish(
        TTSReadyItem("s1", 1, 0, "English", "Stable translation.")
    )
    endpoint = f"/api/tts/broadcast/jobs/{job.job_id}/audio"

    assert client.post(endpoint).status_code == 404
    unavailable = client.post(
        endpoint,
        headers={"X-TTS-Listener-ID": listener.listener_id},
    )
    assert unavailable.status_code == 503


def test_tts_audio_is_authenticated_cached_and_acknowledged():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    synth = _FakeTTSSynthesizer()
    app = _create_app(args, _FakeASR(), tts_synthesizer=synth)
    client = TestClient(app)
    owner_key = _login_tts_owner(client)
    job = app.state.tts_jobs.create(
        owner_key=owner_key,
        client_id="client-a-12345678",
        sentence_id="s1",
        revision=1,
        source_order=0,
        target_language="English",
        text="Stable translation.",
    )

    response = client.post(f"/api/tts/jobs/{job.job_id}/audio")
    retry = client.post(f"/api/tts/jobs/{job.job_id}/audio")

    assert response.status_code == 200
    assert response.content == b"RIFF-fake-wav"
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-tts-sample-rate"] == "24000"
    assert response.headers["x-tts-duration-ms"] == "750"
    assert retry.content == response.content
    assert synth.calls == [("Stable translation.", "English")]
    assert client.delete(f"/api/tts/jobs/{job.job_id}").json() == {"ok": True}
    assert client.post(f"/api/tts/jobs/{job.job_id}/audio").status_code == 404


def test_tts_job_owner_isolation_returns_not_found():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    owner_client = TestClient(app)
    other_client = TestClient(app)
    owner_key = _login_tts_owner(owner_client)
    _login_tts_owner(other_client)
    job = app.state.tts_jobs.create(
        owner_key=owner_key,
        client_id="client-a-12345678",
        sentence_id="s1",
        revision=1,
        source_order=0,
        target_language="English",
        text="Private translation.",
    )

    assert other_client.post(f"/api/tts/jobs/{job.job_id}/audio").status_code == 404
    assert other_client.delete(f"/api/tts/jobs/{job.job_id}").status_code == 404
    assert owner_client.post(f"/api/tts/jobs/{job.job_id}/audio").status_code == 200


def test_tts_client_cancellation_preserves_other_clients():
    app = _create_app(_args(), _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    client = TestClient(app)
    jobs = [
        app.state.tts_jobs.create(
            owner_key="anonymous",
            client_id=client_id,
            sentence_id=f"s{index}",
            revision=1,
            source_order=index,
            target_language="English",
            text=f"Translation {index}.",
        )
        for index, client_id in enumerate(
            ["client-a-12345678", "client-a-12345678", "client-b-12345678"]
        )
    ]

    response = client.delete("/api/tts/clients/client-a-12345678/jobs")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "removed": 2}
    assert client.post(f"/api/tts/jobs/{jobs[0].job_id}/audio").status_code == 404
    assert client.post(f"/api/tts/jobs/{jobs[2].job_id}/audio").status_code == 200


def test_tts_audio_returns_503_when_synthesizer_is_unavailable():
    app = _create_app(_args(), _FakeASR())
    client = TestClient(app)
    job = app.state.tts_jobs.create(
        owner_key="anonymous",
        client_id="client-a-12345678",
        sentence_id="s1",
        revision=1,
        source_order=0,
        target_language="English",
        text="Stable translation.",
    )

    response = client.post(f"/api/tts/jobs/{job.job_id}/audio")

    assert response.status_code == 503
    assert response.json()["detail"] == "TTS is unavailable"


def test_broadcast_stress_preserves_order_and_shares_synthesis_across_listeners():
    args = _args()
    args.tts_listener_queue_size = 256
    synthesizer = _FakeTTSSynthesizer()
    app = _create_app(
        args,
        _FakeASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=synthesizer,
    )
    first = app.state.tts_broadcast.register("anonymous")
    second = app.state.tts_broadcast.register("anonymous")
    jobs = [
        app.state.tts_broadcast.publish(
            TTSReadyItem(
                sentence_id=f"sentence-{index}",
                revision=1,
                source_order=index,
                target_language="English",
                text=f"Stable translation {index}.",
            )
        )
        for index in range(200)
    ]
    first_events = [first.queue.get_nowait() for _ in jobs]
    second_events = [second.queue.get_nowait() for _ in jobs]

    assert [event["source_order"] for event in first_events] == list(range(200))
    assert [event["source_order"] for event in second_events] == list(range(200))

    with TestClient(app) as client:
        for job in jobs:
            endpoint = f"/api/tts/broadcast/jobs/{job.job_id}/audio"
            first_audio = client.post(
                endpoint,
                headers={"X-TTS-Listener-ID": first.listener_id},
            )
            second_audio = client.post(
                endpoint,
                headers={"X-TTS-Listener-ID": second.listener_id},
            )
            assert first_audio.status_code == 200
            assert second_audio.status_code == 200
            assert first_audio.content == second_audio.content
            assert app.state.tts_broadcast.acknowledge(
                job.job_id,
                first.listener_id,
                "anonymous",
            )

        for job in jobs[:100]:
            assert app.state.tts_broadcast.acknowledge(
                job.job_id,
                second.listener_id,
                "anonymous",
            )

    assert len(synthesizer.calls) == 200
    assert app.state.tts_broadcast.job_count == 100
    assert app.state.tts_broadcast.unregister(second.listener_id, "anonymous") == 100
    assert app.state.tts_broadcast.job_count == 0


class _StableTTSSentenceASR(_FakeASR):
    sentences = (
        "这是第一句已经稳定完成并且长度足够。",
        "这是第二句已经稳定完成并且长度足够。",
        "这是第三句已经稳定完成并且长度足够。",
    )

    def streaming_transcribe(self, wav, state):
        state.language = "Chinese"
        state.text = "".join(self.sentences)
        state.audio_accum = np.concatenate([state.audio_accum, np.asarray(wav, dtype=np.float32)])
        return state

    def finish_streaming_transcribe(self, state):
        self.finish_calls += 1
        state.language = "Chinese"
        state.text = "".join(self.sentences)
        return state


class _FastRevisionASR(_FakeASR):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.s1 = "First sentence is stable and complete."
        self.s2_short = "Second sentence starts as a complete long sentence."
        self.s2_long = (
            "Second sentence starts as a complete long sentence and later receives "
            "important extra words."
        )
        self.s3 = "Third sentence is stable and complete."

    def streaming_transcribe(self, wav, state):
        self.calls += 1
        state.language = "English"
        second = self.s2_short if self.calls <= 2 else self.s2_long
        state.text = f"{self.s1} {second} {self.s3}"
        return state

    def finish_streaming_transcribe(self, state):
        self.finish_calls += 1
        state.language = "English"
        state.text = f"{self.s1} {self.s2_long} {self.s3}"
        return state


def _collect_through_final(ws, max_steps=160):
    events = []
    for _ in range(max_steps):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") == "final":
            return events
    pytest.fail(f"did not receive final, seen={[event.get('type') for event in events]}")


def _drain_listener_events(subscription):
    events = []
    while True:
        try:
            events.append(subscription.queue.get_nowait())
        except asyncio.QueueEmpty:
            return events


def _poll_ws_with_ping(ws, events, predicate, max_polls=40):
    for _ in range(max_polls):
        ws.send_json({"type": "ping"})
        while True:
            message = ws.receive_json()
            events.append(message)
            if predicate(message):
                return message
            if message.get("type") == "pong":
                break
        time.sleep(0.01)
    pytest.fail("expected WebSocket event was not emitted")


def test_ws_tts_defaults_off_and_marks_translation_stable():
    args = _args()
    args.final_redecode_on_stop = False
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["tts_available"] is True
        assert ready["tts_enabled"] is False
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    translations = [event for event in events if event.get("type") == "sentence_translation"]
    assert translations
    assert all(event.get("is_stable") is True for event in translations)
    assert not [event for event in events if event.get("type") == "tts_job"]


def test_ws_broadcasts_stable_translations_without_legacy_tts_fields():
    args = _args()
    args.final_redecode_on_stop = False
    args.translation_workers = 3
    args.tts_final_translation_drain_sec = 2.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        started = _receive_until_type(ws, "started")
        assert started["tts_enabled"] is False
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    broadcast_events = []
    while True:
        try:
            broadcast_events.append(listener.queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    jobs = [event for event in broadcast_events if event.get("type") == "tts_job"]
    statuses = [
        event for event in broadcast_events if event.get("type") == "producer_status"
    ]
    assert [event["source_order"] for event in jobs] == [0, 1, 2]
    assert statuses == [
        {"type": "producer_status", "active": True},
        {"type": "producer_status", "active": False},
    ]
    assert max(broadcast_events.index(event) for event in jobs) < broadcast_events.index(
        statuses[-1]
    )
    assert app.state.tts_broadcast.job_count == 3
    assert not [event for event in events if event.get("type") == "tts_job"]


def test_ws_tts_stability_scheduler_releases_without_more_audio():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 0.12
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "sentence_translation")
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]

        time.sleep(0.16)
        jobs = [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]
        assert jobs


def test_ws_translation_done_prepares_exact_tts_revision_before_stability_release():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 60.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    prepared = []

    async def record_preparation(item):
        prepared.append(item)
        return True

    app.state.tts_hls.prepare = record_preparation
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        translated = _receive_until_type(ws, "sentence_translation")

        deadline = time.monotonic() + 1.0
        while len(prepared) < len(_StableTTSSentenceASR.sentences) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(prepared) == len(_StableTTSSentenceASR.sentences)
        assert len(
            {(item.sentence_id, item.revision, item.text) for item in prepared}
        ) == len(prepared)
        matching = [
            item
            for item in prepared
            if item.sentence_id == translated["sentence_id"]
            and item.revision == translated["revision"]
        ]
        assert len(matching) == 1
        assert matching[0].text == translated["translation"]
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]


def test_ws_finish_force_releases_ready_translations_before_producer_inactive():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 60.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "sentence_translation")
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]

        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)

    broadcast_events = _drain_listener_events(listener)
    jobs = [event for event in broadcast_events if event.get("type") == "tts_job"]
    inactive = [
        event
        for event in broadcast_events
        if event.get("type") == "producer_status" and event.get("active") is False
    ]
    assert [event["source_order"] for event in jobs] == [0, 1, 2]
    assert inactive
    assert max(broadcast_events.index(event) for event in jobs) < broadcast_events.index(
        inactive[-1]
    )


def test_ws_abrupt_disconnect_discards_pending_tts():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 0.2
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "sentence_translation")
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]

    time.sleep(0.24)
    assert not [
        event
        for event in _drain_listener_events(listener)
        if event.get("type") == "tts_job"
    ]


def test_tts_quiet_window_is_bypassed_only_by_orderly_finalization():
    source = inspect.getsource(demo_streaming_ws._create_app)

    assert source.count("_drain_tts_stability(force=True)") == 1
    vad_start = source.index("async def _maybe_vad_silence_cut")
    consumer_start = source.index("async def _audio_consumer")
    assert "_drain_tts_stability(force=True)" not in source[vad_start:consumer_start]


def test_segment_finalization_seals_tts_after_final_reconciliation():
    source = inspect.getsource(demo_streaming_ws._create_app)
    finalize_start = source.index("async def _finalize_segment_and_rotate")
    finalize_end = source.index("async def _maybe_vad_silence_cut", finalize_start)
    finalize_source = source[finalize_start:finalize_end]

    reconcile_pos = finalize_source.index("final_reconcile=True")
    seal_pos = finalize_source.index("await _seal_tts_sources_through_current_segment")
    reset_pos = finalize_source.index("_reset_completed_candidate_cursor()")

    assert reconcile_pos < seal_pos < reset_pos


def test_ws_rollback_safe_source_avoids_global_grace_and_hardcut_tail_waits_for_finish(tmp_path):
    class _TwoSentenceSegmentASR(_FakeASR):
        sentences = (
            "这是第一句已经稳定完成并且长度足够。",
            "这是第二句已经稳定完成并且长度足够。",
        )

        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.text = (
                "".join(self.sentences)
                if state.segment_no <= 2
                else self.sentences[1]
            )
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = (
                "".join(self.sentences)
                if state.segment_no <= 2
                else self.sentences[1]
            )
            return state

    args = _args()
    args.final_redecode_on_stop = False
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 1
    args.tts_revision_stable_sec = 0.0
    args.tts_latest_revision_grace_sec = 60.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "tts-segment-seal.jsonl")
    app = _create_app(
        args,
        _TwoSentenceSegmentASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        frame = np.array([0, 1000, -1000] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        ws.send_bytes(frame)
        _receive_until_type(ws, "sentence_translation")
        released_before_cut = [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]
        assert [event["source_order"] for event in released_before_cut] == [0]

        time.sleep(1.1)
        ws.send_bytes(frame)
        jobs = list(released_before_cut)
        _receive_until_type(ws, "partial")
        jobs.extend(
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        )
        assert [event["source_order"] for event in jobs] == [0]

        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)
        jobs.extend(
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        )

    assert [event["source_order"] for event in jobs] == [0, 1]
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    release_reasons = [
        row.get("release_reason")
        for row in trace_rows
        if row.get("event") == "tts_stability_release"
    ]
    assert release_reasons == ["rollback_safe", "final_force"]


def test_ws_translation_direction_change_discards_pending_tts():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 0.3
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "translation_direction": "zh2en"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "sentence_translation")
        _drain_listener_events(listener)

        ws.send_json(
            {
                "type": "set_translation_direction",
                "translation_direction": "en2zh",
            }
        )
        direction = _receive_until_type(ws, "translation_direction")
        assert direction["translation_direction"] == "en2zh"
        time.sleep(0.35)

        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job"
        ]


def test_ws_does_not_retain_broadcast_jobs_without_listeners():
    args = _args()
    args.final_redecode_on_stop = False
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)

    assert app.state.tts_broadcast.job_count == 0


def test_ws_tts_jobs_follow_source_order_and_precede_final():
    class _OutOfOrderTranslator(_FakeTranslator):
        def translate(self, text: str, source_language: str = None, target_language: str = None):
            if text == _StableTTSSentenceASR.sentences[0]:
                time.sleep(0.15)
            return super().translate(text, source_language, target_language)

    args = _args()
    args.final_redecode_on_stop = False
    args.translation_workers = 3
    args.tts_final_translation_drain_sec = 2.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_OutOfOrderTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["tts_available"] is True
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        started = _receive_until_type(ws, "started")
        assert started["tts_enabled"] is True
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert len(jobs) == 3
    assert [event["source_order"] for event in jobs] == [0, 1, 2]
    assert len({event["sentence_id"] for event in jobs}) == 3
    assert all(event["target_language"] == "English" for event in jobs)
    assert all(event["is_stable"] is True for event in jobs)
    assert max(events.index(event) for event in jobs) < next(
        index for index, event in enumerate(events) if event.get("type") == "final"
    )


@pytest.mark.parametrize(
    ("direction", "source_label", "target_label", "expected_target"),
    (
        ("zh2en", "中文", "英文", "English"),
        ("en2zh", "中文", "英文", "Chinese"),
        ("zh2en", "中文", "英语", "English"),
    ),
)
def test_ws_tts_jobs_use_canonical_language_for_localized_translation_labels(
    direction, source_label, target_label, expected_target
):
    args = _args()
    args.final_redecode_on_stop = False
    args.translation_source_language = source_label
    args.translation_target_language = target_label
    args.tts_final_translation_drain_sec = 2.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "translation_direction": direction,
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert jobs
    assert {event["target_language"] for event in jobs} == {expected_target}


def test_ws_tts_canonical_stop_redecode_does_not_repeat_issued_sentences():
    args = _args()
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    args.translation_workers = 3
    args.tts_final_translation_drain_sec = 2.0
    asr = _StableTTSSentenceASR()
    asr.transcribe_language = "Chinese"
    asr.transcribe_text = "".join(asr.sentences)
    app = _create_app(
        args,
        asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        events = []
        for _ in range(80):
            event = ws.receive_json()
            events.append(event)
            if event.get("type") == "partial":
                break
        else:
            pytest.fail("did not receive partial")
        for _ in range(80):
            event = ws.receive_json()
            events.append(event)
            if event.get("type") == "tts_job":
                break
        else:
            pytest.fail("did not receive a streaming TTS job before stop")
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert len(jobs) == len(asr.sentences)


def test_ws_tts_toggle_does_not_replay_earlier_sentences_at_stop():
    class _RollbackSafeTTSSentenceASR(_StableTTSSentenceASR):
        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.text = f"{''.join(self.sentences)}后续内容正在生成"
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            return state

    args = _args()
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    args.translation_workers = 3
    args.tts_final_translation_drain_sec = 2.0
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 1
    asr = _RollbackSafeTTSSentenceASR()
    asr.transcribe_language = "Chinese"
    asr.transcribe_text = "".join(asr.sentences)
    app = _create_app(
        args,
        asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        events = []
        for _ in range(2):
            ws.send_bytes(raw)
            while True:
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "partial":
                    break
        while len([event for event in events if event.get("type") == "tts_job"]) < 3:
            events.append(ws.receive_json())

        ws.send_json(
            {
                "type": "set_tts_enabled",
                "enabled": False,
                "tts_client_id": "client-a-12345678",
            }
        )
        disabled = _receive_until_type(ws, "tts_status")
        assert disabled["status"] == "disabled"
        ws.send_json(
            {
                "type": "set_tts_enabled",
                "enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        enabled = _receive_until_type(ws, "tts_status")
        assert enabled["status"] == "enabled"
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert len(jobs) == len(asr.sentences)


def test_ws_stop_waits_for_inflight_tts_publication_before_redecode(monkeypatch):
    class _TwoSentenceASR(_StableTTSSentenceASR):
        sentences = _StableTTSSentenceASR.sentences[:2]

    send_started = threading.Event()
    release_send = threading.Event()
    original_send_json = demo_streaming_ws.WebSocket.send_json
    blocked_once = False

    async def delayed_send_json(websocket, data, *args, **kwargs):
        nonlocal blocked_once
        if (
            data.get("type") == "tts_job"
            and data.get("source_order") == 0
            and not blocked_once
        ):
            blocked_once = True
            send_started.set()
            await asyncio.to_thread(release_send.wait)
        return await original_send_json(websocket, data, *args, **kwargs)

    monkeypatch.setattr(demo_streaming_ws.WebSocket, "send_json", delayed_send_json)
    args = _args()
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    args.translation_workers = 2
    args.tts_final_translation_drain_sec = 0.1
    asr = _TwoSentenceASR()
    asr.transcribe_language = "Chinese"
    asr.transcribe_text = "".join(asr.sentences)
    app = _create_app(
        args,
        asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    events = []
    frame = np.array([0, 1000, -1000], dtype="<i2").tobytes()

    try:
        with TestClient(app).websocket_connect("/ws") as ws:
            events.append(ws.receive_json())
            ws.send_json(
                {
                    "type": "start",
                    "tts_enabled": True,
                    "tts_client_id": "client-a-12345678",
                }
            )
            while not any(event.get("type") == "started" for event in events):
                events.append(ws.receive_json())
            for expected_partials in (1, 2):
                ws.send_bytes(frame)
                while len([event for event in events if event.get("type") == "partial"]) < expected_partials:
                    events.append(ws.receive_json())
            assert send_started.wait(timeout=3.0)
            ws.send_json({"type": "finish", "mode": "stop"})
            release_timer = threading.Timer(1.0, release_send.set)
            release_timer.start()
            while not any(event.get("type") == "final" for event in events):
                events.append(ws.receive_json())
            release_timer.cancel()
    finally:
        release_send.set()

    jobs = [event for event in events if event.get("type") == "tts_job"]
    slow_statuses = [
        event
        for event in events
        if event.get("type") == "tts_status"
        and event.get("status") == "translation_drain_timeout"
    ]
    assert [event["source_order"] for event in jobs] == [0, 1]
    assert [event.get("phase") for event in slow_statuses] == ["before_final_redecode"]
    assert asr.transcribe_calls == []
    assert asr.finish_calls == 1


def test_ws_tts_failed_earlier_translation_does_not_block_later_jobs():
    class _FirstFailsTranslator(_FakeTranslator):
        def translate(self, text: str, source_language: str = None, target_language: str = None):
            if text == _StableTTSSentenceASR.sentences[0]:
                return ""
            return super().translate(text, source_language, target_language)

    args = _args()
    args.final_redecode_on_stop = False
    args.tts_final_translation_drain_sec = 2.0
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FirstFailsTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert [event["source_order"] for event in jobs] == [1, 2]


def test_ws_tts_final_waits_past_drain_warning_for_stable_translations():
    first_started = threading.Event()

    class _SlowFirstTranslator(_FakeTranslator):
        def translate(self, text: str, source_language: str = None, target_language: str = None):
            if text == _StableTTSSentenceASR.sentences[0]:
                first_started.set()
                time.sleep(0.35)
            return super().translate(text, source_language, target_language)

    args = _args()
    args.final_redecode_on_stop = False
    args.translation_workers = 3
    args.tts_final_translation_drain_sec = 0.05
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_SlowFirstTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(6):
            ws.send_bytes(raw)
            _receive_until_type(ws, "partial")
            if first_started.wait(timeout=0.2):
                break
        assert first_started.is_set()
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    jobs = [event for event in events if event.get("type") == "tts_job"]
    assert [event["source_order"] for event in jobs] == [0, 1, 2]


def test_ws_disabling_tts_cancels_client_jobs_and_reports_status():
    app = _create_app(
        _args(),
        _FakeASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "tts_enabled": True,
                "tts_client_id": "client-a-12345678",
            }
        )
        _receive_until_type(ws, "started")
        job = app.state.tts_jobs.create(
            owner_key="anonymous",
            client_id="client-a-12345678",
            sentence_id="s1",
            revision=1,
            source_order=0,
            target_language="English",
            text="Stable translation.",
        )
        ws.send_json(
            {
                "type": "set_tts_enabled",
                "enabled": False,
                "tts_client_id": "client-a-12345678",
            }
        )
        status = _receive_until_type(ws, "tts_status")

    assert status["status"] == "disabled"
    assert status["tts_enabled"] is False
    assert client.post(f"/api/tts/jobs/{job.job_id}/audio").status_code == 404


def test_debug_file_requires_auth_and_can_be_disabled(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("debug-ok", encoding="utf-8")

    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")

    app = _create_app(args, _FakeASR())
    client = TestClient(app)

    unauth = client.get("/__debug/file", params={"path": str(probe)})
    assert unauth.status_code == 401

    login = client.post("/login", data={"username": "admin", "password": "secret"}, follow_redirects=False)
    assert login.status_code in {302, 303, 307}
    ok = client.get("/__debug/file", params={"path": str(probe)})
    assert ok.status_code == 200
    assert ok.text == "debug-ok"

    args_disabled = _args()
    args_disabled.auth_enabled = True
    args_disabled.auth_username = "admin"
    args_disabled.auth_password_hash = _hash_auth_password("secret")
    args_disabled.disable_debug_file = True
    disabled_app = _create_app(args_disabled, _FakeASR())
    disabled_client = TestClient(disabled_app)
    disabled_client.post("/login", data={"username": "admin", "password": "secret"}, follow_redirects=False)
    disabled = disabled_client.get("/__debug/file", params={"path": str(probe)})
    assert disabled.status_code == 404


def test_ws_rejects_bad_binary_frame():
    app = _create_app(_args(), _FakeASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_bytes(b"\x00")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "even" in err["message"]


def test_ws_transformers_mode_partial_and_final():
    args = _args()
    args.backend = "transformers"
    app = _create_app(args, _FakeASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000] * 7000, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        partial = ws.receive_json()
        assert partial["type"] == "partial"
        assert partial["language"] == "Chinese"
        assert partial["text"] == "pseudo"

        ws.send_text('{"type":"finish"}')
        final = _receive_until_type(ws, "final")
        assert final["type"] == "final"
        assert final["text"] == "pseudo"


def test_ws_transformers_mode_applies_session_context_to_final_decode():
    fake_asr = _FakeASR()
    args = _args()
    args.backend = "transformers"
    args.asr_context_apply_mode = "segment_final"
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": ["Elisha", "Jordan"],
            }
        )
        started = _receive_until_type(ws, "started")
        assert started["asr_context_active"] is True
        assert started["asr_context_term_count"] == 2

        raw = np.array([0, 1000, -1000] * 7000, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        assert fake_asr.transcribe_calls[-1]["context"] == ""

        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final")

    assert fake_asr.transcribe_calls[-1]["context"] == "Elisha Jordan"


def test_ws_uses_cli_force_language_for_initial_state():
    fake_asr = _FakeASR()
    args = _args()
    args.force_language = "Chinese"
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready

    assert fake_asr.init_calls[-1]["language"] == "Chinese"


def test_ws_passes_empty_context_when_schedule_is_disabled():
    fake_asr = _FakeASR()
    app = _create_app(_args(), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

    assert fake_asr.init_calls[-1]["context"] == ""


def _write_context_schedule(tmp_path, *, language="English"):
    path = tmp_path / "context.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": language,
                "global_terms": ["ScheduledTerm"],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _context_enabled_args(tmp_path):
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(_write_context_schedule(tmp_path))
    args.asr_context_apply_mode = "streaming"
    args.asr_context_max_terms = 24
    args.asr_context_max_chars = 160
    args.asr_context_lookaround_sec = 0.0
    args.final_redecode_on_stop = False
    return args


def test_ws_nonempty_session_context_overrides_schedule_and_reports_metadata(tmp_path):
    fake_asr = _FakeASR()
    app = _create_app(_context_enabled_args(tmp_path), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "translation_direction": "en2zh",
                "asr_context_terms": ["Elisha", "Jordan"],
            }
        )
        started = _receive_until_type(ws, "started")

    assert started["asr_context_active"] is True
    assert started["asr_context_term_count"] == 2
    assert started["asr_context_chars"] == len("Elisha Jordan")
    assert fake_asr.init_calls[-1]["context"] == "Elisha Jordan"


def test_ws_explicit_empty_session_context_disables_schedule(tmp_path):
    fake_asr = _FakeASR()
    app = _create_app(_context_enabled_args(tmp_path), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": [],
            }
        )
        started = _receive_until_type(ws, "started")

    assert started["asr_context_active"] is False
    assert started["asr_context_term_count"] == 0
    assert started["asr_context_chars"] == 0
    assert fake_asr.init_calls[-1]["context"] == ""


def test_ws_omitted_session_context_preserves_schedule_for_legacy_clients(tmp_path):
    fake_asr = _FakeASR()
    app = _create_app(_context_enabled_args(tmp_path), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "language": "English"})
        started = _receive_until_type(ws, "started")

    assert started["asr_context_active"] is True
    assert started["asr_context_term_count"] == 1
    assert fake_asr.init_calls[-1]["context"] == "ScheduledTerm"


def test_ws_session_context_survives_hard_cut_rotation(tmp_path):
    fake_asr = _FakeASR()
    args = _context_enabled_args(tmp_path)
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": ["Elisha", "Jordan"],
            }
        )
        _receive_until_type(ws, "started")
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

    post_start_contexts = [call.get("context", "") for call in fake_asr.init_calls[1:]]
    assert len(post_start_contexts) >= 2
    assert set(post_start_contexts) == {"Elisha Jordan"}


def test_ws_later_start_omission_resets_session_override_to_schedule(tmp_path):
    fake_asr = _FakeASR()
    app = _create_app(_context_enabled_args(tmp_path), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": ["Elisha"],
            }
        )
        _receive_until_type(ws, "started")
        ws.send_json({"type": "start", "language": "English"})
        _receive_until_type(ws, "started")

    assert fake_asr.init_calls[-2]["context"] == "Elisha"
    assert fake_asr.init_calls[-1]["context"] == "ScheduledTerm"


def test_ws_invalid_context_rejects_start_without_replacing_live_state_or_leaking_text(
    tmp_path,
):
    secret = "SECRET_CONTEXT_TERM."
    fake_asr = _FakeASR()
    args = _context_enabled_args(tmp_path)
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "context-trace.jsonl")
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": ["Elisha"],
            }
        )
        _receive_until_type(ws, "started")
        init_count = len(fake_asr.init_calls)

        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": [secret],
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "sentence punctuation" in error["message"]
        assert secret not in error["message"]
        assert len(fake_asr.init_calls) == init_count

    trace_text = Path(args.subtitle_trace_log_file).read_text(encoding="utf-8")
    assert secret not in trace_text
    failed_rows = [
        json.loads(line)
        for line in trace_text.splitlines()
        if json.loads(line).get("event") == "start_failed"
    ]
    assert failed_rows
    assert "error_type" in failed_rows[-1]
    assert "error_sha256" in failed_rows[-1]
    assert "error" not in failed_rows[-1]


@pytest.mark.parametrize(
    ("bad_terms", "max_terms", "max_chars", "message"),
    [
        ("Elisha", 24, 160, "list of strings"),
        (["Elisha", 7], 24, 160, "term 1 must be a string"),
        (["Moses", "Aaron", "Jordan"], 2, 160, "at most 2 terms"),
        (["LongTerm", "Aaron"], 24, 8, "at most 8 characters"),
    ],
)
def test_ws_invalid_context_shape_or_limit_does_not_replace_state(
    tmp_path,
    bad_terms,
    max_terms,
    max_chars,
    message,
):
    fake_asr = _FakeASR()
    args = _context_enabled_args(tmp_path)
    args.asr_context_max_terms = max_terms
    args.asr_context_max_chars = max_chars
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": [],
            }
        )
        _receive_until_type(ws, "started")
        init_count = len(fake_asr.init_calls)

        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": bad_terms,
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert message in error["message"]
    assert len(fake_asr.init_calls) == init_count


def test_ws_session_context_is_isolated_between_connections(tmp_path):
    fake_asr = _FakeASR()
    app = _create_app(_context_enabled_args(tmp_path), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": ["Elisha"],
            }
        )
        _receive_until_type(ws, "started")
    first_context = fake_asr.init_calls[-1]["context"]

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "language": "English"})
        _receive_until_type(ws, "started")
    second_context = fake_asr.init_calls[-1]["context"]

    assert first_context == "Elisha"
    assert second_context == "ScheduledTerm"


def test_ws_selects_context_by_consumed_audio_time_after_hard_cut(tmp_path):
    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [
                    {"start_sec": 0.0, "end_sec": 0.3, "terms": ["暗兰"]},
                    {"start_sec": 0.3, "end_sec": 10.0, "terms": ["利未支派"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake_asr = _FakeASR()
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_max_terms = 24
    args.asr_context_max_chars = 160
    args.asr_context_lookaround_sec = 0.0
    args.asr_context_apply_mode = "streaming"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

    contexts = [str(call.get("context", "")) for call in fake_asr.init_calls]
    assert contexts[0] == "出埃及记 暗兰"
    assert "出埃及记 利未支派" in contexts[1:]


def test_ws_filters_streaming_context_glossary_echo_before_partial_output(tmp_path):
    context_terms = ["南区", "服侍", "属灵", "尼希米"]
    context = " ".join(context_terms)

    class ContextEchoASR(_FakeASR):
        def _echo_text(self):
            return (
                "The genealogy introduces several families. "
                f"{context}. "
                "Moses later returned to Egypt."
            )

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            self.last_state = state
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "English"
            state.text = self._echo_text()
            state._raw_decoded = state.text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.text = self._echo_text()
            state._raw_decoded = state.text
            return state

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": context_terms,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = ContextEchoASR()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "streaming"
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 320, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        partial = _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        final = _receive_until_type(ws, "final")

    assert context not in partial["text"]
    assert context not in final["text"]
    assert "The genealogy introduces several families." in partial["text"]
    assert "Moses later returned to Egypt." in partial["text"]
    assert fake_asr.last_state._raw_decoded == partial["text"]


def test_ws_keeps_spoken_context_terms_when_they_grow_incrementally(tmp_path):
    context_terms = ["Reuben", "Saul", "Canaanite", "Amram", "Jochebed", "Aaron", "Moses"]
    context = " ".join(context_terms)
    prefix = (
        "The reading begins here with a long explanation of the historical setting. "
        "Another introductory sentence gives the audience additional background."
    )
    previous = prefix + " " + " ".join(context_terms[:5])
    spoken = prefix + " " + context + "."

    class IncrementalContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_count = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            self.last_state = state
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            self.decode_count += 1
            state.language = "English"
            state.text = previous if self.decode_count == 1 else spoken
            state._raw_decoded = state.text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": context_terms,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = IncrementalContextASR()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "streaming"
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 320, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_bytes(raw)
        deadline = time.time() + 2.0
        while fake_asr.decode_count < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert fake_asr.decode_count >= 2
        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final")

    assert fake_asr.last_state._raw_decoded == spoken


def test_ws_removes_consecutive_context_echo_but_preserves_real_sentence_text(
    tmp_path,
):
    context_terms = ["Alpha", "Beta", "Gamma", "Delta"]
    contaminated = (
        "This uncertain sentence contains enough unrelated words before Alpha Beta Gamma."
    )
    preserved = "This uncertain sentence contains enough unrelated words before."
    natural = "This normal sentence remains available for translation and speech output."

    class ConsecutiveContextRunASR(_FakeASR):
        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            state.context = str(kwargs.get("context", "") or "")
            state._raw_decoded = ""
            return state

        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "English"
            state.text = f"{contaminated} {natural}"
            state._raw_decoded = state.text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    trace_path = tmp_path / "streaming-context-commit-gate.jsonl"
    fake_asr = ConsecutiveContextRunASR()
    translator = _FakeTranslator()
    args = _args()
    args.force_language = "English"
    args.asr_context_apply_mode = "streaming"
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)
    app = _create_app(
        args,
        fake_asr,
        translator=translator,
        tts_synthesizer=_FakeTTSSynthesizer(),
    )

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "English",
                "asr_context_terms": context_terms,
                "tts_enabled": True,
                "tts_client_id": "context-gate-client",
            }
        )
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1200, -1200] * 320, dtype="<i2").tobytes())
        _receive_until_type(ws, "partial")
        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    committed = [event["text"] for event in events if event.get("type") == "sentence_committed"]
    translated_sources = [call[0] for call in translator.calls]
    tts_jobs = [event for event in events if event.get("type") == "tts_job"]
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert fake_asr.init_calls[-1]["context"] == " ".join(context_terms)
    assert contaminated not in committed
    assert contaminated not in translated_sources
    assert preserved in committed
    assert preserved in translated_sources
    assert natural in committed
    assert natural in translated_sources
    assert len(tts_jobs) == 2
    assert any(row.get("event") == "context_run_commit_trimmed" for row in trace_rows)


def test_ws_segment_final_mode_applies_context_once_to_complete_segment(tmp_path):
    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [
                    {"start_sec": 0.0, "end_sec": 10.0, "terms": ["暗兰"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake_asr = _FakeASR()
    fake_asr.sampling_params = SimpleNamespace(max_tokens=32)
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_max_terms = 24
    args.asr_context_max_chars = 160
    args.asr_context_lookaround_sec = 0.0
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        deadline = time.time() + 2.0
        while not fake_asr.transcribe_calls and time.time() < deadline:
            time.sleep(0.02)

    assert fake_asr.init_calls
    assert all(str(call.get("context", "")) == "" for call in fake_asr.init_calls)
    assert fake_asr.transcribe_calls
    assert fake_asr.transcribe_calls[0]["context"] == "出埃及记 暗兰"
    assert fake_asr.transcribe_calls[0]["language"] == "Chinese"
    assert fake_asr.transcribe_calls[0]["sampling_max_tokens"] == 512
    assert fake_asr.sampling_params.max_tokens == 32
    segment_audio = fake_asr.transcribe_calls[0]["audio"][0][0]
    assert isinstance(segment_audio, np.ndarray)
    assert segment_audio.size > 0


def test_segment_context_redecode_rejects_unsubstantiated_glossary_fragment(tmp_path):
    context_terms = ["尼希米", "城墙", "羊门", "粪门", "祭司", "圣经"]
    context_fragment = "所以说，城墙、羊门、粪门。"
    natural_text = "让我们来赞美他，让我们来敬拜他。"

    class FragmentEchoASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "Chinese"
            state.text = natural_text
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            state.text = natural_text
            return state

        def transcribe(self, audio, context="", language=None):
            self.transcribe_text = context_fragment if context else natural_text
            self.transcribe_language = "Chinese"
            return super().transcribe(audio, context=context, language=language)

    trace_path = tmp_path / "segment-context-fragment.jsonl"
    fake_asr = FragmentEchoASR()
    fake_asr.sampling_params = SimpleNamespace(max_tokens=32)
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(trace_path)

    with TestClient(_create_app(args, fake_asr)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "start",
                "language": "Chinese",
                "asr_context_terms": context_terms,
            }
        )
        _receive_until_type(ws, "started")
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        assert _receive_until_type(ws, "partial")["text"] == natural_text
        time.sleep(1.1)
        ws.send_bytes(raw)
        deadline = time.time() + 2.0
        while not fake_asr.transcribe_calls and time.time() < deadline:
            time.sleep(0.02)
        assert fake_asr.transcribe_calls

        ws.send_json({"type": "finish", "mode": "stop"})
        events = _collect_through_final(ws)

    assert all(context_fragment not in str(event.get("text", "") or "") for event in events)
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rejected = [
        row
        for row in rows
        if row.get("event") == "asr_context_final_redecode_skipped"
        and row.get("reason") == "context_fragment_echo"
    ]
    assert rejected


def test_ws_keeps_stop_redecode_when_schedule_does_not_match_language(tmp_path):
    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake_asr = _FakeASR()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final")

    assert fake_asr.transcribe_calls
    assert fake_asr.transcribe_calls[-1]["context"] == ""
    assert fake_asr.transcribe_calls[-1]["language"] == "English"


def test_segment_context_lexical_correction_updates_sentence_and_translation(tmp_path):
    class LexicalCorrectionASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "English"
            state.text = "Amron went home. Another sentence. tail"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            state.text = "Amron went home. Another sentence. tail"
            return state

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": ["Amram"],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = LexicalCorrectionASR()
    fake_asr.transcribe_language = "English"
    fake_asr.transcribe_text = "Amram went home. Another sentence. tail"
    fake_translator = _FakeTranslator()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr, fake_translator)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_bytes(raw)
        committed = _receive_until_type(ws, "sentence_committed")
        assert committed["text"] == "Amron went home."
        time.sleep(1.1)
        ws.send_bytes(raw)
        updated = _receive_until_type(ws, "sentence_updated", max_steps=80)

        assert updated["sentence_id"] == committed["sentence_id"]
        assert updated["revision"] > committed["revision"]
        assert updated["text"] == "Amram went home."
        translated = None
        for _ in range(80):
            event = ws.receive_json()
            if (
                event.get("type") == "sentence_translation"
                and event.get("sentence_id") == committed["sentence_id"]
                and event.get("revision") == updated["revision"]
            ):
                translated = event
                break
        assert translated is not None
        assert "Amram went home." in translated["translation"]

    assert any(call[0] == "Amram went home." for call in fake_translator.calls)


def test_segment_context_correction_does_not_recommit_covered_sentences(tmp_path):
    first = "The family record is simple and it contains many ancestral names."
    repeated = "Therefore it became the complete genealogy of our family."
    tail = "The next explanation continues with another important detail."

    class ResegmentedCorrectionASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "English"
            state.text = f"{first} {repeated} {tail}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": ["genealogy"],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = ResegmentedCorrectionASR()
    fake_asr.transcribe_language = "English"
    fake_asr.transcribe_text = (
        "The family record is simple. "
        "It contains many ancestral names. "
        f"{repeated} {tail}"
    )
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)
    committed_texts = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_bytes(raw)
        while len(committed_texts) < 2:
            event = ws.receive_json()
            if event.get("type") == "sentence_committed":
                committed_texts.append(str(event.get("text") or ""))
        time.sleep(1.1)
        ws.send_bytes(raw)
        deadline = time.time() + 2.0
        while not fake_asr.transcribe_calls and time.time() < deadline:
            time.sleep(0.02)
        assert fake_asr.transcribe_calls
        ws.send_text('{"type":"finish"}')
        for _ in range(120):
            event = ws.receive_json()
            if event.get("type") == "sentence_committed":
                committed_texts.append(str(event.get("text") or ""))
            if event.get("type") == "final":
                break
        else:
            pytest.fail("did not receive final")

    assert committed_texts.count(repeated) == 1


def test_new_start_resets_segment_context_application_state(tmp_path):
    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake_asr = _FakeASR()
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        deadline = time.time() + 2.0
        while not fake_asr.transcribe_calls and time.time() < deadline:
            time.sleep(0.02)
        assert fake_asr.transcribe_calls

        calls_before_restart = len(fake_asr.transcribe_calls)
        ws.send_text('{"type":"start","language":"English"}')
        _receive_until_type(ws, "started")
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final")

    restarted_calls = fake_asr.transcribe_calls[calls_before_restart:]
    assert restarted_calls
    assert restarted_calls[-1]["context"] == ""
    assert restarted_calls[-1]["language"] == "English"


def test_stale_context_correction_cannot_mark_restarted_session(tmp_path):
    class BlockingContextASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.context_started = threading.Event()
            self.context_release = threading.Event()

        def transcribe(self, audio, context="", language=None):
            if context:
                self.context_started.set()
                if not self.context_release.wait(timeout=5.0):
                    raise TimeoutError("context correction was not released")
            return super().transcribe(audio, context=context, language=language)

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake_asr = BlockingContextASR()
    args = _args()
    args.force_language = "Chinese"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        assert fake_asr.context_started.wait(timeout=2.5)

        ws.send_text('{"type":"start","language":"English"}')
        _receive_until_type(ws, "started")
        calls_before_restart = len(fake_asr.transcribe_calls)
        fake_asr.context_release.set()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final", max_steps=80)

    restarted_calls = fake_asr.transcribe_calls[calls_before_restart:]
    assert restarted_calls
    assert restarted_calls[-1]["context"] == ""
    assert restarted_calls[-1]["language"] == "English"


def test_stop_segment_context_correction_is_not_overwritten_by_full_redecode(tmp_path):
    class ContextAwareASR(_FakeASR):
        def transcribe(self, audio, context="", language=None):
            self.transcribe_text = "Amram went home." if context else "context-free overwrite"
            self.transcribe_language = "English"
            return super().transcribe(audio, context=context, language=language)

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": ["Amram"],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = ContextAwareASR()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        final = _receive_until_type(ws, "final")

    assert final["text"] == "Amram went home."
    assert [call["context"] for call in fake_asr.transcribe_calls] == ["Amram"]


def test_unchanged_stop_context_result_keeps_context_free_final_redecode(tmp_path):
    sentence = "The family record remains unchanged."

    class NoOpContextASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            state.audio_accum = np.concatenate(
                [state.audio_accum, np.asarray(wav, dtype=np.float32)]
            )
            state.language = "English"
            state.text = sentence
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

        def transcribe(self, audio, context="", language=None):
            self.transcribe_text = (
                "The family record remains unchanged!"
                if context
                else "Context-free final result."
            )
            self.transcribe_language = "English"
            return super().transcribe(audio, context=context, language=language)

    schedule_path = tmp_path / "context.json"
    schedule_path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "English",
                "global_terms": ["genealogy"],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    fake_asr = NoOpContextASR()
    args = _args()
    args.force_language = "English"
    args.asr_context_schedule = str(schedule_path)
    args.asr_context_apply_mode = "segment_final"
    args.final_redecode_on_stop = True
    args.final_redecode_max_sec = 30.0
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        final = _receive_until_type(ws, "final")

    assert final["text"] == "Context-free final result."
    assert [call["context"] for call in fake_asr.transcribe_calls] == ["genealogy", ""]


def test_ws_start_message_overrides_force_language():
    fake_asr = _FakeASR()
    app = _create_app(_args(), fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready

        ws.send_text('{"type":"start","language":"English"}')
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["language"] == "English"

        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        partial = ws.receive_json()
        assert partial["type"] == "partial"
        assert fake_asr.init_calls[-1]["language"] == "English"


def test_http_index_not_blocked_by_vllm_streaming_call():
    started = threading.Event()
    release = threading.Event()
    ws_done = threading.Event()
    get_done = threading.Event()
    get_result = {}
    ws_errors = []

    class _BlockingASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            started.set()
            release.wait(timeout=3.0)
            return super().streaming_transcribe(wav, state)

    app = _create_app(_args(), _BlockingASR())
    client = TestClient(app)

    def _ws_worker():
        try:
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # ready
                raw = np.array([0, 1000, -1000] * 700, dtype="<i2").tobytes()
                ws.send_bytes(raw)
                ws.receive_json()  # partial
                ws_done.set()
        except Exception as e:
            ws_errors.append(e)
            ws_done.set()

    def _get_worker():
        t0 = time.perf_counter()
        resp = client.get("/")
        get_result["status_code"] = resp.status_code
        get_result["elapsed"] = time.perf_counter() - t0
        get_done.set()

    ws_thread = threading.Thread(target=_ws_worker, daemon=True)
    get_thread = threading.Thread(target=_get_worker, daemon=True)
    ws_thread.start()
    assert started.wait(timeout=2.0), "streaming_transcribe was not called"
    get_thread.start()

    try:
        assert get_done.wait(timeout=0.6), "GET / should remain responsive during streaming inference"
        assert get_result["status_code"] == 200
    finally:
        release.set()
        ws_thread.join(timeout=3.0)
        get_thread.join(timeout=3.0)

    assert not ws_errors


def test_audio_queue_spills_small_frames_without_dropping_audio():
    class BlockingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_started = threading.Event()
            self.decode_release = threading.Event()
            self.final_audio_samples = 0

        def streaming_transcribe(self, wav, state):
            if not self.decode_started.is_set():
                self.decode_started.set()
                if not self.decode_release.wait(timeout=5.0):
                    raise TimeoutError("streaming decode was not released")
            return super().streaming_transcribe(wav, state)

        def finish_streaming_transcribe(self, state):
            result = super().finish_streaming_transcribe(state)
            self.final_audio_samples = int(state.audio_accum.size)
            return result

    fake_asr = BlockingASR()
    args = _args()
    args.audio_queue_size = 2
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)
    frame = np.array([0, 1200, -1200] * 320, dtype="<i2").tobytes()
    frame_samples = len(frame) // 2
    frame_count = 11

    try:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_bytes(frame)
            assert fake_asr.decode_started.wait(timeout=2.0)
            for _ in range(frame_count - 1):
                ws.send_bytes(frame)
            ws.send_text('{"type":"finish"}')
            time.sleep(0.1)
            fake_asr.decode_release.set()
            _receive_until_type(ws, "final", max_steps=80)
    finally:
        fake_asr.decode_release.set()

    assert fake_asr.final_audio_samples == frame_count * frame_samples


def test_audio_queue_hard_pressure_preserves_pcm_until_consumer_recovers():
    class BlockingASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.decode_started = threading.Event()
            self.decode_release = threading.Event()
            self.final_audio_samples = 0

        def streaming_transcribe(self, wav, state):
            if not self.decode_started.is_set():
                self.decode_started.set()
                if not self.decode_release.wait(timeout=5.0):
                    raise TimeoutError("streaming decode was not released")
            return super().streaming_transcribe(wav, state)

        def finish_streaming_transcribe(self, state):
            result = super().finish_streaming_transcribe(state)
            self.final_audio_samples = int(state.audio_accum.size)
            return result

    fake_asr = BlockingASR()
    args = _args()
    args.audio_queue_size = 2
    args.backpressure_target_queue_sec = 0.1
    args.backpressure_max_queue_sec = 0.2
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)
    frame = np.array([0, 1200, -1200] * 320, dtype="<i2").tobytes()
    frame_samples = len(frame) // 2
    frame_count = 11

    try:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_bytes(frame)
            assert fake_asr.decode_started.wait(timeout=2.0)
            for _ in range(frame_count - 1):
                ws.send_bytes(frame)
            fake_asr.decode_release.set()
            ws.send_text('{"type":"finish"}')
            _receive_until_type(ws, "final", max_steps=80)
    finally:
        fake_asr.decode_release.set()

    assert fake_asr.final_audio_samples == frame_count * frame_samples


def test_ws_does_not_leak_active_connections_when_init_fails():
    class _FailOnceASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def init_streaming_state(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("init boom")
            return super().init_streaming_state(**kwargs)

    args = _args()
    args.max_connections = 1
    app = _create_app(args, _FailOnceASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "init boom" in msg["message"]

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"


def test_ws_disconnect_does_not_finalize_when_disabled():
    fake_asr = _FakeASR()
    args = _args()
    args.finalize_on_disconnect = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        ws.receive_json()  # partial
        # exit context without finish

    assert fake_asr.finish_calls == 0


def test_ws_defers_commit_of_newest_completed_sentence_until_next_sentence_arrives():
    class _SeqASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "Chinese"
            if self.calls == 1:
                state.text = "这是一个非常非常长的第一句内容。"
            else:
                state.text = "这是一个非常非常长的第一句内容。这是第二句也足够长并且完整。"
            return state

    app = _create_app(_args(), _SeqASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        ws.send_bytes(raw)
        first_partial = ws.receive_json()
        assert first_partial["type"] == "partial"
        assert first_partial["text"] == "这是一个非常非常长的第一句内容。"

        ws.send_bytes(raw)
        committed = ws.receive_json()
        assert committed["type"] == "sentence_committed"
        assert committed["text"] == "这是一个非常非常长的第一句内容。"

        second_partial = ws.receive_json()
        assert second_partial["type"] == "partial"
        assert second_partial["text"] == "这是一个非常非常长的第一句内容。这是第二句也足够长并且完整。"


def test_ws_keeps_last_completed_sentence_as_tentative_until_next_sentence():
    class _GrowingSecondSentenceASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.s1 = "这是第一句已经稳定完成并且长度足够。"
            self.s2_v1 = "这是第二句最开始版本。"
            self.s2_v2 = "这是第二句最开始版本继续补充更多内容。"

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "Chinese"
            if self.calls == 1:
                state.text = f"{self.s1}{self.s2_v1}"
            else:
                state.text = f"{self.s1}{self.s2_v2}"
            return state

    app = _create_app(_args(), _GrowingSecondSentenceASR())
    client = TestClient(app)

    committed = []
    partials = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(3):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                if msg["type"] == "sentence_committed":
                    committed.append(msg["text"])
                    continue
                if msg["type"] == "partial":
                    partials.append(msg)
                    break

    assert committed == ["这是第一句已经稳定完成并且长度足够。"]
    assert partials[-1]["tentative_text"] == "这是第二句最开始版本继续补充更多内容。"
    assert partials[-1]["is_stable"] is False
    assert partials[-1]["stability"]["phase"] == "generating"
    assert partials[-1]["stability"]["is_stable"] is False
    assert partials[-1]["stability"]["unstable_chars"] == len("这是第二句最开始版本继续补充更多内容。")


def test_ws_sentence_events_explicitly_mark_solidified_text_as_stable():
    class _TwoSentenceASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = "这是第一句已经稳定完成并且长度足够。这是第二句正在生成"
            return state

    app = _create_app(_args(), _TwoSentenceASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        committed = _receive_until_type(ws, "sentence_committed")
        assert committed["type"] == "sentence_committed"
        assert committed["is_stable"] is True
        assert committed["stability"]["phase"] == "solidified"
        assert committed["stability"]["is_stable"] is True
        assert committed["stability"]["reason"] == "sentence_committed"


def test_ws_keeps_stable_terminal_hypothesis_tentative_without_rollback_safe_lookahead():
    class _StableSingleSentenceASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = "这是一个已经稳定完成并且长度足够的句子。"
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    translator = _FakeTranslator()
    app = _create_app(args, _StableSingleSentenceASR(), translator=translator)
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        for _ in range(4):
            ws.send_bytes(raw)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    assert message["tentative_text"] == "这是一个已经稳定完成并且长度足够的句子。"
                    break

    assert [message for message in events if message.get("type") == "sentence_committed"] == []
    assert [message for message in events if message.get("type") == "sentence_translation"] == []
    assert translator.calls == []


def test_ws_early_translates_sentence_after_rollback_safe_lookahead(tmp_path):
    sentence = "这是一个已经稳定完成并且长度足够的句子。"
    lookahead = "后续内容正在生成"

    class _StableSentenceWithLookaheadASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = f"{sentence}{lookahead}"
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "rollback-safe-early-commit.jsonl")
    translator = _FakeTranslator()
    app = _create_app(args, _StableSentenceWithLookaheadASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(raw)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

    commits = [message for message in events if message.get("type") == "sentence_committed"]
    assert [message.get("text") for message in commits] == [sentence]
    assert [call[0] for call in translator.calls] == [sentence]
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    promoted = [row for row in trace_rows if row.get("event") == "early_translation_stable_commit"]
    assert len(promoted) == 1
    assert promoted[0]["rollback_safe"] is True
    assert int(promoted[0]["lookahead_tokens"]) >= int(promoted[0]["required_lookahead_tokens"])


def test_ws_commits_long_comma_text_as_rollback_safe_clauses_without_cutting_asr(tmp_path):
    text = (
        "一个在南边的家，一个在 P C C 的教会，另外一个家在 P C C O 的家，"
        "都是神托付给我看管的家，我有责任看顾家人的灵命成长，"
        "我也有责任看顾两个儿子与神的关系，但是最重要的是，"
        "我在教会中也有责任看管神所托付给我的羊，我也需要帮助牧者，"
        "我也需要分担他们的责任，所以刚开始必须建立好自己属灵的家"
    )

    class _LongCommaASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = text
            return state

    args = _args()
    args.stable_clause_target_cjk_chars = 32
    args.stable_clause_target_latin_words = 24
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "stable-clause.jsonl")
    translator = _FakeTranslator()
    app = _create_app(args, _LongCommaASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(raw)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

    commits = [message for message in events if message.get("type") == "sentence_committed"]
    assert len(commits) >= 2
    assert all(message["text"].endswith(("，", "；", "：")) for message in commits)
    assert all(message.get("boundary_kind") == "stable_clause" for message in commits)
    assert [call[0] for call in translator.calls] == [message["text"] for message in commits]

    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    committed_rows = [row for row in trace_rows if row.get("event") == "sentence_new_commit"]
    assert committed_rows
    assert all(row.get("rollback_safe") is True for row in committed_rows)
    assert all(row.get("boundary_kind") == "stable_clause" for row in committed_rows)
    assert not [row for row in trace_rows if row.get("event") == "segment_cut_decision"]


def test_ws_holds_stable_clause_until_following_text_exits_rollback_window(tmp_path):
    clause = f"{'甲' * 32}，"
    short_following = "后续。"
    safe_following = "后续内容已经足够。"

    class _GrowingLookaheadASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            self.calls += 1
            state.language = "Chinese"
            following = short_following if self.calls <= 3 else safe_following
            state.text = f"{clause}{following}"
            return state

    args = _args()
    args.stable_clause_target_cjk_chars = 32
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "stable-clause-lookahead.jsonl")
    translator = _FakeTranslator()
    app = _create_app(args, _GrowingLookaheadASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for call_no in range(5):
            ws.send_bytes(raw)
            while True:
                message = ws.receive_json()
                events.append((call_no, message))
                if message.get("type") == "partial":
                    break

    commits = [
        (call_no, message)
        for call_no, message in events
        if message.get("type") == "sentence_committed"
    ]
    assert commits
    assert commits[0][0] >= 3
    assert commits[0][1]["text"] == clause
    assert [call[0] for call in translator.calls] == [clause]

    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    waits = [row for row in trace_rows if row.get("event") == "stable_clause_rollback_wait"]
    assert waits
    assert all(int(row.get("lookahead_tokens", 0)) < 5 for row in waits)
    committed = [row for row in trace_rows if row.get("event") == "sentence_new_commit"]
    assert committed[0]["rollback_safe"] is True


def test_ws_does_not_early_translate_sentence_inside_rollback_window():
    sentence = "这是一个已经稳定完成并且长度足够的句子。"
    lookahead = "后续"

    class _SentenceInsideRollbackWindowASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = f"{sentence}{lookahead}"
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    translator = _FakeTranslator()
    app = _create_app(args, _SentenceInsideRollbackWindowASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(4):
            ws.send_bytes(raw)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

    assert [message for message in events if message.get("type") == "sentence_committed"] == []
    assert translator.calls == []


def test_ws_does_not_early_commit_short_english_completed_sentence():
    class _ShortEnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = "The Short Session Topic."
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    app = _create_app(args, _ShortEnglishASR())
    client = TestClient(app)

    seen = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        ws.send_bytes(raw)
        first_partial = _receive_until_type(ws, "partial")
        assert first_partial["tentative_text"] == "The Short Session Topic."

        ws.send_bytes(raw)
        for _ in range(8):
            msg = ws.receive_json()
            seen.append(msg)
            if msg.get("type") == "partial":
                break

    assert [msg.get("type") for msg in seen] == ["partial"]
    assert seen[-1]["tentative_text"] == "The Short Session Topic."


def test_ws_early_translates_stable_short_english_after_rollback_safe_lookahead(tmp_path):
    sentence = "The Short Session Topic."
    lookahead = " Follow up"

    class _ShortEnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = f"{sentence}{lookahead}"
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.early_translation_short_stable_sec = 0.0
    args.early_translation_short_stable_hits = 4
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "early-short-english.jsonl")
    translator = _FakeTranslator()
    app = _create_app(args, _ShortEnglishASR(), translator=translator)
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_text('{"type":"set_translation_direction","translation_direction":"en2zh"}')
        _receive_until_type(ws, "translation_direction")
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        for _ in range(4):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break

        time.sleep(0.05)
        ws.send_bytes(raw)
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "partial":
                break

    commits = [msg for msg in events if msg.get("type") == "sentence_committed"]
    translations = [msg for msg in events if msg.get("type") == "sentence_translation"]
    assert [msg.get("text") for msg in commits] == [sentence]
    assert len(translations) == 1
    assert translations[0]["translation"].startswith("[English->Chinese]")
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    waits = [row for row in trace_rows if row.get("event") == "early_translation_stability_wait"]
    promoted = [row for row in trace_rows if row.get("event") == "early_translation_stable_commit"]
    assert waits
    assert len(promoted) == 1
    assert promoted[0]["short_english"] is True
    assert int(promoted[0]["required_hits"]) == 4
    assert int(promoted[0]["stable_hits"]) >= 4
    assert int(promoted[0]["terminal_first_seen_ms"]) > 0


def test_ws_does_not_early_commit_incomplete_short_english_tail():
    tail = "The Short Session Topic"

    class _IncompleteEnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = tail
            return state

    args = _args()
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 1
    args.early_translation_short_stable_sec = 0.0
    args.early_translation_short_stable_hits = 1
    translator = _FakeTranslator()
    app = _create_app(args, _IncompleteEnglishASR(), translator=translator)
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(4):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    assert msg.get("tentative_text") == tail
                    break

    assert [msg for msg in events if msg.get("type") == "sentence_committed"] == []
    assert [msg for msg in events if msg.get("type") == "sentence_translation"] == []
    assert translator.calls == []


def test_ws_segment_finalize_does_not_slice_commit_short_english_completed_sentence():
    class _ShortEnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = "The Short Session Topic."
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            return state

    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_hits = 99
    app = _create_app(args, _ShortEnglishASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000] * 2400, dtype="<i2").tobytes()

        ws.send_bytes(raw)
        events.append(_receive_until_type(ws, "partial"))

        time.sleep(1.1)
        ws.send_bytes(raw)
        ws.send_text('{"type":"finish"}')
        for _ in range(80):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                break

    slice_commits = [
        msg
        for msg in events
        if msg.get("type") == "sentence_committed" and bool(msg.get("slice_commit"))
    ]
    assert slice_commits == []
    final_commits = [msg for msg in events if msg.get("type") == "sentence_committed"]
    assert any(msg.get("text") == "The Short Session Topic." for msg in final_commits)


def test_ws_segment_finalize_holds_short_english_fragment_after_long_sentence():
    first_sentence = "A completed English sentence with enough words ends before a segment cut."
    short_fragment = "Short fragment."
    grown_sentence = "Short fragment later grows into a complete sentence with enough words."

    class _ShortFragmentAfterSentenceASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self._segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self._segment_no += 1
            state.segment_no = self._segment_no
            return state

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            if int(getattr(state, "segment_no", 1)) == 1:
                state.text = f"{first_sentence} {short_fragment}"
            else:
                state.text = grown_sentence
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = state.language or "English"
            return state

    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 99
    app = _create_app(args, _ShortFragmentAfterSentenceASR())
    client = TestClient(app)

    def receive_until_type(expected_type: str, max_steps: int = 40):
        seen = []
        for _ in range(max_steps):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == expected_type:
                return msg
            seen.append(msg.get("type"))
        pytest.fail(f"did not receive {expected_type}, seen={seen}")

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

        ws.send_bytes(raw)
        receive_until_type("partial")

        time.sleep(1.1)
        ws.send_bytes(raw)
        receive_until_type("partial")

        ws.send_bytes(raw)
        receive_until_type("partial")

        ws.send_text('{"type":"finish"}')
        final_msg = None
        for _ in range(100):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                final_msg = msg
                break

    assert final_msg is not None
    committed = [str(msg.get("text", "")).strip() for msg in events if msg.get("type") == "sentence_committed"]
    assert first_sentence in committed
    assert short_fragment not in committed
    assert grown_sentence in committed
    assert grown_sentence in str(final_msg.get("committed_text", ""))


def test_ws_emits_sentence_updated_when_committed_sentence_grows():
    class _UpdateASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.s1 = "这是第一句已经稳定完成并且长度足够。"
            self.s2_short = "这是第二句初版已经成句并且长度足够。"
            self.s2_long = "这是第二句初版已经成句并且长度足够继续补充更多更完整的内容。"
            self.s3 = "这是第三句已经稳定完成并且长度足够。"

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "Chinese"
            if self.calls <= 2:
                state.text = f"{self.s1}{self.s2_short}{self.s3}"
            else:
                state.text = f"{self.s1}{self.s2_long}{self.s3}"
            return state

    app = _create_app(_args(), _UpdateASR())
    client = TestClient(app)

    committed = []
    updated = []
    partials = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(3):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                msg_type = msg["type"]
                if msg_type == "sentence_committed":
                    committed.append(msg["text"])
                    continue
                if msg_type == "sentence_updated":
                    updated.append(msg["text"])
                    continue
                if msg_type == "partial":
                    partials.append(msg)
                    break

    assert "这是第二句初版已经成句并且长度足够。" in committed
    assert "这是第二句初版已经成句并且长度足够继续补充更多更完整的内容。" in updated
    assert partials[-1]["tentative_text"] == "这是第三句已经稳定完成并且长度足够。"


def test_ws_promotes_only_stable_small_sentence_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_streaming_ws, "_SMALL_UPGRADE_REQUIRED_HITS", 3)
    monkeypatch.setattr(demo_streaming_ws, "_SMALL_UPGRADE_STABLE_SEC", 0.0)

    class _SmallTailASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.intro = "Opening sentence is complete."
            self.old = "The result is ready."
            self.new = "The result is ready now."
            self.following = "Following sentence is complete."

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "English"
            middle = self.old if self.calls <= 2 else self.new
            state.text = f"{self.intro} {middle} {self.following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    asr = _SmallTailASR()
    args = _args()
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "small-tail-streaming.jsonl")
    translator = _FakeTranslator()
    app = _create_app(args, asr, translator=translator)
    events = []
    raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        for call_index in range(5):
            ws.send_bytes(raw)
            frame_events = []
            while True:
                msg = ws.receive_json()
                frame_events.append(msg)
                events.append(msg)
                if msg.get("type") == "partial":
                    break
            if call_index == 2:
                assert not any(msg.get("type") == "sentence_updated" for msg in frame_events)
        ws.send_text('{"type":"finish"}')
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                break

    committed = next(
        msg
        for msg in events
        if msg.get("type") == "sentence_committed" and msg.get("text") == asr.old
    )
    updated = next(
        msg
        for msg in events
        if msg.get("type") == "sentence_updated" and msg.get("text") == asr.new
    )
    assert updated["sentence_id"] == committed["sentence_id"]
    assert updated["revision"] == committed["revision"] + 1

    rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len([row for row in rows if row.get("event") == "sentence_upgrade_deferred"]) == 1
    small_commits = [row for row in rows if row.get("event") == "sentence_upgrade_small_commit"]
    assert len(small_commits) == 1
    assert small_commits[0]["source"] == "streaming_stable"
    assert small_commits[0]["hits"] == 3
    assert [call[0] for call in translator.calls].count(asr.new) == 1
    latest_translations = [
        msg
        for msg in events
        if msg.get("type") == "sentence_translation"
        and msg.get("sentence_id") == updated.get("sentence_id")
    ]
    assert latest_translations
    assert latest_translations[-1]["revision"] == updated["revision"]


def test_ws_rejects_retracted_small_sentence_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_streaming_ws, "_SMALL_UPGRADE_REQUIRED_HITS", 3)
    monkeypatch.setattr(demo_streaming_ws, "_SMALL_UPGRADE_STABLE_SEC", 0.0)

    class _RetractedTailASR(_FakeASR):
        intro = "Opening sentence is complete."
        old = "The result is ready."
        transient = "The result is ready so."
        stable = "The result is ready now."
        following = "Following sentence is complete."

        def __init__(self):
            super().__init__()
            self.calls = 0

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "English"
            if self.calls <= 2 or self.calls == 4:
                middle = self.old
            elif self.calls == 3:
                middle = self.transient
            else:
                middle = self.stable
            state.text = f"{self.intro} {middle} {self.following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            return state

    asr = _RetractedTailASR()
    translator = _FakeTranslator()
    args = _args()
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "small-tail-retraction.jsonl")
    app = _create_app(args, asr, translator=translator)
    events = []
    raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(7):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break
        ws.send_text('{"type":"finish"}')
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                break

    updates = [msg for msg in events if msg.get("type") == "sentence_updated"]
    assert asr.transient not in {msg.get("text") for msg in updates}
    stable_update = next(msg for msg in updates if msg.get("text") == asr.stable)
    rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len([row for row in rows if row.get("event") == "sentence_upgrade_rejected"]) == 1
    assert any(
        row.get("event") == "sentence_upgrade_candidate_reset"
        and row.get("reason") == "candidate_retracted_or_rewritten"
        for row in rows
    )
    assert [call[0] for call in translator.calls].count(asr.stable) == 1
    latest_translations = [
        msg
        for msg in events
        if msg.get("type") == "sentence_translation"
        and msg.get("sentence_id") == stable_update.get("sentence_id")
    ]
    assert latest_translations
    assert latest_translations[-1]["revision"] == stable_update["revision"]


def test_ws_final_stop_reconciles_small_monotonic_suffix():
    class _FinalTailASR(_FakeASR):
        intro = "Opening sentence is complete."
        old = "The result is ready."
        new = "The result is ready now."
        following = "Following sentence is complete."

        def streaming_transcribe(self, wav, state):
            state.language = "English"
            state.text = f"{self.intro} {self.old} {self.following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            state.text = f"{self.intro} {self.new} {self.following}"
            return state

    asr = _FinalTailASR()
    args = _args()
    args.final_redecode_on_stop = False
    app = _create_app(args, asr, translator=_FakeTranslator())
    events = []
    raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_bytes(raw)
            _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                final = msg
                break

    assert any(
        msg.get("type") == "sentence_updated" and msg.get("text") == asr.new
        for msg in events
    )
    assert asr.new in final["committed_text"]


def test_ws_hard_cut_reconciles_small_monotonic_suffix(tmp_path):
    class _HardCutFinalTailASR(_FakeASR):
        intro = "Opening sentence is complete."
        old = "The result is ready."
        new = "The result is ready now."
        following = "Following sentence is complete."

        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "English"
            if state.segment_no == 1:
                state.text = f"{self.intro} {self.old} {self.following}"
            else:
                state.text = "The next segment remains complete."
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            if state.segment_no == 1:
                state.text = f"{self.intro} {self.new} {self.following}"
            return state

    asr = _HardCutFinalTailASR()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "small-tail-hard-cut.jsonl")
    app = _create_app(args, asr, translator=_FakeTranslator())
    events = []
    raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_bytes(raw)
            _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(2):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break
        ws.send_text('{"type":"finish"}')
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                break

    assert any(
        msg.get("type") == "sentence_updated" and msg.get("text") == asr.new
        for msg in events
    )
    rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "sentence_upgrade_small_commit"
        and row.get("source") == "final_reconcile"
        and row.get("new_preview") == asr.new
        for row in rows
    )


def test_ws_updates_committed_sentence_after_model_corrects_its_tail():
    class _CorrectedTailUpdateASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.s1 = "The first sentence is stable and complete."
            self.s2_old = (
                "But I would have ultimately been okay with it because yeah, like when you take out "
                "a piece of paper and you start writing down, why do I want to."
            )
            self.s2_new = (
                "But I would have ultimately been okay with it because yeah, like when you take out "
                "a piece of paper and you start writing down, why do I want a third kid, and you're "
                "like, I don't even know, this is going to be a lot more work, and we're going to "
                "have to have a minivan forever and stuff like that."
            )
            self.s3 = "The third sentence is stable and complete."

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "English"
            second = self.s2_old if self.calls <= 2 else self.s2_new
            state.text = f"{self.s1} {second} {self.s3}"
            return state

    asr = _CorrectedTailUpdateASR()
    app = _create_app(_args(), asr)
    client = TestClient(app)

    committed = []
    updated = []
    latest_by_id = {}
    partials = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(4):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                if msg["type"] == "sentence_committed":
                    committed.append(msg["text"])
                    latest_by_id[str(msg.get("sentence_id", ""))] = str(msg["text"])
                    continue
                if msg["type"] == "sentence_updated":
                    updated.append(msg["text"])
                    latest_by_id[str(msg.get("sentence_id", ""))] = str(msg["text"])
                    continue
                if msg["type"] == "partial":
                    partials.append(msg)
                    break

    assert asr.s2_old.replace(" ", "") in "".join(committed).replace(" ", "")
    assert updated
    assert asr.s2_new.replace(" ", "") in "".join(latest_by_id.values()).replace(" ", "")
    assert partials[-1]["tentative_text"] == asr.s3


def test_ws_sentence_update_keeps_previous_translation_until_replacement_is_ready():
    class _UpdateASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.s1 = "First sentence is stable and complete."
            self.s2_short = "Second sentence starts as a complete long sentence."
            self.s2_long = "Second sentence starts as a complete long sentence and later receives important extra words."
            self.s3 = "Third sentence is stable and complete."

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "English"
            if self.calls <= 2:
                state.text = f"{self.s1} {self.s2_short} {self.s3}"
            else:
                state.text = f"{self.s1} {self.s2_long} {self.s3}"
            return state

    translator = _FakeTranslator()
    app = _create_app(_args(), _UpdateASR(), translator=translator)
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_text('{"type":"set_translation_direction","translation_direction":"en2zh"}')
        _receive_until_type(ws, "translation_direction")

        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(3):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg["type"] == "partial":
                    break

    updated_ids = [str(msg.get("sentence_id", "")) for msg in events if msg.get("type") == "sentence_updated"]
    assert updated_ids
    empty_updates = [
        msg
        for msg in events
        if msg.get("type") == "sentence_translation"
        and str(msg.get("sentence_id", "")) in updated_ids
        and not str(msg.get("translation", "")).strip()
    ]
    assert empty_updates == []


def test_ws_fast_translation_waits_for_latest_sentence_revision(tmp_path):
    asr = _FastRevisionASR()
    args = _args()
    args.final_redecode_on_stop = False
    args.translation_workers = 3
    args.tts_revision_stable_sec = 60.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "tts-revision-stability.jsonl")
    app = _create_app(
        args,
        asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "translation_direction": "en2zh"})
        _receive_until_type(ws, "started")
        frame = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        for _ in range(2):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

        committed = next(
            message
            for message in events
            if message.get("type") == "sentence_committed"
            and message.get("text") == asr.s2_short
        )
        sentence_id = str(committed["sentence_id"])
        revision_one = _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_translation"
                and str(message.get("sentence_id", "")) == sentence_id
                and int(message.get("revision", 0)) == 1
            ),
        )
        assert revision_one["translation"]
        assert not [
            event
            for event in _drain_listener_events(listener)
            if event.get("type") == "tts_job" and event.get("sentence_id") == sentence_id
        ]

        updated = None
        for _ in range(8):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if (
                    message.get("type") == "sentence_updated"
                    and str(message.get("sentence_id", "")) == sentence_id
                    and message.get("text") == asr.s2_long
                ):
                    updated = message
                if message.get("type") == "partial":
                    break
            if updated is not None:
                break
            time.sleep(0.1)
        assert updated is not None
        latest_revision = int(updated["revision"])
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_translation"
                and str(message.get("sentence_id", "")) == sentence_id
                and int(message.get("revision", 0)) == latest_revision
            ),
        )

        ws.send_json({"type": "finish", "mode": "stop"})
        _collect_through_final(ws)

    matching_jobs = [
        event
        for event in _drain_listener_events(listener)
        if event.get("type") == "tts_job" and event.get("sentence_id") == sentence_id
    ]
    assert len(matching_jobs) == 1
    assert int(matching_jobs[0]["revision"]) == latest_revision

    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_names = {row.get("event") for row in trace_rows}
    assert "tts_stability_wait" in event_names
    assert "tts_stability_reset" in event_names
    assert "tts_stability_release" in event_names
    private_events = {
        "tts_stability_wait",
        "tts_stability_reset",
        "tts_stability_release",
        "tts_late_revision_after_release",
    }
    for row in trace_rows:
        if row.get("event") not in private_events:
            continue
        assert "text" not in row
        assert "translation" not in row
        assert "sentence_id" not in row
        assert "job_id" not in row
        assert len(str(row.get("sentence_hash8", ""))) == 8


def test_ws_late_revision_after_tts_release_is_traced(tmp_path):
    asr = _FastRevisionASR()
    args = _args()
    args.final_redecode_on_stop = False
    args.translation_workers = 3
    args.tts_revision_stable_sec = 0.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "tts-late-revision.jsonl")
    app = _create_app(
        args,
        asr,
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    listener = app.state.tts_broadcast.register("anonymous")
    events = []

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "translation_direction": "en2zh"})
        _receive_until_type(ws, "started")
        frame = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break

        committed = next(
            message
            for message in events
            if message.get("type") == "sentence_committed"
            and message.get("text") == asr.s2_short
        )
        sentence_id = str(committed["sentence_id"])
        _poll_ws_with_ping(
            ws,
            events,
            lambda message: (
                message.get("type") == "sentence_translation"
                and str(message.get("sentence_id", "")) == sentence_id
                and int(message.get("revision", 0)) == 1
            ),
        )

        matching_jobs = []
        for _ in range(40):
            matching_jobs.extend(
                event
                for event in _drain_listener_events(listener)
                if event.get("type") == "tts_job" and event.get("sentence_id") == sentence_id
            )
            if matching_jobs:
                break
            _poll_ws_with_ping(ws, events, lambda message: message.get("type") == "pong", max_polls=1)
        assert len(matching_jobs) == 1
        assert int(matching_jobs[0]["revision"]) == 1

        updated = None
        for _ in range(8):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if (
                    message.get("type") == "sentence_updated"
                    and str(message.get("sentence_id", "")) == sentence_id
                    and message.get("text") == asr.s2_long
                ):
                    updated = message
                if message.get("type") == "partial":
                    break
            if updated is not None:
                break
            time.sleep(0.1)
        assert updated is not None

    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    late_rows = [
        row for row in trace_rows if row.get("event") == "tts_late_revision_after_release"
    ]
    assert len(late_rows) == 1
    assert late_rows[0]["released_revision"] == 1
    assert late_rows[0]["incoming_revision"] == int(updated["revision"])
    assert late_rows[0]["elapsed_since_release_ms"] >= 0
    assert len(str(late_rows[0]["sentence_hash8"])) == 8
    assert "sentence_id" not in late_rows[0]


def test_ws_tts_scheduler_failure_does_not_stop_asr(monkeypatch, tmp_path):
    original_next_deadline = RevisionStableTTSBuffer.next_deadline
    failed = False

    def fail_once(buffer):
        nonlocal failed
        if buffer.pending_count and not failed:
            failed = True
            raise RuntimeError("synthetic scheduler failure")
        return original_next_deadline(buffer)

    monkeypatch.setattr(RevisionStableTTSBuffer, "next_deadline", fail_once)
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 1.0
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "tts-scheduler-failure.jsonl")
    app = _create_app(
        args,
        _StableTTSSentenceASR(),
        translator=_FakeTranslator(),
        tts_synthesizer=_FakeTTSSynthesizer(),
    )

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        frame = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(frame)
        unavailable = None
        while True:
            message = ws.receive_json()
            if message.get("type") == "tts_status":
                unavailable = message
            if message.get("type") == "sentence_translation":
                break

        for _ in range(20):
            if unavailable is not None:
                break
            ws.send_json({"type": "ping"})
            while True:
                message = ws.receive_json()
                if message.get("type") == "tts_status":
                    unavailable = message
                if message.get("type") == "pong":
                    break
            if unavailable is not None:
                break
            time.sleep(0.01)
        assert unavailable is not None
        assert unavailable["status"] == "unavailable"

        ws.send_bytes(frame)
        assert _receive_until_type(ws, "partial")["type"] == "partial"

    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = [row for row in trace_rows if row.get("event") == "tts_stability_scheduler_failed"]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert "error" not in failures[0]


def test_ws_discards_translation_from_superseded_sentence_revision(tmp_path):
    class _UpdateASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.s1 = "First sentence is stable and complete."
            self.s2_short = "Second sentence starts as a complete long sentence."
            self.s2_long = (
                "Second sentence starts as a complete long sentence and later receives "
                "important extra words."
            )
            self.s3 = "Third sentence is stable and complete."

        def streaming_transcribe(self, wav, state):
            self.calls += 1
            state.language = "English"
            second = self.s2_short if self.calls <= 2 else self.s2_long
            state.text = f"{self.s1} {second} {self.s3}"
            return state

    class _OutOfOrderTranslator(_FakeTranslator):
        def __init__(self, blocked_text):
            super().__init__()
            self.blocked_text = blocked_text
            self.blocked_started = threading.Event()
            self.release_blocked = threading.Event()

        def translate(self, text: str, source_language: str = None, target_language: str = None):
            if text == self.blocked_text:
                self.blocked_started.set()
                assert self.release_blocked.wait(timeout=5.0)
            return f"translated:{text}"

    asr = _UpdateASR()
    translator = _OutOfOrderTranslator(asr.s2_short)
    args = _args()
    args.translation_workers = 3
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "translation-revisions.jsonl")
    app = _create_app(
        args,
        asr,
        translator=translator,
        tts_synthesizer=_FakeTTSSynthesizer(),
    )
    client = TestClient(app)

    events = []
    try:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.send_text('{"type":"set_translation_direction","translation_direction":"en2zh"}')
            _receive_until_type(ws, "translation_direction")
            ws.send_json(
                {
                    "type": "set_tts_enabled",
                    "enabled": True,
                    "tts_client_id": "client-a-12345678",
                }
            )
            _receive_until_type(ws, "tts_status")

            raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
            for _ in range(2):
                ws.send_bytes(raw)
                while True:
                    msg = ws.receive_json()
                    events.append(msg)
                    if msg.get("type") == "partial":
                        break
            assert translator.blocked_started.wait(timeout=2.0)

            ws.send_bytes(raw)
            latest_update = None
            latest_translation = None
            for _ in range(100):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "sentence_updated" and msg.get("text") == asr.s2_long:
                    latest_update = msg
                if (
                    latest_update
                    and msg.get("type") == "sentence_translation"
                    and msg.get("sentence_id") == latest_update.get("sentence_id")
                    and msg.get("translation") == f"translated:{asr.s2_long}"
                ):
                    latest_translation = msg
                    break
            assert latest_update is not None
            assert latest_translation is not None

            translator.release_blocked.set()
            time.sleep(0.05)
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break
    finally:
        translator.release_blocked.set()

    sid = str(latest_update["sentence_id"])
    latest_revision = int(latest_update["revision"])
    committed = [
        msg
        for msg in events
        if msg.get("type") == "sentence_committed" and str(msg.get("sentence_id", "")) == sid
    ]
    assert committed
    assert {int(msg["revision"]) for msg in committed} == {1}
    assert latest_revision == 2
    published = [
        msg
        for msg in events
        if msg.get("type") == "sentence_translation" and str(msg.get("sentence_id", "")) == sid
    ]
    assert published
    assert {int(msg["revision"]) for msg in published} == {latest_revision}
    assert {str(msg.get("translation", "")) for msg in published} == {f"translated:{asr.s2_long}"}
    spoken = [
        msg
        for msg in events
        if msg.get("type") == "tts_job" and str(msg.get("sentence_id", "")) == sid
    ]
    assert spoken
    assert {int(msg["revision"]) for msg in spoken} == {latest_revision}
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stale_rows = [row for row in trace_rows if row.get("event") == "translation_stale_drop"]
    assert any(
        row.get("sentence_id") == sid
        and int(row.get("queued_revision", 0)) == 1
        and int(row.get("current_revision", 0)) == 2
        and row.get("phase") == "post_inference"
        for row in stale_rows
    )


def test_ws_hard_cut_carries_unfinished_tail_to_next_segment():
    class _HardCutCarryASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self._segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self._segment_no += 1
            state.segment_no = self._segment_no
            return state

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            if int(getattr(state, "segment_no", 1)) == 1:
                state.text = "第一句不完整"
            else:
                state.text = "继续补全成句。"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = state.language or "Chinese"
            return state

    fake_asr = _HardCutCarryASR()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    collected = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

        time.sleep(1.1)
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

        ws.send_text('{"type":"finish"}')
        final_msg = None
        for _ in range(80):
            msg = ws.receive_json()
            collected.append(msg)
            if msg.get("type") == "final":
                final_msg = msg
                break

    assert final_msg is not None
    assert fake_asr.finish_calls >= 2
    committed = [str(m.get("text", "")).strip() for m in collected if m.get("type") == "sentence_committed"]
    assert "第一句不完整" not in committed
    assert "继续补全成句。" not in committed
    assert "第一句不完整继续补全成句。" in committed
    assert "第一句不完整继续补全成句。" in str(final_msg.get("committed_text", ""))


def test_ws_mid_speech_hard_cut_holds_terminal_hypothesis_for_continuation(tmp_path):
    prior = "感谢赞美主，奉靠我主基督。"
    continuation = "得胜的名求，Amen。"
    following = "现在开始下一段完整内容。"

    class _MidSpeechHardCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.segment_calls += 1
            if state.segment_no == 1:
                state.text = prior
            elif state.segment_calls <= 2:
                state.text = continuation
            else:
                state.text = f"{continuation}{following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            return state

    translator = _FakeTranslator()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "mid-speech-hard-cut.jsonl")
    app = _create_app(args, _MidSpeechHardCutASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        frame = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(6):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    source_texts = [call[0] for call in translator.calls]
    combined = f"{prior.removesuffix('。')}{continuation}"
    assert prior not in source_texts
    assert continuation not in source_texts
    assert combined in source_texts
    committed = [
        str(message.get("text", ""))
        for message in events
        if message.get("type") in {"sentence_committed", "sentence_updated"}
    ]
    assert combined in committed
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "hard_cut_mid_speech_tail_held" for row in trace_rows)


def test_ws_hard_cut_preserves_terminal_stable_clause_from_single_long_sentence(
    tmp_path,
):
    first_full = (
        "好，感谢神给我们机会，让我们一起分享神的话语，"
        "今天继续看尼希米记第八章，"
        "所以从第一章到第七章我们看见尼希米带领以色列人来建造。"
    )
    boundary_clause = "所以从第一章到第七章我们看见尼希米带领以色列人来建造。"
    following = "城墙，从第八章开始我们继续看神怎样修复他的子民。"

    class _SingleTerminalLongSentenceASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.segment_calls += 1
            if state.segment_no == 1:
                state.text = first_full
            elif state.segment_calls <= 2:
                state.text = "啊。"
            else:
                state.text = following
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            return state

    translator = _FakeTranslator()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "hard-cut-long-sentence-clause.jsonl")
    app = _create_app(args, _SingleTerminalLongSentenceASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        frame = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(8):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    latest_by_id = {
        str(message.get("sentence_id", "")): str(message.get("text", "")).strip()
        for message in events
        if message.get("type") in {"sentence_committed", "sentence_updated"}
    }
    committed = list(latest_by_id.values())
    assert boundary_clause in committed
    assert boundary_clause in [call[0] for call in translator.calls]
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "hard_cut_mid_speech_tail_held"
        and row.get("terminal_chars") == len(boundary_clause)
        for row in trace_rows
    )
    assert not any(
        row.get("event") == "pending_prefix_hard_cut_fallback_skip"
        for row in trace_rows
    )


def test_ws_mid_speech_hard_cut_waits_for_boundary_sentence_to_finish_growing(tmp_path):
    prior = "那卖过房子的人都知道，最麻烦的地方就是。"
    provisional = "我们要上市之前。"
    corrected = "我们要上市之前，就是把这个房子准备好，才能上市。"
    following_tail = "那我一直都以为自己会把房子整理得还不错"

    class _GrowingBoundaryAfterHardCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.segment_calls += 1
            if state.segment_no == 1:
                state.text = prior
            elif state.segment_calls <= 4:
                state.text = provisional
            elif state.segment_calls <= 8:
                state.text = corrected
            else:
                state.text = f"{corrected}{following_tail}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            return state

    translator = _FakeTranslator()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "growing-hard-cut-boundary.jsonl")
    app = _create_app(args, _GrowingBoundaryAfterHardCutASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        frame = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(14):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    provisional_combined = f"{prior.removesuffix('。')}{provisional}"
    corrected_combined = f"{prior.removesuffix('。')}{corrected}"
    source_texts = [call[0] for call in translator.calls]
    assert provisional_combined not in source_texts
    assert corrected_combined in "".join(source_texts)
    committed = [
        str(message.get("text", ""))
        for message in events
        if message.get("type") in {"sentence_committed", "sentence_updated"}
    ]
    assert corrected_combined in "".join(committed)
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "hard_cut_boundary_commit_wait" for row in trace_rows)


def test_ws_hard_cut_keeps_alignment_when_boundary_punctuation_is_retracted(tmp_path):
    prior = "那今天的经文在《尼希米记》第三章，我们知道尼希米终于回到。"
    provisional = "耶路撒冷要重建这个城墙。"
    provisional_with_tail = f"{provisional}从"
    corrected = "耶路撒冷要重建这个城墙，重建这些门和塔楼。"
    following = "那我们就看到复兴也是重建，重建也是复兴。"

    class _RetractedBoundaryAfterHardCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self.segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self.segment_no += 1
            state.segment_no = self.segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            state.language = "Chinese"
            state.segment_calls += 1
            if state.segment_no == 1:
                state.text = prior
            elif state.segment_calls <= 2:
                state.text = provisional
            elif state.segment_calls <= 4:
                state.text = provisional_with_tail
            elif state.segment_calls <= 8:
                state.text = corrected
            else:
                state.text = f"{corrected}{following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "Chinese"
            return state

    translator = _FakeTranslator()
    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "retracted-hard-cut-boundary.jsonl")
    app = _create_app(args, _RetractedBoundaryAfterHardCutASR(), translator=translator)

    events = []
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        frame = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()
        ws.send_bytes(frame)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(14):
            ws.send_bytes(frame)
            while True:
                message = ws.receive_json()
                events.append(message)
                if message.get("type") == "partial":
                    break
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    prior_base = prior.removesuffix("。")
    provisional_combined = f"{prior_base}{provisional}"
    corrected_combined = f"{prior_base}{corrected}"
    latest_by_id = {}
    for message in events:
        if message.get("type") in {"sentence_committed", "sentence_updated"}:
            latest_by_id[str(message.get("sentence_id", ""))] = str(message.get("text", "")).strip()

    assert provisional_combined not in latest_by_id.values()
    assert corrected_combined in "".join(latest_by_id.values())
    assert corrected_combined in "".join(call[0] for call in translator.calls)
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("event") == "pending_prefix_retained_for_candidate_alignment"
        and row.get("carry_alignment") is True
        for row in trace_rows
    )


def test_ws_hard_cut_keeps_terminal_prefix_separate_when_next_sentence_grows(tmp_path):
    prior = "She's a girl."
    initial_next = "Yes, so we have."
    grown_next = (
        "Yes, so we have a five-year-old boy, a three-year-old boy, "
        "and now we have a third child who's a girl."
    )
    following = "The following sentence is stable and complete."

    class _GrowingAfterHardCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self._segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self._segment_no += 1
            state.segment_no = self._segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.segment_calls += 1
            if int(state.segment_no) == 1:
                state.text = prior
            elif int(state.segment_calls) <= 2:
                state.text = initial_next
            else:
                state.text = f"{grown_next} {following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            return state

    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.early_translation_short_stable_sec = 0.0
    args.early_translation_short_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "hard-cut-terminal-prefix.jsonl")
    app = _create_app(args, _GrowingAfterHardCutASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(6):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break

        ws.send_text('{"type":"finish"}')
        final_msg = None
        for _ in range(100):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                final_msg = msg
                break

    assert final_msg is not None
    latest_by_id = {}
    for msg in events:
        if msg.get("type") in {"sentence_committed", "sentence_updated"}:
            latest_by_id[str(msg.get("sentence_id", ""))] = str(msg.get("text", "")).strip()
    latest = list(latest_by_id.values())
    assert prior in latest
    assert grown_next in latest
    assert following in latest
    assert f"{prior.removesuffix('.')} {initial_next}" not in latest
    committed_text = str(final_msg.get("committed_text", ""))
    assert prior in committed_text
    assert grown_next in committed_text
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "pending_prefix_terminal_boundary_preserved" for row in trace_rows)
    assert any(row.get("event") == "pending_prefix_retained_for_candidate_alignment" for row in trace_rows)


def test_ws_hard_cut_keeps_short_terminal_from_multi_sentence_segment_separate(tmp_path):
    earlier = "First of all, is our baby a boy or a girl?"
    prior = "She's a girl."
    initial_next = "Yes, so we have."
    grown_next = (
        "Yes, so we have a five-year-old boy, a three-year-old boy, "
        "and now we have a third child who's a girl."
    )
    following = "The following sentence is stable and complete."

    class _GrowingAfterMultiSentenceHardCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self._segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self._segment_no += 1
            state.segment_no = self._segment_no
            state.segment_calls = 0
            return state

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.segment_calls += 1
            if int(state.segment_no) == 1:
                state.text = f"{earlier} {prior}"
            elif int(state.segment_calls) <= 2:
                state.text = initial_next
            else:
                state.text = f"{grown_next} {following}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            return state

    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.early_translation_short_stable_sec = 0.0
    args.early_translation_short_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "hard-cut-multi-sentence-terminal-prefix.jsonl")
    app = _create_app(args, _GrowingAfterMultiSentenceHardCutASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        time.sleep(1.1)
        for _ in range(6):
            ws.send_bytes(raw)
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break

        ws.send_text('{"type":"finish"}')
        final_msg = None
        for _ in range(100):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                final_msg = msg
                break

    assert final_msg is not None
    latest_by_id = {}
    for msg in events:
        if msg.get("type") in {"sentence_committed", "sentence_updated"}:
            latest_by_id[str(msg.get("sentence_id", ""))] = str(msg.get("text", "")).strip()
    latest = list(latest_by_id.values())
    assert earlier in latest
    assert prior in latest
    assert grown_next in latest
    assert following in latest
    assert f"{prior.removesuffix('.')} {initial_next}" not in latest


def test_ws_hard_cut_skips_recent_long_duplicate_from_next_segment(tmp_path):
    duplicate = (
        "Long duplicate segment without any inner terminal boundary keeps enough words "
        "to be considered a replay after a segment cut and should only be committed once."
    )
    followup = "The followup sentence is complete and distinct enough to be committed safely."
    trailing = "Additional context continues beyond the stable boundary"

    class _DuplicateAfterCutASR(_FakeASR):
        def __init__(self):
            super().__init__()
            self._segment_no = 0

        def init_streaming_state(self, **kwargs):
            state = super().init_streaming_state(**kwargs)
            self._segment_no += 1
            state.segment_no = self._segment_no
            return state

        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            if int(getattr(state, "segment_no", 1)) == 1:
                state.text = duplicate
            else:
                state.text = f"{duplicate} {followup} {trailing}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            return state

    args = _args()
    args.segment_hard_cut_sec = 1.0
    args.segment_overlap_sec = 0.0
    args.final_redecode_on_stop = False
    args.early_translation_stable_sec = 0.0
    args.early_translation_stable_hits = 2
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "hard-cut-duplicate-trace.jsonl")
    app = _create_app(args, _DuplicateAfterCutASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1200, -1200] * 2400, dtype="<i2").tobytes()

        def send_and_collect_partial():
            ws.send_bytes(raw)
            for _ in range(40):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    return
            pytest.fail("did not receive partial")

        send_and_collect_partial()
        send_and_collect_partial()

        time.sleep(1.1)
        send_and_collect_partial()
        send_and_collect_partial()
        send_and_collect_partial()

        time.sleep(1.1)
        send_and_collect_partial()
        send_and_collect_partial()

    committed = [str(m.get("text", "")).strip() for m in events if m.get("type") == "sentence_committed"]
    assert committed.count(duplicate) == 1
    assert committed.count(followup) == 1
    trace_rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commit_evals = [row for row in trace_rows if row.get("event") == "commit_eval"]
    assert commit_evals
    assert all("candidate_cursor_before" in row and "candidate_cursor_after" in row for row in commit_evals)
    assert any(
        row.get("event") == "candidate_action" and row.get("action") == "structural_overlap_skip"
        for row in trace_rows
    )


def test_ws_preserves_intentional_repeated_short_english_sentences():
    duplicate = "Tiny replay confirmed."
    first = "The first complete sentence is long enough to be committed safely."
    bridge = "A nearby bridge sentence keeps the replay within the duplicate window."
    last = "The final completed sentence is only here to keep the duplicate out of holdback."

    class _ShortDuplicateASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = f"{first} {duplicate} {bridge} {duplicate} {duplicate} {last}"
            return state

    app = _create_app(_args(), _ShortDuplicateASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(2):
            ws.send_bytes(raw)
            for _ in range(40):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break

    committed = [str(msg.get("text", "")).strip() for msg in events if msg.get("type") == "sentence_committed"]
    assert duplicate in committed
    assert committed.count(duplicate) == 3


def test_ws_consumes_suffix_matching_candidate_without_repeating_following_sentence():
    intro = "Let's talk about the first question, which comes from Olga."
    repeat = "Olga."
    following = "Well, you might or might not know that we have a name for our baby."
    holdback = "The next complete sentence remains held back for stability."

    class _RepeatedNameASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = f"{intro} {repeat} {following} {holdback}"
            return state

    app = _create_app(_args(), _RepeatedNameASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        for _ in range(3):
            ws.send_bytes(raw)
            for _ in range(40):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "partial":
                    break

    committed = [str(msg.get("text", "")).strip() for msg in events if msg.get("type") == "sentence_committed"]
    assert committed.count(intro) == 1
    assert committed.count(repeat) == 1
    assert committed.count(following) == 1


def test_ws_writes_backend_subtitle_trace_jsonl_file(tmp_path):
    class _TraceASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = "The trace file test sentence is complete and long enough."
            return state

    args = _args()
    args.subtitle_trace_log = True
    args.subtitle_trace_log_file = str(tmp_path / "subtitle-trace.jsonl")
    app = _create_app(args, _TraceASR())
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")
        ws.send_text('{"type":"finish"}')
        _receive_until_type(ws, "final")

    rows = [
        json.loads(line)
        for line in Path(args.subtitle_trace_log_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("topic") == "subtitle_state" and row.get("event") == "ws_send" for row in rows)
    assert any(row.get("topic") == "text_pool" for row in rows)


def test_ws_supports_runtime_translation_direction_switch():
    class _EnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = (
                "First sentence is complete and long enough. "
                "Second sentence is complete and long enough."
            )
            return state

    args = _args()
    args.translation_source_language = "Chinese"
    args.translation_target_language = "English"
    translator = _FakeTranslator()
    app = _create_app(args, _EnglishASR(), translator=translator)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["translation_direction"] == "zh2en"

        ws.send_text('{"type":"set_translation_direction","translation_direction":"en2zh"}')
        changed = _receive_until_type(ws, "translation_direction")
        assert changed["translation_direction"] == "en2zh"

        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "sentence_committed")
        tr = _receive_until_type(ws, "sentence_translation")
        assert tr["type"] == "sentence_translation"
        assert tr["translation"].startswith("[English->Chinese]")

    assert translator.calls
    _, src_lang, tgt_lang = translator.calls[-1]
    assert src_lang == "English"
    assert tgt_lang == "Chinese"


def test_ws_preserves_zh2en_policy_direction_after_latin_source_autofallback():
    class _LatinTextASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = (
                "The name Jesus Christ appears in this complete sentence. "
                "Another complete sentence follows with enough words."
            )
            return state

    class _DirectionRecordingTranslator:
        def __init__(self):
            self.calls = []

        def translate(
            self,
            text,
            source_language=None,
            target_language=None,
            translation_direction=None,
        ):
            self.calls.append(
                (
                    str(text or ""),
                    str(source_language or ""),
                    str(target_language or ""),
                    str(translation_direction or ""),
                )
            )
            return f"translated:{text}"

    translator = _DirectionRecordingTranslator()
    app = _create_app(_args(), _LatinTextASR(), translator=translator)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["translation_direction"] == "zh2en"

        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(raw)
        _receive_until_type(ws, "sentence_committed")
        _receive_until_type(ws, "sentence_translation")

    assert translator.calls
    _, source_language, target_language, direction = translator.calls[-1]
    assert source_language == "English"
    assert target_language == "English"
    assert direction == "zh2en"


def test_ws_retries_builtin_zh2en_translation_that_keeps_chinese_text():
    class _ChineseASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = "牧师和同工一起服侍。下一句话已经完整。"
            return state

    class _MixedThenEnglishTranslator:
        enforce_target_language_output = True

        def __init__(self):
            self.calls = []

        def translate(
            self,
            text,
            source_language=None,
            target_language=None,
            translation_direction=None,
            strict_target_language=False,
        ):
            self.calls.append(
                {
                    "text": str(text or ""),
                    "source_language": str(source_language or ""),
                    "target_language": str(target_language or ""),
                    "translation_direction": str(translation_direction or ""),
                    "strict_target_language": bool(strict_target_language),
                }
            )
            if not strict_target_language:
                return "The pastor and 同工 serve together."
            return "The pastor and fellow workers serve together."

    translator = _MixedThenEnglishTranslator()
    app = _create_app(_args(), _ChineseASR(), translator=translator)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        committed = _receive_until_type(ws, "sentence_committed")
        translated = _receive_until_type(ws, "sentence_translation")

    assert committed["text"] == "牧师和同工一起服侍。"
    assert translated["translation"] == "The pastor and fellow workers serve together."
    strict_flags = [call["strict_target_language"] for call in translator.calls]
    assert strict_flags.count(False) == strict_flags.count(True)
    assert strict_flags.count(True) >= 1


def test_ws_rejects_builtin_zh2en_translation_if_strict_retry_still_keeps_chinese_text():
    class _ChineseASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "Chinese"
            state.text = "第一句话已经完整。下一句话也已经完整。"
            return state

    class _AlwaysMixedTranslator:
        enforce_target_language_output = True

        def __init__(self):
            self.calls = []

        def translate(
            self,
            text,
            source_language=None,
            target_language=None,
            translation_direction=None,
            strict_target_language=False,
        ):
            del text, source_language, target_language, translation_direction
            self.calls.append(bool(strict_target_language))
            return "The translation still contains 中文."

    args = _args()
    args.final_redecode_on_stop = False
    translator = _AlwaysMixedTranslator()
    app = _create_app(args, _ChineseASR(), translator=translator)
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        committed = _receive_until_type(ws, "sentence_committed")
        events.append(committed)
        ws.send_json({"type": "finish", "mode": "stop"})
        events.extend(_collect_through_final(ws))

    assert translator.calls
    assert len(translator.calls) % 2 == 0
    assert all(
        translator.calls[index : index + 2] == [False, True]
        for index in range(0, len(translator.calls), 2)
    )
    assert [event for event in events if event.get("type") == "sentence_translation"] == []


def test_ws_stop_does_not_reset_committed_subtitles_for_noncanonical_final_tail():
    first = "The first committed sentence is complete and long enough."
    second = "The second held sentence is complete and long enough."
    noncanonical_final = "A disconnected final tail should not replace committed subtitles."

    class _NoncanonicalFinalASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = f"{first} {second}"
            return state

        def finish_streaming_transcribe(self, state):
            self.finish_calls += 1
            state.language = "English"
            state.text = noncanonical_final
            return state

    args = _args()
    args.final_redecode_on_stop = False
    app = _create_app(args, _NoncanonicalFinalASR())
    client = TestClient(app)

    events = []
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        ws.send_bytes(raw)
        _receive_until_type(ws, "partial")

        ws.send_bytes(raw)
        committed = _receive_until_type(ws, "sentence_committed")
        assert committed["text"] == first

        ws.send_text('{"type":"finish"}')
        final_msg = None
        for _ in range(80):
            msg = ws.receive_json()
            events.append(msg)
            if msg.get("type") == "final":
                final_msg = msg
                break

    assert final_msg is not None
    assert [
        msg
        for msg in events
        if msg.get("type") == "sentence_reset" and msg.get("reason") == "final_commit_reconcile"
    ] == []
    assert first in str(final_msg.get("committed_text", ""))
