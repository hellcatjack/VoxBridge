from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voxbridge.cli.demo_streaming_ws import _create_app


class _FakeASR:
    def __init__(self):
        self.init_calls = []
        self.finish_calls = 0

    def init_streaming_state(self, **kwargs):
        self.init_calls.append(kwargs)
        return SimpleNamespace(language="", text="", kwargs=kwargs)

    def streaming_transcribe(self, wav, state):
        assert isinstance(wav, np.ndarray)
        state.language = "Chinese"
        state.text = state.text + "partial"
        return state

    def finish_streaming_transcribe(self, state):
        self.finish_calls += 1
        state.language = state.language or "Chinese"
        state.text = state.text + "|final"
        return state

    def transcribe(self, audio, context="", language=None):
        class Out:
            language = "Chinese"
            text = "pseudo"

        return [Out()]


class _FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text: str, source_language: str = None, target_language: str = None):
        src = str(source_language or "")
        tgt = str(target_language or "")
        self.calls.append((str(text or ""), src, tgt))
        return f"[{src}->{tgt}] {text}"


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
    )


def _receive_until_type(ws, expected_type: str, max_steps: int = 40):
    seen = []
    for _ in range(max_steps):
        msg = ws.receive_json()
        if msg.get("type") == expected_type:
            return msg
        seen.append(msg.get("type"))
    pytest.fail(f"did not receive {expected_type}, seen={seen}")


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


def test_ws_uses_cli_force_language_for_initial_state():
    fake_asr = _FakeASR()
    args = _args()
    args.force_language = "Chinese"
    app = _create_app(args, fake_asr)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready

    assert fake_asr.init_calls[-1]["language"] == "Chinese"


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
