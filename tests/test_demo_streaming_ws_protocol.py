from types import SimpleNamespace
import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import voxbridge.cli.demo_streaming_ws as demo_streaming_ws
from voxbridge.cli.demo_streaming_ws import _create_app, _hash_auth_password
from voxbridge.tts.jobs import TTSReadyItem
from voxbridge.tts.kokoro_onnx import SynthesizedAudio


class _FakeASR:
    def __init__(self):
        self.init_calls = []
        self.finish_calls = 0
        self.transcribe_calls = []

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

    def synthesize(self, text: str, target_language: str):
        self.calls.append((text, target_language))
        return SynthesizedAudio(b"RIFF-fake-wav", sample_rate=24000, duration_ms=750)


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


def test_listener_page_requires_auth_and_login_returns_to_listener():
    args = _args()
    args.auth_enabled = True
    args.auth_username = "admin"
    args.auth_password_hash = _hash_auth_password("secret")
    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())
    client = TestClient(app)

    redirect = client.get("/listen", follow_redirects=False)
    assert redirect.status_code in {302, 303, 307}
    assert redirect.headers["location"] == "/login?next=%2Flisten"
    login_page = client.get(redirect.headers["location"])
    assert login_page.status_code == 200
    assert 'name="next" value="/listen"' in login_page.text

    login = client.post(
        "/login",
        data={"username": "admin", "password": "secret", "next": "/listen"},
        follow_redirects=False,
    )
    assert login.status_code in {302, 303, 307}
    assert login.headers["location"] == "/listen"
    page = client.get("/listen")
    assert page.status_code == 200
    assert "译文实时朗读" in page.text


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


def _collect_through_final(ws, max_steps=160):
    events = []
    for _ in range(max_steps):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") == "final":
            return events
    pytest.fail(f"did not receive final, seen={[event.get('type') for event in events]}")


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


def test_ws_applies_context_once_to_complete_segment_by_default(tmp_path):
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


def test_ws_early_translates_stable_newest_completed_sentence_without_waiting_for_next_sentence():
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

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        raw = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        ws.send_bytes(raw)
        first_partial = _receive_until_type(ws, "partial")
        assert first_partial["type"] == "partial"
        assert first_partial["tentative_text"] == "这是一个已经稳定完成并且长度足够的句子。"

        ws.send_bytes(raw)
        committed = ws.receive_json()
        assert committed["type"] == "sentence_committed"
        assert committed["text"] == "这是一个已经稳定完成并且长度足够的句子。"
        tr = _receive_until_type(ws, "sentence_translation")
        assert tr["translation"].startswith("[Chinese->English]")

    assert translator.calls


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


def test_ws_early_translates_stable_short_english_terminal_sentence(tmp_path):
    sentence = "The Short Session Topic."

    class _ShortEnglishASR(_FakeASR):
        def streaming_transcribe(self, wav, state):
            assert isinstance(wav, np.ndarray)
            state.language = "English"
            state.text = sentence
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
                if msg["type"] == "sentence_updated":
                    updated.append(msg["text"])
                    continue
                if msg["type"] == "partial":
                    partials.append(msg)
                    break

    assert asr.s2_old in committed
    assert asr.s2_new in updated
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
                state.text = f"{duplicate} {followup}"
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
