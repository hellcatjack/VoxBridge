from __future__ import annotations

import pytest

from voxbridge.cli.demo_streaming_ws import INDEX_HTML_TEMPLATE

sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = sync_api.sync_playwright


def _render_index() -> str:
    return (
        INDEX_HTML_TEMPLATE.replace("__CHUNK_MS__", "200")
        .replace("__SUBTITLE_TRACE__", "false")
        .replace("__SUBTITLE_TRACE_MAX_EVENTS__", "1200")
        .replace("__ASR_CONTEXT_MAX_TERMS__", "24")
        .replace("__ASR_CONTEXT_MAX_CHARS__", "160")
    )


@pytest.fixture(params=("chromium", "webkit"))
def main_page(request):
    with sync_playwright() as playwright:
        browser_type = getattr(playwright, request.param)
        try:
            browser = browser_type.launch(headless=True)
        except Exception as exc:
            pytest.skip(f"{request.param} browser unavailable: {exc}")
        page = browser.new_page()
        page.route(
            "https://voxbridge.test/",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body=_render_index(),
            ),
        )
        page.goto("https://voxbridge.test/")
        yield page
        browser.close()


def _new_gate(main_page, **options):
    return main_page.evaluate(
        "options => window.__audioActivityGateDebug.create(options)",
        options,
    )


def test_audio_gate_replays_short_pause_instead_of_cutting(main_page):
    gate_id = _new_gate(
        main_page,
        sampleRate=1000,
        frameMs=20,
        speechStartMs=40,
        silenceGateMs=100,
        preRollMs=60,
        heartbeatMs=200,
    )

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.2, 60);
          gate.drainEvents();
          gate.feedConstant(0.0, 60);
          gate.feedConstant(0.2, 20);
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    assert [event["type"] for event in events] == ["audio"]
    assert events[0]["samples"] == 80


def test_audio_gate_suppresses_long_silence_and_keeps_heartbeat(main_page):
    gate_id = _new_gate(
        main_page,
        sampleRate=1000,
        frameMs=20,
        speechStartMs=40,
        silenceGateMs=100,
        preRollMs=60,
        endTailMs=40,
        heartbeatMs=200,
    )

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.2, 60);
          gate.drainEvents();
          gate.feedConstant(0.0, 320);
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    controls = [event for event in events if event["type"] == "control"]
    audio = [event for event in events if event["type"] == "audio"]
    assert [event["message"]["type"] for event in controls] == [
        "audio_silence",
        "audio_silence",
    ]
    assert controls[0]["message"]["duration_ms"] == 100
    assert controls[1]["message"]["duration_ms"] == 200
    assert [event["samples"] for event in audio] == [40]


def test_audio_gate_default_preserves_400ms_endpoint_tail(main_page):
    gate_id = _new_gate(main_page, sampleRate=1000)

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.2, 200);
          gate.drainEvents();
          gate.feedConstant(0.0, 700);
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    assert [event["type"] for event in events] == ["audio", "control"]
    assert events[0]["samples"] == 400
    assert events[1]["message"]["type"] == "audio_silence"
    assert events[1]["message"]["duration_ms"] == 700


def test_audio_gate_sends_preroll_before_resumed_speech(main_page):
    gate_id = _new_gate(
        main_page,
        sampleRate=1000,
        frameMs=20,
        speechStartMs=40,
        silenceGateMs=100,
        preRollMs=60,
        heartbeatMs=200,
    )

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.0, 200);
          gate.drainEvents();
          gate.feedConstant(0.2, 40);
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    assert [event["type"] for event in events] == ["control", "audio"]
    start = events[0]["message"]
    assert start["type"] == "audio_speech_start"
    assert start["preroll_samples"] == 60
    assert events[1]["samples"] == 60


def test_audio_gate_flushes_unconfirmed_short_speech_on_finish(main_page):
    gate_id = _new_gate(
        main_page,
        sampleRate=1000,
        frameMs=20,
        speechStartMs=120,
        silenceGateMs=700,
        preRollMs=400,
        heartbeatMs=1000,
    )

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.2, 80);
          gate.finish();
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    assert [event["type"] for event in events] == ["control", "audio"]
    assert events[0]["message"]["type"] == "audio_speech_start"
    assert events[0]["message"]["preroll_samples"] == 80
    assert events[1]["samples"] == 80


def test_audio_gate_does_not_promote_floor_noise_after_long_digital_silence(main_page):
    gate_id = _new_gate(
        main_page,
        sampleRate=1000,
        frameMs=20,
        speechStartMs=40,
        silenceGateMs=100,
        preRollMs=60,
        heartbeatMs=200,
    )

    events = main_page.evaluate(
        """
        ([id]) => {
          const gate = window.__audioActivityGateDebug.get(id);
          gate.feedConstant(0.0, 5000);
          gate.drainEvents();
          gate.feedConstant(0.0001, 100);
          return gate.drainEvents();
        }
        """,
        [gate_id],
    )

    assert not [event for event in events if event["type"] == "audio"]
    assert not [
        event
        for event in events
        if event["type"] == "control"
        and event["message"]["type"] == "audio_speech_start"
    ]
