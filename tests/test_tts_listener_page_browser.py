from __future__ import annotations

import json
import shutil

import pytest

from voxbridge.tts.listener_page import TTS_LISTENER_HTML

sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = sync_api.sync_playwright


@pytest.fixture
def listener_page():
    chrome = shutil.which("google-chrome")
    if chrome is None:
        pytest.skip("system Google Chrome is unavailable")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.route(
            "https://voxbridge.test/listen",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body=TTS_LISTENER_HTML,
            ),
        )
        page.set_default_timeout(3000)
        yield page
        browser.close()


def _install_hls_harness(
    listener_page,
    *,
    reject_first_play: bool = False,
    caption_snapshot: dict | None = None,
):
    snapshot = caption_snapshot or {"live_edge_at_ms": None, "cues": []}
    listener_page.add_init_script(
        f"""
        window.__ttsEvents = [];
        window.__ttsPlayCalls = [];
        window.__ttsPauseCalls = 0;
        window.__ttsFetchCalls = [];
        window.__ttsCaptionSnapshot = {json.dumps(snapshot)};
        window.__ttsMediaActions = {{}};
        window.__ttsMediaSession = {{
          metadata: null,
          playbackState: "none",
          setActionHandler(name, handler) {{
            window.__ttsMediaActions[name] = handler;
          }},
        }};
        Object.defineProperty(navigator, "mediaSession", {{
          configurable: true,
          value: window.__ttsMediaSession,
        }});
        window.MediaMetadata = class FakeMediaMetadata {{
          constructor(values) {{ Object.assign(this, values); }}
        }};
        HTMLMediaElement.prototype.play = function() {{
          const call = {{
            src: String(this.src),
            muted: this.muted,
            playsInline: this.playsInline,
            userActive: navigator.userActivation
              ? navigator.userActivation.isActive
              : true,
          }};
          window.__ttsPlayCalls.push(call);
          window.__ttsEvents.push("play");
          if ({str(reject_first_play).lower()} && window.__ttsPlayCalls.length === 1) {{
            return Promise.reject(new DOMException("blocked", "NotAllowedError"));
          }}
          return Promise.resolve();
        }};
        HTMLMediaElement.prototype.pause = function() {{
          window.__ttsPauseCalls += 1;
        }};
        HTMLMediaElement.prototype.load = function() {{}};
        window.fetch = function(url, options = {{}}) {{
          const call = {{
            url: String(url),
            method: String(options.method || "GET"),
          }};
          window.__ttsFetchCalls.push(call);
          window.__ttsEvents.push(`fetch:${{call.method}}`);
          const payload = call.url.includes("/captions")
            ? window.__ttsCaptionSnapshot
            : {{
                available: true,
                listener_count: 2,
                queue_depth: 0,
                pending_audio_ms: 6500,
                encoder_active: true,
                producer_active: true,
                last_error: "",
              }};
          return Promise.resolve({{
            ok: true,
            status: 200,
            json: () => Promise.resolve(payload),
          }});
        }};
        """
    )


def _start_hls_harness(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Connected'"
    )


def _set_live_lag(listener_page, *, current_time: float, live_edge: float):
    listener_page.evaluate(
        """({ currentTime, liveEdge }) => {
          const playback = document.querySelector("#ttsPlayback");
          window.__ttsCurrentTime = currentTime;
          window.__ttsLiveEdge = liveEdge;
          Object.defineProperty(playback, "currentTime", {
            configurable: true,
            get: () => window.__ttsCurrentTime,
            set: value => { window.__ttsCurrentTime = Number(value); },
          });
          Object.defineProperty(playback, "seekable", {
            configurable: true,
            get: () => ({
              length: 1,
              start: () => 0,
              end: () => window.__ttsLiveEdge,
            }),
          });
          playback.dispatchEvent(new Event("timeupdate"));
        }""",
        {"currentTime": current_time, "liveEdge": live_edge},
    )


def test_listener_rate_selection_persists_after_reload(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    listener_page.reload()
    assert listener_page.input_value("#playbackRate") == "1.2"


def test_unsupported_legacy_rate_falls_back_to_default(listener_page):
    listener_page.add_init_script(
        "window.localStorage.setItem('voxbridge.ttsPlaybackRate', '1.5');"
    )
    listener_page.goto("https://voxbridge.test/listen")
    assert listener_page.input_value("#playbackRate") == "1"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (390, 844), (844, 390), (320, 568)],
)
def test_listener_fits_viewport_without_document_scrollbars(
    listener_page,
    width,
    height,
):
    listener_page.set_viewport_size({"width": width, "height": height})
    listener_page.goto("https://voxbridge.test/listen")
    overflow = listener_page.evaluate(
        """({
          htmlX: document.documentElement.scrollWidth > window.innerWidth,
          htmlY: document.documentElement.scrollHeight > window.innerHeight,
          bodyX: document.body.scrollWidth > window.innerWidth,
          bodyY: document.body.scrollHeight > window.innerHeight,
        })"""
    )
    assert overflow == {"htmlX": False, "htmlY": False, "bodyX": False, "bodyY": False}
    for selector in (
        "#connectionStatus",
        "#liveCaption",
        "#playbackRate",
        "#startListening",
        "#stopListening",
    ):
        locator = listener_page.locator(selector)
        assert locator.is_visible()
        box = locator.bounding_box()
        assert box is not None
        assert box["x"] >= 0 and box["y"] >= 0
        assert box["x"] + box["width"] <= width + 1
        assert box["y"] + box["height"] <= height + 1

    listener_page.locator("#liveCaption").evaluate(
        """node => {
          node.textContent = "This is a deliberately long translated sentence that verifies the Live Audio caption wraps naturally while every control remains inside the one-screen listener layout.";
        }"""
    )
    overflow_after_caption = listener_page.evaluate(
        """({
          htmlX: document.documentElement.scrollWidth > window.innerWidth,
          htmlY: document.documentElement.scrollHeight > window.innerHeight,
          bodyX: document.body.scrollWidth > window.innerWidth,
          bodyY: document.body.scrollHeight > window.innerHeight,
        })"""
    )
    assert overflow_after_caption == {
        "htmlX": False,
        "htmlY": False,
        "bodyX": False,
        "bodyY": False,
    }


def test_rate_change_updates_persistent_media_element(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.defaultPlaybackRate"
    ) == 1.2
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.preservesPitch"
    ) is True


def test_listener_start_stop_preserves_rate_selection(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "0.9")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Connected'"
    )
    listener_page.select_option("#playbackRate", "1.1")
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.1
    listener_page.locator("#stopListening").click()
    assert listener_page.input_value("#playbackRate") == "1.1"
    assert listener_page.text_content("#connectionStatus") == "Stopped"


def test_listener_temporarily_catches_up_without_skipping_audio(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "0.8")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Connected'"
    )

    _set_live_lag(listener_page, current_time=80, live_edge=100)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2
    assert "Catching up" in listener_page.text_content("#playbackStatus")
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.currentTime"
    ) == 80

    _set_live_lag(listener_page, current_time=96, live_edge=100)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 0.8
    assert listener_page.text_content("#playbackStatus") == "Listening to live translation"


def test_listener_uses_at_least_normal_speed_while_page_is_hidden(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "0.8")
    _start_hls_harness(listener_page)

    listener_page.evaluate(
        """() => {
          Object.defineProperty(document, "hidden", {
            configurable: true,
            get: () => true,
          });
          Object.defineProperty(document, "visibilityState", {
            configurable: true,
            get: () => "hidden",
          });
          document.dispatchEvent(new Event("visibilitychange"));
        }"""
    )

    assert listener_page.input_value("#playbackRate") == "0.8"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_listener_status_shows_real_pending_audio_seconds(listener_page):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)

    listener_page.wait_for_function(
        "document.querySelector('#queueStatus').textContent.includes('7s queued')"
    )


def test_listener_caption_follows_device_playhead_instead_of_newest_cue(listener_page):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 110_000,
            "cues": [
                {
                    "cue_id": "older-cue",
                    "start_at_ms": 103_000,
                    "end_at_ms": 105_000,
                    "text": "The sentence this device is hearing.",
                },
                {
                    "cue_id": "newer-cue",
                    "start_at_ms": 107_000,
                    "end_at_ms": 109_000,
                    "text": "The newer server-side sentence.",
                },
            ],
        },
    )
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=94, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.includes('this device')"
    )
    assert listener_page.text_content("#liveCaption") == (
        "The sentence this device is hearing."
    )
    assert listener_page.get_attribute("#nowPlaying", "data-speaking") == "true"

    _set_live_lag(listener_page, current_time=98, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.includes('newer server')"
    )
    assert listener_page.text_content("#liveCaption") == (
        "The newer server-side sentence."
    )


def test_listener_retains_caption_between_cues_without_empty_transition(listener_page):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 110_000,
            "cues": [
                {
                    "cue_id": "first-cue",
                    "start_at_ms": 103_000,
                    "end_at_ms": 105_000,
                    "text": "The first spoken sentence remains visible.",
                },
                {
                    "cue_id": "second-cue",
                    "start_at_ms": 107_000,
                    "end_at_ms": 109_000,
                    "text": "The second spoken sentence replaces it directly.",
                },
            ],
        },
    )
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.startsWith('The first')"
    )
    listener_page.evaluate(
        """() => {
          const caption = document.querySelector("#liveCaption");
          window.__captionValues = [caption.textContent];
          new MutationObserver(() => {
            window.__captionValues.push(caption.textContent);
          }).observe(caption, { childList: true, characterData: true, subtree: true });
        }"""
    )

    _set_live_lag(listener_page, current_time=95.5, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#nowPlaying').dataset.speaking === 'false'"
    )
    assert listener_page.text_content("#liveCaption") == (
        "The first spoken sentence remains visible."
    )

    _set_live_lag(listener_page, current_time=98, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.startsWith('The second')"
    )
    observed = listener_page.evaluate("window.__captionValues.slice()")
    assert observed[-1] == "The second spoken sentence replaces it directly."
    assert all(value.strip() for value in observed)


def test_listener_stop_resets_caption_and_stops_caption_polling(listener_page):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 110_000,
            "cues": [
                {
                    "cue_id": "active-cue",
                    "start_at_ms": 107_000,
                    "end_at_ms": 109_000,
                    "text": "A translated sentence is currently playing.",
                }
            ],
        },
    )
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=98, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.startsWith('A translated')"
    )

    listener_page.locator("#stopListening").click()
    caption_calls_at_stop = listener_page.evaluate(
        "window.__ttsFetchCalls.filter(call => call.url.includes('/captions')).length"
    )
    listener_page.wait_for_timeout(650)

    assert listener_page.text_content("#liveCaption") == "Waiting to start"
    assert listener_page.get_attribute("#nowPlaying", "data-speaking") == "false"
    assert listener_page.evaluate(
        "window.__ttsFetchCalls.filter(call => call.url.includes('/captions')).length"
    ) == caption_calls_at_stop


def test_listener_starts_one_unmuted_hls_stream_inside_click(listener_page):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)

    calls = listener_page.evaluate("window.__ttsPlayCalls.slice()")
    assert len(calls) == 1
    assert "/api/tts/live/iphone-" in calls[0]["src"]
    assert calls[0]["src"].endswith("/index.m3u8")
    assert calls[0]["muted"] is False
    assert calls[0]["playsInline"] is True
    assert calls[0]["userActive"] is True
    assert listener_page.evaluate("window.__ttsEvents[0]") == "play"
    assert listener_page.locator("#resumeListening").is_hidden()


def test_listener_keeps_hls_source_and_retries_after_ios_play_rejection(listener_page):
    _install_hls_harness(listener_page, reject_first_play=True)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Tap to continue'"
    )
    original_src = listener_page.eval_on_selector("#ttsPlayback", "node => node.src")
    assert original_src.endswith("/index.m3u8")
    assert listener_page.locator("#resumeListening").is_visible()

    listener_page.locator("#resumeListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Connected'"
    )
    calls = listener_page.evaluate("window.__ttsPlayCalls.slice()")
    assert len(calls) == 2
    assert calls[0]["src"] == calls[1]["src"] == original_src
    assert listener_page.locator("#resumeListening").is_hidden()


def test_listener_stop_releases_only_its_hls_lease(listener_page):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)
    stream_src = listener_page.eval_on_selector("#ttsPlayback", "node => node.src")
    listener_id = stream_src.split("/api/tts/live/", 1)[1].split("/", 1)[0]

    listener_page.locator("#stopListening").click()

    delete_calls = listener_page.evaluate(
        "window.__ttsFetchCalls.filter(call => call.method === 'DELETE')"
    )
    assert delete_calls == [
        {"url": f"/api/tts/live/{listener_id}", "method": "DELETE"}
    ]
    assert listener_page.eval_on_selector("#ttsPlayback", "node => node.src") == ""
    assert listener_page.evaluate("window.__ttsPauseCalls") == 1
    assert listener_page.text_content("#connectionStatus") == "Stopped"


def test_listener_registers_lock_screen_media_session_controls(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")

    media = listener_page.evaluate(
        """({
          title: window.__ttsMediaSession.metadata.title,
          artist: window.__ttsMediaSession.metadata.artist,
          actions: Object.keys(window.__ttsMediaActions).sort(),
        })"""
    )
    assert media == {
        "title": "PCCS Live Translation",
        "artist": "Pittsburgh Christian Church South",
        "actions": ["pause", "play"],
    }
