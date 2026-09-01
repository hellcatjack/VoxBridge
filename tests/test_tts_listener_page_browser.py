from __future__ import annotations

import json
import shutil

import pytest

from voxbridge.tts.listener_page import TTS_LISTENER_HTML

sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = sync_api.sync_playwright

LONG_CAPTION = (
    (
        "When the congregation had gathered together, the speaker explained that "
        "faithfulness in ordinary work, patient service to one another, and careful "
        "attention to the teaching of Scripture are not separate duties, but parts "
        "of the same calling that shapes the life of the whole church. "
    )
    * 4
)[:1000].rstrip()


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
    defer_caption_fetch: bool = False,
    defer_first_play: bool = False,
    native_hls: bool = True,
    hls_js_supported: bool = False,
    managed_media_source: bool = False,
    mse_aac_supported: bool = True,
    translated_audio_backlog_ms: int = 0,
    translated_audio_backlog_estimated: bool = False,
):
    snapshot = caption_snapshot or {"live_edge_at_ms": None, "cues": []}
    listener_page.add_init_script(
        f"""
        window.__ttsEvents = [];
        window.__ttsPlayCalls = [];
        window.__ttsSourceAssignments = [];
        window.__ttsPauseCalls = 0;
        window.__ttsFetchCalls = [];
        window.__ttsCaptionSnapshot = {json.dumps(snapshot)};
        window.__ttsDeferCaptionFetch = {json.dumps(defer_caption_fetch)};
        window.__ttsCaptionRequestAborted = false;
        window.__ttsDeferFirstPlay = {json.dumps(defer_first_play)};
        window.__ttsResolveFirstPlay = null;
        window.__ttsMediaActions = {{}};
        window.__ttsHlsInstances = [];
        const mediaSrcDescriptor = Object.getOwnPropertyDescriptor(
          HTMLMediaElement.prototype,
          "src"
        );
        Object.defineProperty(HTMLMediaElement.prototype, "src", {{
          configurable: true,
          enumerable: mediaSrcDescriptor.enumerable,
          get() {{ return mediaSrcDescriptor.get.call(this); }},
          set(value) {{
            if (this.id === "ttsPlayback") {{
              window.__ttsSourceAssignments.push({{
                playbackRate: this.playbackRate,
                defaultPlaybackRate: this.defaultPlaybackRate,
              }});
            }}
            mediaSrcDescriptor.set.call(this, value);
          }},
        }});
        if ({json.dumps(managed_media_source)}) {{
          window.ManagedMediaSource = class FakeManagedMediaSource {{}};
        }}
        if (window.MediaSource) {{
          window.MediaSource.isTypeSupported = function(type) {{
            return {json.dumps(mse_aac_supported)}
              && String(type).includes("mp4a.40.2");
          }};
        }}
        HTMLMediaElement.prototype.canPlayType = function(type) {{
          return {json.dumps(native_hls)} && String(type).includes("mpegurl")
            ? "maybe"
            : "";
        }};
        window.Hls = class FakeHls {{
          static isSupported() {{ return {json.dumps(hls_js_supported)}; }}
          constructor() {{
            this.loadedSources = [];
            this.attachedMedia = [];
            this.destroyed = false;
            this.playingDate = null;
            window.__ttsHlsInstances.push(this);
          }}
          loadSource(source) {{ this.loadedSources.push(String(source)); }}
          attachMedia(media) {{ this.attachedMedia.push(media.id); }}
          destroy() {{ this.destroyed = true; }}
          on() {{}}
        }};
        window.Hls.Events = {{ ERROR: "hlsError" }};
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
            playbackRate: this.playbackRate,
            defaultPlaybackRate: this.defaultPlaybackRate,
            userActive: navigator.userActivation
              ? navigator.userActivation.isActive
              : true,
          }};
          window.__ttsPlayCalls.push(call);
          window.__ttsEvents.push("play");
          if ({str(reject_first_play).lower()} && window.__ttsPlayCalls.length === 1) {{
            return Promise.reject(new DOMException("blocked", "NotAllowedError"));
          }}
          if (window.__ttsDeferFirstPlay && window.__ttsPlayCalls.length === 1) {{
            return new Promise(resolve => {{
              window.__ttsResolveFirstPlay = resolve;
            }});
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
          if (window.__ttsDeferCaptionFetch && call.url.includes("/captions")) {{
            return new Promise((resolve, reject) => {{
              if (!options.signal) return;
              options.signal.addEventListener("abort", () => {{
                window.__ttsCaptionRequestAborted = true;
                reject(new DOMException("aborted", "AbortError"));
              }}, {{ once: true }});
            }});
          }}
          const payload = call.url.includes("/captions")
            ? window.__ttsCaptionSnapshot
            : {{
                available: true,
                listener_count: 2,
                queue_depth: 0,
                pending_audio_ms: 6500,
                translated_audio_backlog_ms: {int(translated_audio_backlog_ms)},
                translated_audio_backlog_count: 0,
                translated_audio_backlog_estimated: {json.dumps(translated_audio_backlog_estimated)},
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


def _clear_live_lag(listener_page):
    listener_page.evaluate(
        """() => {
          const playback = document.querySelector("#ttsPlayback");
          Object.defineProperty(playback, "seekable", {
            configurable: true,
            get: () => ({
              length: 0,
              start: () => { throw new DOMException("no seekable range"); },
              end: () => { throw new DOMException("no seekable range"); },
            }),
          });
          playback.dispatchEvent(new Event("progress"));
        }"""
    )


def _set_buffered_range(listener_page, *, start: float, end: float):
    listener_page.evaluate(
        """({ start, end }) => {
          const playback = document.querySelector("#ttsPlayback");
          Object.defineProperty(playback, "buffered", {
            configurable: true,
            get: () => ({
              length: 1,
              start: () => start,
              end: () => end,
            }),
          });
          playback.dispatchEvent(new Event("timeupdate"));
        }""",
        {"start": start, "end": end},
    )


def test_listener_rate_selection_persists_after_reload(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.4")
    listener_page.reload()
    assert listener_page.input_value("#playbackRate") == "1.4"


def test_listener_defaults_to_auto_without_saved_preference(listener_page):
    listener_page.goto("https://voxbridge.test/listen")

    assert listener_page.input_value("#playbackRate") == "auto"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_listener_persists_explicit_auto_selection(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    listener_page.select_option("#playbackRate", "auto")
    listener_page.reload()

    assert listener_page.input_value("#playbackRate") == "auto"


def test_auto_rate_uses_ten_thirty_and_forty_second_boundaries(
    listener_page,
):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)
    _set_buffered_range(listener_page, start=0, end=105)

    assert listener_page.input_value("#playbackRate") == "auto"
    for lag, expected_rate, expected_status in (
        (0, 1, "Live translation · Auto 1.0×"),
        (9.99, 1, "Live translation · Auto 1.0×"),
        (
            10,
            1.2,
            "About 10s of translated audio waiting · "
            "about 9s estimated catch-up · Auto 1.2×",
        ),
        (
            29.99,
            1.2,
            "About 30s of translated audio waiting · "
            "about 25s estimated catch-up · Auto 1.2×",
        ),
        (
            30,
            1.4,
            "About 30s of translated audio waiting · "
            "about 22s estimated catch-up · Auto 1.4×",
        ),
        (
            39.99,
            1.4,
            "About 40s of translated audio waiting · "
            "about 29s estimated catch-up · Auto 1.4×",
        ),
        (
            40,
            1.5,
            "About 40s of translated audio waiting · "
            "about 27s estimated catch-up · Auto 1.5×",
        ),
        (
            60,
            1.5,
            "About 1m of translated audio waiting · "
            "about 40s estimated catch-up · Auto 1.5×",
        ),
        (
            60.01,
            1.5,
            "About 1m 1s of translated audio waiting · "
            "about 41s estimated catch-up · Auto 1.5×",
        ),
    ):
        _set_live_lag(listener_page, current_time=100 - lag, live_edge=100)
        assert listener_page.eval_on_selector(
            "#ttsPlayback", "node => node.playbackRate"
        ) == expected_rate
        assert listener_page.text_content("#playbackStatus") == expected_status


def test_live_audio_adds_server_backlog_without_changing_auto_rate(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    _start_hls_harness(listener_page)
    _set_buffered_range(listener_page, start=0, end=105)

    _set_live_lag(listener_page, current_time=79, live_edge=100)

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5
    assert listener_page.text_content("#playbackStatus") == (
        "About 1m 22s of translated audio waiting · "
        "about 55s estimated catch-up · Auto 1.5×"
    )


def test_auto_rate_uses_total_translated_audio_backlog(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    _start_hls_harness(listener_page)
    _set_buffered_range(listener_page, start=0, end=102)

    _set_live_lag(listener_page, current_time=98, live_edge=100)

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5
    assert listener_page.text_content("#playbackStatus") == (
        "About 1m 3s of translated audio waiting · "
        "about 42s estimated catch-up · Auto 1.5×"
    )


def test_auto_rate_uses_native_hls_seekable_headroom(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=98, live_edge=104)

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5


def test_auto_fast_rate_uses_forward_buffer_hysteresis(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=100, live_edge=100)
    _set_buffered_range(listener_page, start=0, end=103.99)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1

    _set_buffered_range(listener_page, start=0, end=104)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5

    _set_buffered_range(listener_page, start=0, end=102.01)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5

    _set_buffered_range(listener_page, start=0, end=102)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_auto_rate_describes_unknown_position_as_preparing(listener_page):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)

    assert listener_page.text_content("#playbackStatus") == "Preparing live translation"


def test_unsupported_legacy_rate_falls_back_to_default(listener_page):
    listener_page.add_init_script(
        "window.localStorage.setItem('voxbridge.ttsPlaybackRate', '1.5');"
    )
    listener_page.goto("https://voxbridge.test/listen")
    assert listener_page.input_value("#playbackRate") == "auto"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def _start_shared_timeline_gap_harness(
    listener_page,
    *,
    gap_ms=3000,
    playing_at_ms=104_100,
    playback_rate="auto",
):
    _install_hls_harness(
        listener_page,
        native_hls=False,
        hls_js_supported=True,
        caption_snapshot={
            "live_edge_at_ms": 120_000,
            "cues": [
                {
                    "cue_id": "spoken-before-gap",
                    "start_at_ms": 100_000,
                    "end_at_ms": 104_000,
                    "text": "The sentence before the historical silence.",
                },
                {
                    "cue_id": "spoken-after-gap",
                    "start_at_ms": 104_000 + gap_ms,
                    "end_at_ms": 110_000 + gap_ms,
                    "text": "The next translated sentence is already buffered.",
                },
            ],
        },
    )
    if playback_rate != "auto":
        listener_page.goto("https://voxbridge.test/listen")
        listener_page.select_option("#playbackRate", playback_rate)
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=50.0, live_edge=80.0)
    listener_page.evaluate(
        f"window.__ttsHlsInstances[0].playingDate = new Date({playing_at_ms})"
    )


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


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1440, 900),
        (844, 390),
        (390, 844),
        (320, 568),
    ],
)
def test_status_cards_share_available_width_evenly(
    listener_page,
    width,
    height,
):
    listener_page.set_viewport_size({"width": width, "height": height})
    listener_page.goto("https://voxbridge.test/listen")

    widths = listener_page.evaluate(
        """() => ({
          connection: document.querySelector("#connectionCard").getBoundingClientRect().width,
          service: document.querySelector("#producerCard").getBoundingClientRect().width,
          listeners: document.querySelector("#queueCard").getBoundingClientRect().width,
        })"""
    )

    assert max(widths.values()) - min(widths.values()) <= 1


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (390, 844)],
)
def test_live_audio_status_stays_anchored_to_card_bottom_when_text_changes(
    listener_page,
    width,
    height,
):
    listener_page.set_viewport_size({"width": width, "height": height})
    listener_page.goto("https://voxbridge.test/listen")

    positions = listener_page.evaluate(
        """() => {
          const card = document.querySelector('#nowPlaying');
          const caption = document.querySelector('#liveCaption');
          const status = document.querySelector('#playbackStatus');
          caption.textContent = 'A stable translated sentence is being read aloud.';

          const measure = (text) => {
            status.textContent = text;
            const cardRect = card.getBoundingClientRect();
            const statusRect = status.getBoundingClientRect();
            const paddingBottom = Number.parseFloat(
              window.getComputedStyle(card).paddingBottom
            );
            return {
              bottom: statusRect.bottom,
              height: statusRect.height,
              insetFromContentBottom:
                cardRect.bottom - paddingBottom - statusRect.bottom,
            };
          };

          return {
            short: measure('Live translation'),
            long: measure(
              'Translated audio backlog 39.9s · Auto 1.4× · Catch-up about 28.5s'
            ),
          };
        }"""
    )

    if width < 600:
        assert positions["long"]["height"] > positions["short"]["height"] + 1
    assert abs(positions["short"]["bottom"] - positions["long"]["bottom"]) <= 1
    assert abs(positions["short"]["insetFromContentBottom"]) <= 1
    assert abs(positions["long"]["insetFromContentBottom"]) <= 1


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (390, 844), (844, 390), (320, 568)],
)
def test_listener_fits_complete_long_caption_inside_live_audio_card(
    listener_page,
    width,
    height,
):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 1_000_000,
            "cues": [
                {
                    "cue_id": "long-caption",
                    "text": LONG_CAPTION,
                    "start_at_ms": 990_000,
                    "end_at_ms": 999_000,
                }
            ],
        },
    )
    listener_page.set_viewport_size({"width": width, "height": height})
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=95, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.startsWith('When the congregation')"
    )

    metrics = listener_page.evaluate(
        """() => {
          const caption = document.querySelector('#liveCaption');
          const card = document.querySelector('#nowPlaying');
          const controls = document.querySelector('.controls');
          const cardRect = card.getBoundingClientRect();
          const controlsRect = controls.getBoundingClientRect();
          return {
            captionClipped: caption.scrollHeight > caption.clientHeight + 1,
            cardClipped: card.scrollHeight > card.clientHeight + 1,
            controlsOverlap: cardRect.bottom > controlsRect.top + 1,
            controlsOutsideViewport:
              controlsRect.left < 0
              || controlsRect.right > window.innerWidth + 1
              || controlsRect.top < 0
              || controlsRect.bottom > window.innerHeight + 1,
            documentOverflow:
              document.documentElement.scrollHeight > window.innerHeight
              || document.documentElement.scrollWidth > window.innerWidth,
          };
        }"""
    )
    assert listener_page.text_content("#liveCaption") == LONG_CAPTION
    assert metrics == {
        "captionClipped": False,
        "cardClipped": False,
        "controlsOverlap": False,
        "controlsOutsideViewport": False,
        "documentOverflow": False,
    }


def test_listener_refits_long_caption_after_device_rotation(listener_page):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 1_000_000,
            "cues": [
                {
                    "cue_id": "rotated-long-caption",
                    "text": LONG_CAPTION,
                    "start_at_ms": 990_000,
                    "end_at_ms": 999_000,
                }
            ],
        },
    )
    listener_page.set_viewport_size({"width": 390, "height": 844})
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=95, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.startsWith('When the congregation')"
    )

    listener_page.set_viewport_size({"width": 844, "height": 390})
    listener_page.wait_for_function(
        """() => {
          const caption = document.querySelector('#liveCaption');
          const card = document.querySelector('#nowPlaying');
          const controls = document.querySelector('.controls');
          const cardRect = card.getBoundingClientRect();
          const controlsRect = controls.getBoundingClientRect();
          return caption.scrollHeight <= caption.clientHeight + 1
            && card.scrollHeight <= card.clientHeight + 1
            && cardRect.bottom <= controlsRect.top + 1;
        }"""
    )
    assert listener_page.text_content("#liveCaption") == LONG_CAPTION


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


def test_listener_starts_fast_selection_at_normal_rate(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")

    _start_hls_harness(listener_page)

    first_play = listener_page.evaluate("window.__ttsPlayCalls[0]")
    source_assignment = listener_page.evaluate("window.__ttsSourceAssignments[0]")
    assert listener_page.input_value("#playbackRate") == "1.2"
    assert source_assignment["playbackRate"] == 1
    assert source_assignment["defaultPlaybackRate"] == 1
    assert first_play["playbackRate"] == 1
    assert first_play["defaultPlaybackRate"] == 1
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_fast_selection_only_runs_with_safe_live_lag(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=94, live_edge=100)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2

    _set_live_lag(listener_page, current_time=99, live_edge=100)
    assert listener_page.input_value("#playbackRate") == "1.2"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_1_4_selection_starts_with_two_seconds_of_safe_buffer(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.4")
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=98, live_edge=100)

    assert listener_page.input_value("#playbackRate") == "1.4"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.4


def test_1_4_selection_is_not_capped_by_automatic_catch_up(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.4")
    _start_hls_harness(listener_page)

    _set_live_lag(listener_page, current_time=80, live_edge=100)

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.4
    assert "translated audio waiting" in listener_page.text_content("#playbackStatus")


def test_fast_selection_falls_back_when_live_lag_is_unknown(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2

    _clear_live_lag(listener_page)

    assert listener_page.input_value("#playbackRate") == "1.2"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1


def test_expected_sentence_gap_preserves_fast_selection(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    _set_buffered_range(listener_page, start=0, end=94.4)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2

    listener_page.locator("#ttsPlayback").dispatch_event("waiting")

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2
    assert listener_page.text_content("#playbackStatus") == (
        "Preparing next translated sentence"
    )


def test_sentence_gap_status_clears_as_soon_as_playback_resumes(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        translated_audio_backlog_estimated=True,
    )
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    _set_buffered_range(listener_page, start=0, end=94.4)

    playback = listener_page.locator("#ttsPlayback")
    playback.dispatch_event("waiting")
    assert listener_page.text_content("#playbackStatus") == (
        "Preparing next translated sentence"
    )

    playback.dispatch_event("playing")
    _set_live_lag(listener_page, current_time=94.1, live_edge=100)

    assert "translated audio waiting" in listener_page.text_content(
        "#playbackStatus"
    )
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.5


def test_stalled_resets_fast_selection_to_normal_rate(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    _set_buffered_range(listener_page, start=0, end=94.4)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2

    listener_page.locator("#ttsPlayback").dispatch_event("stalled")

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1
    assert listener_page.text_content("#playbackStatus") == (
        "Reconnecting to live audio · 0.4s buffered ahead"
    )


def test_listener_start_stop_preserves_rate_selection(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "0.9")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === 'Connected'"
    )
    listener_page.select_option("#playbackRate", "1.1")
    assert listener_page.input_value("#playbackRate") == "1.1"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1
    listener_page.locator("#stopListening").click()
    assert listener_page.input_value("#playbackRate") == "1.1"
    assert listener_page.text_content("#connectionStatus") == "Stopped"


def test_listener_stop_normalizes_active_fast_media_rate(listener_page):
    _install_hls_harness(listener_page)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "1.2")
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.2

    listener_page.locator("#stopListening").click()

    assert listener_page.input_value("#playbackRate") == "1.2"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.defaultPlaybackRate"
    ) == 1


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
    assert listener_page.text_content("#playbackStatus") == (
        "About 20s of translated audio waiting · "
        "about 17s estimated catch-up · 1.2×"
    )
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


def test_listener_status_keeps_server_backlog_out_of_listener_count(listener_page):
    _install_hls_harness(listener_page)
    _start_hls_harness(listener_page)

    listener_page.wait_for_function(
        "document.querySelector('#queueStatus').textContent === 'Live · 2 listening'"
    )
    assert "queued" not in listener_page.text_content("#queueStatus")


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


def test_listener_uses_hls_start_date_when_server_playlist_is_ahead(listener_page):
    _install_hls_harness(
        listener_page,
        caption_snapshot={
            "live_edge_at_ms": 115_000,
            "cues": [
                {
                    "cue_id": "device-cue",
                    "start_at_ms": 103_000,
                    "end_at_ms": 105_000,
                    "text": "The sentence this device is actually playing.",
                },
                {
                    "cue_id": "server-cue",
                    "start_at_ms": 107_000,
                    "end_at_ms": 109_000,
                    "text": "The sentence only the server has reached.",
                },
            ],
        },
    )
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=94, live_edge=100)
    listener_page.locator("#ttsPlayback").evaluate(
        "node => { node.getStartDate = () => new Date(10_000); }"
    )
    listener_page.locator("#ttsPlayback").dispatch_event("timeupdate")

    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.includes('actually playing')"
    )
    assert listener_page.text_content("#liveCaption") == (
        "The sentence this device is actually playing."
    )


def test_desktop_caption_uses_hls_playing_date_when_playlist_edges_desynchronize(
    listener_page,
):
    _install_hls_harness(
        listener_page,
        native_hls=False,
        hls_js_supported=True,
        caption_snapshot={
            "live_edge_at_ms": 110_000,
            "cues": [
                {
                    "cue_id": "older-cue",
                    "start_at_ms": 103_000,
                    "end_at_ms": 106_000,
                    "text": "The older sentence.",
                },
                {
                    "cue_id": "newer-cue",
                    "start_at_ms": 107_000,
                    "end_at_ms": 110_000,
                    "text": "The sentence Chrome is actually playing.",
                },
            ],
        },
    )
    _start_hls_harness(listener_page)
    listener_page.locator("#ttsPlayback").evaluate(
        "node => { node.getStartDate = undefined; }"
    )
    listener_page.evaluate(
        "window.__ttsHlsInstances[0].playingDate = new Date(108500)"
    )
    _set_live_lag(listener_page, current_time=98, live_edge=100)
    listener_page.wait_for_function(
        "document.querySelector('#liveCaption').textContent.includes('actually playing')"
    )

    listener_page.evaluate(
        "window.__ttsCaptionSnapshot.live_edge_at_ms = 111000"
    )
    _set_live_lag(listener_page, current_time=99, live_edge=104)
    listener_page.wait_for_timeout(650)

    assert listener_page.text_content("#liveCaption") == (
        "The sentence Chrome is actually playing."
    )


def test_listener_keeps_shared_playhead_when_future_speech_is_buffered(listener_page):
    _start_shared_timeline_gap_harness(
        listener_page,
        gap_ms=3000,
        playing_at_ms=104_100,
    )

    _set_buffered_range(listener_page, start=0.0, end=80.0)

    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.currentTime"
    ) == pytest.approx(50.0)
    assert listener_page.get_attribute("#nowPlaying", "data-playing") == "true"


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
    listener_page.wait_for_function(
        "document.querySelector('#nowPlaying').dataset.speaking === 'true'"
    )
    listener_page.wait_for_timeout(250)
    speaking_style = listener_page.eval_on_selector(
        "#liveCaption",
        "node => ({ color: getComputedStyle(node).color, opacity: getComputedStyle(node).opacity })",
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
    listener_page.wait_for_timeout(250)
    assert listener_page.text_content("#liveCaption") == (
        "The first spoken sentence remains visible."
    )
    retained_style = listener_page.eval_on_selector(
        "#liveCaption",
        "node => ({ color: getComputedStyle(node).color, opacity: getComputedStyle(node).opacity })",
    )
    assert retained_style == speaking_style

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


def test_listener_stop_aborts_inflight_caption_request_before_releasing_lease(
    listener_page,
):
    _install_hls_harness(listener_page, defer_caption_fetch=True)
    _start_hls_harness(listener_page)
    listener_page.wait_for_function(
        "window.__ttsFetchCalls.some(call => call.url.includes('/captions'))"
    )

    listener_page.locator("#stopListening").click()

    listener_page.wait_for_function("window.__ttsCaptionRequestAborted === true")
    assert listener_page.evaluate("window.__ttsCaptionRequestAborted") is True


def test_listener_waits_for_hls_playback_before_polling_captions(listener_page):
    _install_hls_harness(listener_page, defer_first_play=True)
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function("typeof window.__ttsResolveFirstPlay === 'function'")

    assert listener_page.evaluate(
        "window.__ttsFetchCalls.some(call => call.url.includes('/captions'))"
    ) is False

    listener_page.evaluate("window.__ttsResolveFirstPlay()")
    listener_page.wait_for_function(
        "window.__ttsFetchCalls.some(call => call.url.includes('/captions'))"
    )


def test_listener_starts_one_unmuted_hls_stream_inside_click(listener_page):
    _install_hls_harness(
        listener_page,
        native_hls=True,
        hls_js_supported=True,
        managed_media_source=True,
    )
    _start_hls_harness(listener_page)

    calls = listener_page.evaluate("window.__ttsPlayCalls.slice()")
    assert len(calls) == 1
    assert "/api/tts/live/iphone-" in calls[0]["src"]
    assert calls[0]["src"].endswith("/index.m3u8")
    assert calls[0]["muted"] is False
    assert calls[0]["playsInline"] is True
    assert calls[0]["userActive"] is True
    assert listener_page.evaluate("window.__ttsEvents[0]") == "play"
    assert listener_page.evaluate("window.__ttsHlsInstances.length") == 0
    assert listener_page.locator("#resumeListening").is_hidden()


def test_desktop_listener_prefers_hls_js_when_native_support_is_unreliable(
    listener_page,
):
    _install_hls_harness(
        listener_page,
        native_hls=True,
        hls_js_supported=True,
        managed_media_source=False,
    )
    _start_hls_harness(listener_page)

    assert listener_page.evaluate("window.__ttsHlsInstances.length") == 1
    assert listener_page.evaluate(
        "window.__ttsHlsInstances[0].attachedMedia"
    ) == ["ttsPlayback"]


def test_desktop_listener_uses_hls_js_fallback_and_destroys_it_on_stop(listener_page):
    _install_hls_harness(
        listener_page,
        native_hls=False,
        hls_js_supported=True,
    )
    _start_hls_harness(listener_page)

    hls_state = listener_page.evaluate(
        """() => {
          const instance = window.__ttsHlsInstances[0];
          return {
            count: window.__ttsHlsInstances.length,
            loadedSources: instance.loadedSources,
            attachedMedia: instance.attachedMedia,
            destroyed: instance.destroyed,
          };
        }"""
    )
    assert hls_state["count"] == 1
    assert len(hls_state["loadedSources"]) == 1
    assert "/api/tts/live/iphone-" in hls_state["loadedSources"][0]
    assert hls_state["loadedSources"][0].endswith("/index.m3u8")
    assert hls_state["attachedMedia"] == ["ttsPlayback"]
    assert hls_state["destroyed"] is False

    listener_page.locator("#stopListening").click()
    assert listener_page.evaluate("window.__ttsHlsInstances[0].destroyed") is True


def test_listener_rejects_hls_js_when_browser_lacks_mse_aac_codec(listener_page):
    _install_hls_harness(
        listener_page,
        native_hls=False,
        hls_js_supported=True,
        mse_aac_supported=False,
    )
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.locator("#startListening").click()

    assert listener_page.evaluate("window.__ttsHlsInstances.length") == 0
    assert listener_page.locator("#connectionStatus").text_content() == (
        "Audio unavailable"
    )


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
