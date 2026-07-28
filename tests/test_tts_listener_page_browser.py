from __future__ import annotations

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


def _install_queue_harness(listener_page, *, deferred_fetch: bool = False):
    listener_page.add_init_script(
        f"""
        window.__ttsPlayStarts = [];
        HTMLMediaElement.prototype.play = function() {{
          if (!this.muted && String(this.src).startsWith("blob:")) {{
            window.__ttsPlayStarts.push(performance.now());
          }}
          return Promise.resolve();
        }};
        HTMLMediaElement.prototype.pause = function() {{}};
        HTMLMediaElement.prototype.load = function() {{}};
        const nativeMediaAddEventListener = HTMLMediaElement.prototype.addEventListener;
        const nativeMediaRemoveEventListener = HTMLMediaElement.prototype.removeEventListener;
        const controlledEndedListeners = new Set();
        const controlledErrorListeners = new Set();
        HTMLMediaElement.prototype.addEventListener = function(type, listener, options) {{
          if (this.id === "ttsPlayback" && type === "ended") {{
            controlledEndedListeners.add(listener);
            return;
          }}
          if (this.id === "ttsPlayback" && type === "error") {{
            controlledErrorListeners.add(listener);
            return;
          }}
          return nativeMediaAddEventListener.call(this, type, listener, options);
        }};
        HTMLMediaElement.prototype.removeEventListener = function(type, listener, options) {{
          if (this.id === "ttsPlayback" && type === "ended") {{
            controlledEndedListeners.delete(listener);
            return;
          }}
          if (this.id === "ttsPlayback" && type === "error") {{
            controlledErrorListeners.delete(listener);
            return;
          }}
          return nativeMediaRemoveEventListener.call(this, type, listener, options);
        }};
        window.__finishTTSPlayback = function() {{
          const event = new Event("ended");
          for (const listener of Array.from(controlledEndedListeners)) {{
            listener.call(document.querySelector("#ttsPlayback"), event);
          }}
        }};
        window.__ttsFetchCalls = [];
        window.__ttsSentMessages = [];
        window.__ttsAbortCount = 0;
        window.fetch = function(url, options = {{}}) {{
          const parts = String(url).split("/");
          const jobId = decodeURIComponent(parts[parts.length - 2]);
          window.__ttsFetchCalls.push(jobId);
          if ({str(deferred_fetch).lower()}) {{
            return new Promise((resolve, reject) => {{
              const signal = options.signal;
              const abort = () => {{
                window.__ttsAbortCount += 1;
                reject(new DOMException("fetch aborted", "AbortError"));
              }};
              if (signal.aborted) abort();
              else signal.addEventListener("abort", abort, {{ once: true }});
            }});
          }}
          return Promise.resolve({{
            ok: true,
            status: 200,
            arrayBuffer: () => Promise.resolve(new Uint8Array([82, 73, 70, 70]).buffer),
          }});
        }};
        window.WebSocket = class FakeWebSocket extends EventTarget {{
          static OPEN = 1;
          static CLOSING = 2;
          constructor() {{
            super();
            this.readyState = FakeWebSocket.OPEN;
            window.__ttsSocket = this;
            window.setTimeout(() => {{
              this.dispatchEvent(new Event("open"));
              this.dispatchEvent(new MessageEvent("message", {{
                data: JSON.stringify({{
                  type: "tts_listener_ready",
                  listener_id: "test-listener",
                  tts_available: true,
                  producer_active: true,
                }}),
              }}));
            }}, 0);
          }}
          emitJob(jobId, sourceOrder) {{
            this.dispatchEvent(new MessageEvent("message", {{
              data: JSON.stringify({{
                type: "tts_job",
                job_id: jobId,
                revision: 1,
                source_order: sourceOrder,
                target_language: "English",
                is_stable: true,
              }}),
            }}));
          }}
          send(payload) {{
            window.__ttsSentMessages.push(JSON.parse(payload));
          }}
          close() {{
            this.readyState = 3;
            this.dispatchEvent(new Event("close"));
          }}
        }};
        """
    )


def _start_queue_harness(listener_page):
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === '已连接'"
    )


def _emit_job(listener_page, job_id: str, source_order: int):
    listener_page.evaluate(
        "([jobId, sourceOrder]) => window.__ttsSocket.emitJob(jobId, sourceOrder)",
        [job_id, source_order],
    )


def _fetch_job_ids(listener_page):
    return listener_page.evaluate("window.__ttsFetchCalls.slice()")


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


def test_listener_rate_control_fits_mobile_without_horizontal_overflow(listener_page):
    listener_page.set_viewport_size({"width": 390, "height": 844})
    listener_page.goto("https://voxbridge.test/listen")
    assert listener_page.locator("#playbackRate").is_visible()
    assert listener_page.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth"
    ) is True
    assert listener_page.locator("#startListening").is_visible()
    assert listener_page.locator("#stopListening").is_visible()


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
    listener_page.add_init_script(
        """
        HTMLMediaElement.prototype.play = function() { return Promise.resolve(); };
        HTMLMediaElement.prototype.pause = function() {};
        window.WebSocket = class FakeWebSocket extends EventTarget {
          static OPEN = 1;
          static CLOSING = 2;
          constructor() {
            super();
            this.readyState = FakeWebSocket.OPEN;
            window.setTimeout(() => {
              this.dispatchEvent(new Event("open"));
              this.dispatchEvent(new MessageEvent("message", {
                data: JSON.stringify({
                  type: "tts_listener_ready",
                  listener_id: "test-listener",
                  tts_available: true,
                  producer_active: false,
                }),
              }));
            }, 0);
          }
          send() {}
          close() {
            this.readyState = 3;
            this.dispatchEvent(new Event("close"));
          }
        };
        """
    )
    listener_page.goto("https://voxbridge.test/listen")
    listener_page.select_option("#playbackRate", "0.9")
    listener_page.locator("#startListening").click()
    listener_page.wait_for_function(
        "document.querySelector('#connectionStatus').textContent === '已连接'"
    )
    listener_page.select_option("#playbackRate", "1.1")
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => node.playbackRate"
    ) == 1.1
    listener_page.locator("#stopListening").click()
    assert listener_page.input_value("#playbackRate") == "1.1"
    assert listener_page.text_content("#connectionStatus") == "已停止"


def test_listener_prefetches_exactly_one_future_fifo_item(listener_page):
    _install_queue_harness(listener_page)
    _start_queue_harness(listener_page)
    _emit_job(listener_page, "job-1", 0)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 1")
    assert _fetch_job_ids(listener_page) == ["job-1"]

    _emit_job(listener_page, "job-2", 1)
    _emit_job(listener_page, "job-3", 2)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 2")
    assert _fetch_job_ids(listener_page) == ["job-1", "job-2"]
    listener_page.wait_for_timeout(100)
    assert _fetch_job_ids(listener_page) == ["job-1", "job-2"]

    listener_page.evaluate("window.__finishTTSPlayback()")
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 3")
    assert _fetch_job_ids(listener_page) == ["job-1", "job-2", "job-3"]


def test_listener_stop_aborts_current_and_prefetched_audio(listener_page):
    _install_queue_harness(listener_page, deferred_fetch=True)
    _start_queue_harness(listener_page)
    _emit_job(listener_page, "job-1", 0)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 1")
    _emit_job(listener_page, "job-2", 1)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 2")

    listener_page.locator("#stopListening").click()
    listener_page.wait_for_function("window.__ttsAbortCount === 2")
    assert listener_page.evaluate("window.__ttsAbortCount") == 2


def test_listener_waits_for_sentence_pause_before_prepared_audio(listener_page):
    _install_queue_harness(listener_page)
    _start_queue_harness(listener_page)
    _emit_job(listener_page, "job-1", 0)
    listener_page.wait_for_function("window.__ttsPlayStarts.length === 1")
    _emit_job(listener_page, "job-2", 1)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 2")

    ended_at = listener_page.evaluate("performance.now()")
    listener_page.evaluate("window.__finishTTSPlayback()")
    listener_page.wait_for_function("window.__ttsPlayStarts.length === 2")
    second_started_at = listener_page.evaluate("window.__ttsPlayStarts[1]")
    assert second_started_at - ended_at >= 280


def test_listener_stop_cancels_active_sentence_pause_immediately(listener_page):
    _install_queue_harness(listener_page)
    _start_queue_harness(listener_page)
    _emit_job(listener_page, "job-1", 0)
    listener_page.wait_for_function("window.__ttsPlayStarts.length === 1")
    _emit_job(listener_page, "job-2", 1)
    listener_page.wait_for_function("window.__ttsFetchCalls.length === 2")

    listener_page.evaluate("window.__finishTTSPlayback()")
    listener_page.wait_for_timeout(50)
    assert listener_page.evaluate("window.__ttsPlayStarts.length") == 1
    stop_started_at = listener_page.evaluate("performance.now()")
    listener_page.locator("#stopListening").click()
    stop_finished_at = listener_page.evaluate("performance.now()")
    assert listener_page.text_content("#connectionStatus") == "已停止"
    assert stop_finished_at - stop_started_at < 200
