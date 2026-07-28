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
        yield page
        browser.close()


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
