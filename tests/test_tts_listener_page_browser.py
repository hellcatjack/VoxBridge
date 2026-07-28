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
    listener_page.select_option("#playbackRate", "1.5")
    listener_page.reload()
    assert listener_page.input_value("#playbackRate") == "1.5"


def test_listener_rate_control_fits_mobile_without_horizontal_overflow(listener_page):
    listener_page.set_viewport_size({"width": 390, "height": 844})
    listener_page.goto("https://voxbridge.test/listen")
    assert listener_page.locator("#playbackRate").is_visible()
    assert listener_page.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth"
    ) is True
    assert listener_page.locator("#startListening").is_visible()
    assert listener_page.locator("#stopListening").is_visible()
