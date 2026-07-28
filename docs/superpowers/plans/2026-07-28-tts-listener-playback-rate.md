# Per-Device TTS Listener Playback Rate Implementation Plan

> **For the current-session executor:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task. Do not use subagents; the user has
> explicitly prohibited them.

**Goal:** Add a persistent per-device playback-rate selector to `/listen` without
changing backend synthesis, broadcast caching, or FIFO semantics.

**Architecture:** Replace the listener page's one-shot Web Audio source with one
persistent `HTMLAudioElement` whose `playbackRate` can change during playback. Store an
allowlisted rate in browser `localStorage`; all HTTP and WebSocket contracts remain
unchanged and every listener continues fetching the same shared WAV.

**Tech Stack:** Embedded HTML/CSS/JavaScript, HTMLMediaElement, Python 3.12, pytest,
Playwright with system Google Chrome, FastAPI user service on port 8024.

---

## File Map

- Modify `voxbridge/tts/listener_page.py`: speed control, persistence, persistent media
  element, pitch preservation, active playback updates, and object URL cleanup.
- Modify `tests/test_demo_streaming_ws_utils.py`: static listener-page contract tests.
- Create `tests/test_tts_listener_page_browser.py`: real-browser preference, active
  media-rate, and narrow-screen layout coverage, skipped only when Playwright or Chrome
  is unavailable.
- Modify `tests/test_release_docs.py`, `README.md`, `docs/API.md`, and
  `docs/DEPLOYMENT.md`: user and operator contract.

## Invariants

- Rate choices are exactly `0.8`, `0.9`, `1.0`, `1.1`, and `1.2`; invalid storage
  falls back to `1.0`.
- The speed value remains local and never enters WebSocket or HTTP requests.
- Changing speed updates the active media element immediately.
- Stop, disconnect, and reload preserve the preference.
- One fetched WAV, one `tts_received`, and one FIFO dequeue remain unchanged per job.
- Every object URL is revoked after completion, failure, stop, or disconnect.
- No systemd arguments, backend `--tts-speed`, authentication, or port change.

## Task 1: Playback-Rate UI And Local Preference

**Files:**
- Modify: `tests/test_demo_streaming_ws_utils.py:1791-1811`
- Create: `tests/test_tts_listener_page_browser.py`
- Modify: `voxbridge/tts/listener_page.py:172-281`

- [ ] **Step 1: Write failing template-contract tests**

Append focused assertions:

```python
def test_listener_page_exposes_allowlisted_per_device_playback_rates():
    assert 'id="playbackRate"' in TTS_LISTENER_HTML
    for value in ("0.8", "0.9", "1", "1.1", "1.2"):
        assert f'<option value="{value}"' in TTS_LISTENER_HTML
    assert 'const PLAYBACK_RATE_STORAGE_KEY = "voxbridge.ttsPlaybackRate";' in TTS_LISTENER_HTML
    assert "const SUPPORTED_PLAYBACK_RATES = new Set([0.8, 0.9, 1, 1.1, 1.2]);" in TTS_LISTENER_HTML


def test_listener_page_normalizes_and_persists_playback_rate_locally():
    assert "function normalizePlaybackRate(value)" in TTS_LISTENER_HTML
    assert "return SUPPORTED_PLAYBACK_RATES.has(parsed) ? parsed : 1;" in TTS_LISTENER_HTML
    assert "window.localStorage.getItem(PLAYBACK_RATE_STORAGE_KEY)" in TTS_LISTENER_HTML
    assert "window.localStorage.setItem(PLAYBACK_RATE_STORAGE_KEY, String(playbackRate))" in TTS_LISTENER_HTML
    assert 'playbackRateInput.addEventListener("change"' in TTS_LISTENER_HTML
    assert "send({ type: \"set_playback_rate\"" not in TTS_LISTENER_HTML
```

Create `tests/test_tts_listener_page_browser.py` with a browser fixture and the
preference/layout tests before changing the page:

```python
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


def test_listener_rate_control_fits_mobile_without_horizontal_overflow(listener_page):
    listener_page.set_viewport_size({"width": 390, "height": 844})
    listener_page.goto("https://voxbridge.test/listen")
    assert listener_page.locator("#playbackRate").is_visible()
    assert listener_page.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth"
    ) is True
    assert listener_page.locator("#startListening").is_visible()
    assert listener_page.locator("#stopListening").is_visible()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_exposes_allowlisted_per_device_playback_rates \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_normalizes_and_persists_playback_rate_locally \
  tests/test_tts_listener_page_browser.py
```

Expected: the static tests and both browser tests fail because the select and storage
functions do not exist.

- [ ] **Step 3: Add the accessible control and preference state**

Add a `.playback-settings` row above `.actions`, with this control:

```html
<div class="playback-settings">
  <label for="playbackRate">朗读速度</label>
  <select id="playbackRate" aria-label="朗读速度">
    <option value="0.8">0.8x</option>
    <option value="0.9">0.9x</option>
    <option value="1" selected>1.0x</option>
    <option value="1.1">1.1x</option>
    <option value="1.2">1.2x</option>
  </select>
</div>
```

Style it as a compact panel using the page's existing colors and stack it cleanly under
`600px`. Add the local state before socket state:

```javascript
const playbackRateInput = document.getElementById("playbackRate");
const PLAYBACK_RATE_STORAGE_KEY = "voxbridge.ttsPlaybackRate";
const SUPPORTED_PLAYBACK_RATES = new Set([0.8, 0.9, 1, 1.1, 1.2]);

function normalizePlaybackRate(value) {
  const parsed = Number(value);
  return SUPPORTED_PLAYBACK_RATES.has(parsed) ? parsed : 1;
}

function readPlaybackRate() {
  try {
    return normalizePlaybackRate(window.localStorage.getItem(PLAYBACK_RATE_STORAGE_KEY));
  } catch (error) {
    return 1;
  }
}

let playbackRate = readPlaybackRate();
playbackRateInput.value = String(playbackRate);
playbackRateInput.addEventListener("change", () => {
  playbackRate = normalizePlaybackRate(playbackRateInput.value);
  playbackRateInput.value = String(playbackRate);
  try {
    window.localStorage.setItem(PLAYBACK_RATE_STORAGE_KEY, String(playbackRate));
  } catch (error) {}
});
```

`applyPlaybackRate()` and the call that updates active audio are introduced under a
failing media test in Task 2. Do not add any network message in this task.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: `2 passed`.

- [ ] **Step 5: Commit the UI preference slice**

```bash
git add voxbridge/tts/listener_page.py tests/test_demo_streaming_ws_utils.py \
  tests/test_tts_listener_page_browser.py
git commit -m "feat: add local TTS playback rate preference"
```

## Task 2: Persistent Pitch-Preserving Media Playback

**Files:**
- Modify: `tests/test_demo_streaming_ws_utils.py:1791-1835`
- Modify: `tests/test_tts_listener_page_browser.py`
- Modify: `voxbridge/tts/listener_page.py:243-502`

- [ ] **Step 1: Write failing media-lifecycle tests**

Add:

```python
def test_listener_page_applies_rate_to_persistent_pitch_preserving_audio():
    assert 'id="ttsPlayback"' in TTS_LISTENER_HTML
    assert "playbackElement.defaultPlaybackRate = playbackRate;" in TTS_LISTENER_HTML
    assert "playbackElement.playbackRate = playbackRate;" in TTS_LISTENER_HTML
    assert '"preservesPitch" in playbackElement' in TTS_LISTENER_HTML
    assert '"mozPreservesPitch" in playbackElement' in TTS_LISTENER_HTML
    assert '"webkitPreservesPitch" in playbackElement' in TTS_LISTENER_HTML


def test_listener_page_uses_one_media_element_and_releases_object_urls():
    assert 'new Blob([buffer], { type: "audio/wav" })' in TTS_LISTENER_HTML
    assert "window.URL.createObjectURL(audioBlob)" in TTS_LISTENER_HTML
    assert "window.URL.revokeObjectURL(activeObjectUrl);" in TTS_LISTENER_HTML
    assert 'playbackElement.addEventListener("ended"' in TTS_LISTENER_HTML
    assert "playbackElement.pause();" in TTS_LISTENER_HTML
    assert "sourceNode" not in TTS_LISTENER_HTML
    assert "createBufferSource" not in TTS_LISTENER_HTML


def test_listener_page_unlocks_media_before_opening_listener_socket():
    assert "const SILENT_WAV_DATA_URL =" in TTS_LISTENER_HTML
    assert "async function unlockPlaybackElement()" in TTS_LISTENER_HTML
    assert "await unlockPlaybackElement();" in TTS_LISTENER_HTML
    assert TTS_LISTENER_HTML.index("await unlockPlaybackElement();") < TTS_LISTENER_HTML.index(
        'new WebSocket(wsUrl("/ws/tts"))'
    )
```

Update the existing stop test to assert `stopActivePlayback();` rather than
`sourceNode.stop();`.

Add a real-browser assertion to `tests/test_tts_listener_page_browser.py`:

```python
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
```

- [ ] **Step 2: Run the lifecycle tests and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_applies_rate_to_persistent_pitch_preserving_audio \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_uses_one_media_element_and_releases_object_urls \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_unlocks_media_before_opening_listener_socket \
  tests/test_demo_streaming_ws_utils.py::test_listener_page_fetches_only_the_fifo_head_and_stops_locally \
  tests/test_tts_listener_page_browser.py::test_rate_change_updates_persistent_media_element
```

Expected: failures show the old `AudioBufferSourceNode` implementation.

- [ ] **Step 3: Add one persistent media element and rate application**

Add `<audio id="ttsPlayback" preload="auto" hidden></audio>` and replace
`audioContext`/`sourceNode` with:

```javascript
const playbackElement = document.getElementById("ttsPlayback");
let activeObjectUrl = "";
let cancelActivePlayback = null;

function applyPlaybackRate() {
  playbackElement.defaultPlaybackRate = playbackRate;
  playbackElement.playbackRate = playbackRate;
  if ("preservesPitch" in playbackElement) playbackElement.preservesPitch = true;
  if ("mozPreservesPitch" in playbackElement) playbackElement.mozPreservesPitch = true;
  if ("webkitPreservesPitch" in playbackElement) playbackElement.webkitPreservesPitch = true;
}

function releaseActiveObjectUrl() {
  if (!activeObjectUrl) return;
  window.URL.revokeObjectURL(activeObjectUrl);
  activeObjectUrl = "";
}

function stopActivePlayback() {
  if (cancelActivePlayback) {
    const cancel = cancelActivePlayback;
    cancelActivePlayback = null;
    cancel();
  }
  playbackElement.pause();
  playbackElement.removeAttribute("src");
  playbackElement.load();
  releaseActiveObjectUrl();
}
```

Call `applyPlaybackRate()` once after state initialization and from the existing
`playbackRateInput` change handler after persistence.

- [ ] **Step 4: Unlock the persistent element from the explicit Start gesture**

Add a valid one-sample WAV data URI and an unlock function:

```javascript
const SILENT_WAV_DATA_URL =
  "data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA==";

async function unlockPlaybackElement() {
  playbackElement.muted = true;
  playbackElement.src = SILENT_WAV_DATA_URL;
  try {
    await playbackElement.play();
  } finally {
    playbackElement.pause();
    playbackElement.muted = false;
    playbackElement.removeAttribute("src");
    playbackElement.load();
    applyPlaybackRate();
  }
}
```

Call `resetLocalPlayback()` and then `await unlockPlaybackElement()` inside
`startListening()` before constructing the WebSocket. A rejected unlock flows through
the existing Start failure handler and does not subscribe the device.

- [ ] **Step 5: Replace buffered-source playback with cancellable media playback**

Use one Blob/object URL per FIFO head:

```javascript
async function playAudioBuffer(buffer, localGeneration) {
  if (localGeneration !== generation) return;
  const audioBlob = new Blob([buffer], { type: "audio/wav" });
  activeObjectUrl = window.URL.createObjectURL(audioBlob);
  playbackElement.src = activeObjectUrl;
  applyPlaybackRate();
  try {
    await new Promise((resolve, reject) => {
      let settled = false;
      const settle = (error) => {
        if (settled) return;
        settled = true;
        playbackElement.removeEventListener("ended", onEnded);
        playbackElement.removeEventListener("error", onError);
        cancelActivePlayback = null;
        if (error) reject(error); else resolve();
      };
      const onEnded = () => settle();
      const onError = () => settle(new Error("audio playback failed"));
      cancelActivePlayback = () => settle(new DOMException("playback stopped", "AbortError"));
      playbackElement.addEventListener("ended", onEnded, { once: true });
      playbackElement.addEventListener("error", onError, { once: true });
      const playPromise = playbackElement.play();
      if (playPromise) playPromise.catch(onError);
    });
  } finally {
    playbackElement.removeAttribute("src");
    playbackElement.load();
    releaseActiveObjectUrl();
  }
}
```

`resetLocalPlayback()` increments `generation`, aborts a fetch, calls
`stopActivePlayback()`, and then clears the queue. `startListening()` keeps the existing
explicit user gesture by invoking `playbackElement.load()` and `applyPlaybackRate()`
before it opens `/ws/tts`; it must not create a second audio element.

- [ ] **Step 6: Run listener template and browser tests and verify GREEN**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py -k 'listener_page'
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_listener_page_browser.py
```

Expected: all selected listener tests pass.

- [ ] **Step 7: Commit the media playback slice**

```bash
git add voxbridge/tts/listener_page.py tests/test_demo_streaming_ws_utils.py \
  tests/test_tts_listener_page_browser.py
git commit -m "feat: apply TTS rate during listener playback"
```

## Task 3: Public Contract And Full Verification

**Files:**
- Modify: `tests/test_release_docs.py`
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`

- [ ] **Step 1: Add failing documentation assertions**

Extend `test_public_docs_describe_optional_kokoro_tts_contract()`:

```python
assert "0.8x" in readme
assert "1.2x" in readme
assert "per-device" in api.lower()
assert "localStorage" in api
assert "--tts-speed" in deployment
assert "listener-side" in deployment.lower()
```

- [ ] **Step 2: Run the release-doc test and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_release_docs.py::test_public_docs_describe_optional_kokoro_tts_contract
```

Expected: failure because the per-device playback contract is undocumented.

- [ ] **Step 3: Document the behavior without changing the protocol**

Document that `/listen` supports `0.8x` through `1.2x` in `0.1x` steps, persists the preference per
browser, applies it immediately, and does not change shared synthesis. In deployment
docs distinguish backend `--tts-speed` from the listener-side multiplier.

- [ ] **Step 4: Run documentation and focused feature tests**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_release_docs.py \
  tests/test_demo_streaming_ws_utils.py -k 'listener_page or release_docs'
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_listener_page_browser.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the public contract**

```bash
git add README.md docs/API.md docs/DEPLOYMENT.md tests/test_release_docs.py
git commit -m "docs: explain per-device TTS playback rate"
```

## Task 4: Integration And Port 8024 Deployment

**Files:**
- No systemd file changes.

- [ ] **Step 1: Run full verification from the feature worktree**

```bash
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
git diff --check
git status --short --branch
```

Expected: compile succeeds, the full suite passes, no diff errors, worktree clean.

- [ ] **Step 2: Audit the network boundary**

```bash
git diff main...HEAD -- voxbridge/tts/listener_page.py | rg "/ws/tts|api/tts|set_playback_rate|PLAYBACK_RATE_STORAGE_KEY"
```

Confirm existing URLs are unchanged and `set_playback_rate` does not exist.

- [ ] **Step 3: Fast-forward the verified branch into main**

From `/data/Qwen3-ASR/VoxBridge`, require a clean tree and unchanged merge base, then:

```bash
git merge --ff-only docs/tts-listener-playback-rate-design
```

- [ ] **Step 4: Rerun full verification from main**

```bash
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
```

Expected: the same passing count as the feature worktree.

- [ ] **Step 5: Restart only the managed user service**

```bash
systemctl --user restart voxbridge-8024.service
```

Wait for `8024` to listen. Do not launch the CLI manually.

- [ ] **Step 6: Verify production topology and HTTPS**

Confirm one `voxbridge.cli.demo_streaming_ws`, one `VLLM::EngineCore`, one `8024`
listener, `NRestarts=0`, HTTPS root `303`, `/login` `200`, and no traceback/error lines
since restart. Confirm the service still uses `.venv/bin/python` and retains
`--tts-speed 1.05` plus `--tts-revision-stable-sec 3.0`.

- [ ] **Step 7: Run Playwright against the deployed authenticated page**

Use an authenticated browser session without writing credentials into source or logs.
Verify the five options, select `1.2x`, reload, verify persistence, switch to a mobile
viewport, and confirm no horizontal overflow. Do not open an ASR WebSocket or a second
backend during this UI-only smoke test.

- [ ] **Step 8: Record final evidence**

Report the commit, exact test count, active PID, EngineCore count, `8024` listener,
HTTPS status, startup error count, and Playwright desktop/mobile results. Do not push
GitHub unless the user explicitly requests it.
