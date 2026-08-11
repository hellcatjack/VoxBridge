# HLS Live Audio Caption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the stable translated sentence that each `/listen` device is actually hearing, synchronized to that device's shared-HLS playhead.

**Architecture:** The FFmpeg encoder returns a wall-clock append receipt and parses the latest complete HLS segment edge. The shared publisher records bounded caption cues only when stable PCM is published. A listener-scoped API exposes cues and live edge; the existing page maps `seekable.end - currentTime` to the correct cue and retains the last cue between sentences.

**Tech Stack:** Python 3.12 via `../.venv/bin/python`, asyncio, FastAPI, FFmpeg HLS, vanilla HTML/CSS/JavaScript, pytest, Playwright Chromium.

## Global Constraints

- All Python commands run through `/data/Qwen3-ASR/.venv/bin/python` or `../.venv/bin/python`.
- The managed local service remains `voxbridge-8024.service` on port `8024`.
- Do not use subagents.
- Do not add per-device TTS synthesis, per-device audio queues, HLS seeking, or audio skipping.
- Caption polling must never gate native HLS audio or iPhone lock-screen playback.
- The fixed `300ms` sentence pause remains audible but is outside the active cue interval.
- Caption text is never split, punctuated, or classified with language-specific frontend rules.
- The listener remains public, English-only in its interface copy, one-screen, and scrollbar-free.
- Keep at most 256 caption cues for the active HLS epoch and clear them when the final listener leaves.
- Internal `docs/superpowers/` files remain local planning material and are not part of the public release tree.

---

## File Structure

- Modify `voxbridge/tts/hls.py`: append receipts, playlist live-edge parser, cue data types, bounded epoch cue store, caption snapshot.
- Modify `voxbridge/cli/demo_streaming_ws.py`: public listener-scoped caption endpoint and JSON serialization.
- Modify `voxbridge/tts/listener_page.py`: `Live Audio` caption markup, matching styles, per-device cue polling and playhead selection.
- Modify `tests/test_tts_hls.py`: receipt, live-edge, cue lifecycle, pause exclusion tests.
- Modify `tests/test_demo_streaming_ws_protocol.py`: caption endpoint lease and response contract tests.
- Modify `tests/test_tts_listener_page_browser.py`: browser synchronization, retention, replacement, stop, and viewport tests.
- Modify `docs/API.md`, `README.md`, `docs/DEPLOYMENT.md`, and `CHANGELOG.md`: public API and operations contract.

### Task 1: Encoder Append Receipts And HLS Live Edge

**Files:**
- Modify: `voxbridge/tts/hls.py`
- Test: `tests/test_tts_hls.py`

**Interfaces:**
- Produces: `HLSAppendReceipt(start_at_ms: int, end_at_ms: int)`.
- Produces: `parse_hls_live_edge_at_ms(playlist: str) -> int | None`.
- Changes: `HLSEncoder.append_pcm(pcm: bytes) -> HLSAppendReceipt | None`.
- Adds: `HLSEncoder.live_edge_at_ms() -> int | None`.

- [ ] **Step 1: Write failing append receipt tests**

Add deterministic wall-clock support and assert that two queued payloads receive
contiguous FIFO spans:

```python
@pytest.mark.asyncio
async def test_ffmpeg_append_receipts_follow_pending_pcm_fifo(tmp_path):
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=1000,
        wall_clock=lambda: 100.0,
    )
    encoder._process = SimpleNamespace(returncode=None)
    first = await encoder.append_pcm(bytes(1000 * 2))
    second = await encoder.append_pcm(bytes(500 * 2))
    assert first == HLSAppendReceipt(start_at_ms=100_000, end_at_ms=101_000)
    assert second == HLSAppendReceipt(start_at_ms=101_000, end_at_ms=101_500)
    encoder._process = None
```

- [ ] **Step 2: Run the receipt test and verify RED**

Run:

```bash
../.venv/bin/python -m pytest -q \
  tests/test_tts_hls.py::test_ffmpeg_append_receipts_follow_pending_pcm_fifo
```

Expected: FAIL because `HLSAppendReceipt` and `wall_clock` do not exist.

- [ ] **Step 3: Implement immutable receipts in the encoder**

Add:

```python
@dataclass(frozen=True, slots=True)
class HLSAppendReceipt:
    start_at_ms: int
    end_at_ms: int
```

After `await self._pcm_queue.put(data)`, calculate queued duration from the
existing `_pending_pcm_bytes`, increment pending bytes, and return the scheduled
span using injected `wall_clock=time.time`. Preserve current backpressure.

- [ ] **Step 4: Write failing playlist live-edge parser tests**

Cover multiple segments, timezone offsets, incomplete final entries, malformed
timestamps, and empty playlists. The complete final segment must win:

```python
def test_hls_live_edge_uses_last_complete_program_date_time_segment():
    playlist = """#EXTM3U
#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:00.000-04:00
#EXTINF:1.024,
segment_000000001.ts
#EXT-X-PROGRAM-DATE-TIME:2026-08-11T10:00:01.024-04:00
#EXTINF:0.512,
segment_000000002.ts
"""
    assert parse_hls_live_edge_at_ms(playlist) == 1786456801536
```

- [ ] **Step 5: Run parser tests and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_tts_hls.py -k 'live_edge'`.
Expected: FAIL because the parser is missing.

- [ ] **Step 6: Implement parser and encoder accessor**

Parse `EXT-X-PROGRAM-DATE-TIME`, `EXTINF`, and the following media URI as one
complete record. Use `datetime.fromisoformat`, require timezone-aware values,
return the last complete end in epoch milliseconds, and return `None` on absent
or invalid data. `FFmpegHLSEncoder.live_edge_at_ms()` delegates to the parser on
`playlist_text()` and catches `HLSUnavailable`.

- [ ] **Step 7: Run encoder tests GREEN**

Run:

```bash
../.venv/bin/python -m pytest -q tests/test_tts_hls.py \
  -k 'ffmpeg or live_edge or append_receipt'
```

Expected: all selected tests pass, including existing backpressure tests.

### Task 2: Bounded Epoch Caption Cues

**Files:**
- Modify: `voxbridge/tts/hls.py`
- Test: `tests/test_tts_hls.py`

**Interfaces:**
- Consumes: `HLSAppendReceipt` and `HLSEncoder.live_edge_at_ms()` from Task 1.
- Produces: `HLSCaptionCue(cue_id, start_at_ms, end_at_ms, text)`.
- Produces: `HLSCaptionSnapshot(live_edge_at_ms, cues)`.
- Produces: `SharedHLSTTSPublisher.caption_snapshot(listener_id, owner_key)`.

- [ ] **Step 1: Write failing stable-publication cue tests**

Make `FakeEncoder.append_pcm()` return deterministic receipts. Verify preparation
creates no cue, stable publish creates one, and speech end excludes the pause:

```python
snapshot = publisher.caption_snapshot("iphone-a", "owner-a")
assert snapshot.cues[0].text == "Prepared exact revision."
assert snapshot.cues[0].start_at_ms == 100_000
assert snapshot.cues[0].end_at_ms == 100_250
assert encoder.receipts[0].end_at_ms == 100_550
```

- [ ] **Step 2: Run cue test and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_tts_hls.py -k 'caption_cue'`.
Expected: FAIL because caption data types and snapshot are absent.

- [ ] **Step 3: Implement cue creation only after stable append**

Add frozen cue/snapshot dataclasses and a `deque(maxlen=256)`. In the release
worker, retain the receipt from `append_pcm`; clamp cue end to
`min(receipt.end_at_ms, receipt.start_at_ms + prepared.audio_ms)`. Generate
`cue_id` as the first 16 hexadecimal characters of SHA-256 over the active epoch,
sentence ID, revision, and receipt start. Do not log text.

- [ ] **Step 4: Write failing lifecycle tests**

Assert 257 releases retain only the newest 256 cues, a foreign lease raises
`HLSListenerNotFound`, and removing the last lease clears all cues before a new
epoch starts.

- [ ] **Step 5: Run lifecycle tests and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_tts_hls.py -k 'caption'`.
Expected: lifecycle assertions fail until reset and bounds are implemented.

- [ ] **Step 6: Implement snapshot validation, bounds, and epoch cleanup**

`caption_snapshot()` calls `_require_lease`, reads `encoder.live_edge_at_ms()`,
and returns an immutable tuple copy. `_stop_stream()` clears cues with prepared
audio and keys. A missing live edge returns `None` without changing cues.

- [ ] **Step 7: Run all publisher tests GREEN**

Run `../.venv/bin/python -m pytest -q tests/test_tts_hls.py`.
Expected: all HLS tests pass.

### Task 3: Listener-Scoped Caption API

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py`
- Modify: `tests/test_demo_streaming_ws_protocol.py`

**Interfaces:**
- Consumes: `SharedHLSTTSPublisher.caption_snapshot()` from Task 2.
- Produces: `GET /api/tts/live/{listener_id}/captions`.

- [ ] **Step 1: Write failing public API contract tests**

Extend `_FakeHLSEncoder` with deterministic live edge and append receipts. After
creating the playlist lease and publishing an item, assert:

```python
captions = client.get(f"/api/tts/live/{listener_id}/captions")
assert captions.status_code == 200
assert captions.headers["cache-control"] == "no-store"
assert captions.json() == {
    "live_edge_at_ms": 100_550,
    "cues": [{
        "cue_id": captions.json()["cues"][0]["cue_id"],
        "start_at_ms": 100_000,
        "end_at_ms": 100_250,
        "text": "Stable translation 0.",
    }],
}
```

Also assert an unknown listener returns `404` and authentication remains
unnecessary for a valid public lease.

- [ ] **Step 2: Run API tests and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_protocol.py -k 'hls_caption'`.
Expected: `404` because the route does not exist.

- [ ] **Step 3: Implement the no-store JSON endpoint**

Validate the listener with `_validated_tts_client_id`, derive
`_public_hls_owner_key`, call `caption_snapshot`, translate
`HLSListenerNotFound` to `404`, and serialize only cue ID, times, and text.

- [ ] **Step 4: Run HLS protocol tests GREEN**

Run:

```bash
../.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_protocol.py \
  -k 'shared_hls or hls_caption or removing_one_hls_listener'
```

Expected: selected tests pass.

### Task 4: Device-Accurate Live Audio Caption UI

**Files:**
- Modify: `voxbridge/tts/listener_page.py`
- Modify: `tests/test_tts_listener_page_browser.py`

**Interfaces:**
- Consumes: `GET /api/tts/live/{listener_id}/captions` from Task 3.
- Produces DOM: `#liveCaption`, `#playbackStatus`, and `#nowPlaying[data-speaking]`.
- Produces JS: `pollCaptions()`, `estimatedPlaybackAtMs(snapshot)`, and
  `applyCaptionSnapshot(snapshot, requestListenerId)`.

- [ ] **Step 1: Extend the browser harness and write failing lag tests**

Return status JSON for `/status` and cue JSON for `/captions`. Simulate
`seekable.end=100`, `currentTime=94`, and `live_edge_at_ms=110_000`; assert the
cue containing `104_000` is shown even when a newer cue exists. Repeat with a
different lag and expect the newer cue.

- [ ] **Step 2: Run lag tests and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_tts_listener_page_browser.py -k 'caption_follows or device_lag'`.
Expected: FAIL because `#liveCaption` and caption polling do not exist.

- [ ] **Step 3: Implement semantic markup and matching styles**

Replace the single strong status with:

```html
<div class="now-playing-copy">
  <small>LIVE AUDIO</small>
  <strong id="liveCaption" aria-live="polite" aria-atomic="true">Waiting to start</strong>
  <span id="playbackStatus">Start listening to join the shared stream</span>
</div>
```

Keep the pulse. Use existing CSS variables and fonts, natural wrapping,
`min-width: 0`, and a flexible max-height. Add only a restrained reveal animation
and disable it under `prefers-reduced-motion`.

- [ ] **Step 4: Implement playhead selection and visible-only polling**

Every 500 ms while running and visible, fetch the listener-scoped endpoint. Use
`liveLagSec()` and `live_edge_at_ms` to estimate playhead wall time. Select the
latest cue with `start_at_ms <= playhead`; mark `data-speaking=true` only while
`playhead < end_at_ms`. Ignore responses whose captured listener ID no longer
matches. On visibility restoration, poll immediately. Caption errors retain the
existing text.

- [ ] **Step 5: Write failing retention and stop tests**

Assert a playhead inside the 300 ms gap keeps the prior caption with
`data-speaking=false`, a new cue replaces text without assigning `""`, and Stop
sets caption to `Waiting to start` and cancels further caption requests.

- [ ] **Step 6: Run behavior tests GREEN**

Run `../.venv/bin/python -m pytest -q tests/test_tts_listener_page_browser.py -k 'caption or listener_stop'`.
Expected: selected tests pass.

- [ ] **Step 7: Expand viewport assertions and run browser suite**

Add `#liveCaption` to viewport visibility assertions and inject a long natural
sentence before checking `1440x900`, `390x844`, `844x390`, and `320x568`.

Run `../.venv/bin/python -m pytest -q tests/test_tts_listener_page_browser.py`.
Expected: all browser tests pass with no document scrollbar.

### Task 5: Documentation, Full Verification, And 8024 Deployment

**Files:**
- Modify: `docs/API.md`
- Modify: `README.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/test_release_docs.py`

**Interfaces:**
- Documents the endpoint, synchronization formula, cue lifecycle, public-content
  boundary, and failure independence from audio.

- [ ] **Step 1: Write failing release-document assertions**

Require the API docs to contain the caption route, `live_edge_at_ms`, the
playhead formula, 256-entry bound, and the statement that caption polling does
not gate HLS audio.

- [ ] **Step 2: Run release docs test and verify RED**

Run `../.venv/bin/python -m pytest -q tests/test_release_docs.py`.
Expected: FAIL on the new required phrases.

- [ ] **Step 3: Update public documentation**

Document the exact endpoint response, listener bearer requirement, no-store
header, one-epoch retention, per-device lag mapping, sentence-gap retention, and
the unchanged one-worker/one-encoder architecture. Do not add deployment secrets
or machine-specific paths.

- [ ] **Step 4: Run focused and full automated verification**

Run:

```bash
../.venv/bin/python -m compileall -q voxbridge tests tools
../.venv/bin/python -m pytest -q tests/test_tts_hls.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_tts_listener_page_browser.py \
  tests/test_release_docs.py
../.venv/bin/python -m pytest -q
git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 5: Restart only the managed 8024 service**

Record the existing main and EngineCore PIDs, then run:

```bash
systemctl --user restart voxbridge-8024.service
ss -lntp | rg ':8024'
curl -fsS http://127.0.0.1:8024/api/tts/live/status
```

Verify the old PIDs exited, exactly one main process and one EngineCore remain,
and the status endpoint succeeds.

- [ ] **Step 6: Run a real HLS/API lifecycle check**

Create one public playlist lease, wait for a complete segment, query its caption
endpoint, delete the lease, and verify FFmpeg exits and cues are cleared. Do not
send synthetic text into the production TTS queue.

- [ ] **Step 7: Run Playwright visual verification on deployed 8024**

Open `/listen` at desktop, iPhone portrait, and short landscape sizes. Verify the
page has no horizontal or vertical document scrollbar, the caption occupies the
primary `Live Audio` hierarchy, controls remain visible, and console errors are
empty. Capture screenshots only under ignored temporary output.

- [ ] **Step 8: Review final diff and report evidence**

Report changed files, exact test counts, runtime PIDs, port listener, HLS lease
lifecycle, and any residual synchronization limit from one-second HLS segments.
