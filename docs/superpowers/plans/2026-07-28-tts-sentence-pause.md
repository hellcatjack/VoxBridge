# TTS Sentence Pause Implementation Plan

> **For the current-session executor:** Use `superpowers:executing-plans`.
> Do not use subagents; the user has explicitly prohibited them.

**Goal:** Add a cancellable `300ms` playback gate after each completed TTS clip
without delaying one-item lookahead preparation.

**Architecture:** One listener-local timer promise starts after
`playAudioBuffer()` resolves. The queue keeps its current item during that
window, so new jobs can still occupy and prepare `queue[0]`; reset paths reject
the timer with `AbortError` and generation guards discard stale work.

---

## Task 1: Define Timing And Cancellation

**Files:**
- Modify `tests/test_demo_streaming_ws_utils.py`
- Modify `tests/test_tts_listener_page_browser.py`

- [ ] Add static failing tests requiring:

```python
assert "const INTER_SENTENCE_PAUSE_MS = 300;" in TTS_LISTENER_HTML
assert "async function waitForInterSentencePause()" in TTS_LISTENER_HTML
assert "await waitForInterSentencePause();" in TTS_LISTENER_HTML
assert "function cancelInterSentencePause()" in TTS_LISTENER_HTML
assert TTS_LISTENER_HTML.count("cancelInterSentencePause();") >= 2
```

- [ ] Extend the controlled-media Playwright harness to record non-muted Blob
  playback start times.

- [ ] Add a failing browser test that ends job 1, waits for prepared job 2 to
  start, and requires at least `280ms` elapsed. The tolerance permits timer and
  scheduler precision while enforcing the `300ms` product setting.

- [ ] Add a failing browser test that stops during the timer and requires the
  page to reach `已停止` in under `200ms`.

- [ ] Run focused tests and verify they fail because no pause exists.

## Task 2: Implement The Playback Gate

**Files:**
- Modify `voxbridge/tts/listener_page.py`

- [ ] Add state and cancellation:

```javascript
const INTER_SENTENCE_PAUSE_MS = 300;
let cancelSentencePause = null;

function cancelInterSentencePause() {
  if (!cancelSentencePause) return;
  const cancel = cancelSentencePause;
  cancelSentencePause = null;
  cancel();
}
```

- [ ] Add a cancellable wait:

```javascript
async function waitForInterSentencePause() {
  await new Promise((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => settle(), INTER_SENTENCE_PAUSE_MS);
    const settle = (error) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      cancelSentencePause = null;
      if (error) reject(error); else resolve();
    };
    cancelSentencePause = () => {
      settle(new DOMException("sentence pause cancelled", "AbortError"));
    };
  });
}
```

Define `settle` before scheduling the timeout in final code so no callback can
observe it before initialization.

- [ ] After successful `playAudioBuffer`, set the playing indicator false and
  await the pause. Do not wait after fetch or playback failure.

- [ ] Call cancellation from `resetLocalPlayback()` and `beforeunload`.

- [ ] Run all listener static and Playwright tests; commit production and tests.

## Task 3: Documentation And Full Verification

**Files:**
- Modify `README.md`
- Modify `docs/API.md`
- Modify `docs/DEPLOYMENT.md`
- Modify `tests/test_release_docs.py`

- [ ] Add failing doc assertions for `300ms` and sentence pause.
- [ ] Document fixed wall-clock behavior, prefetch overlap, and cancellation.
- [ ] Run release-doc tests, compileall, and full pytest.
- [ ] Audit one timer, one media element, one prefetch head, and no protocol
  additions.
- [ ] Commit docs.

## Task 4: Integration And Port 8024 Deployment

- [ ] Fast-forward into unchanged clean main and rerun the full suite.
- [ ] Restart only `voxbridge-8024.service` and wait for `8024`.
- [ ] Verify old PIDs exit, one backend and one EngineCore remain,
  `NRestarts=0`, HTTPS `303/200`, and zero startup errors.
- [ ] Use the authenticated `/listen` page for Start/Stop smoke; deterministic
  Playwright tests provide sentence-pause timing evidence.
- [ ] Do not push unless explicitly requested.
