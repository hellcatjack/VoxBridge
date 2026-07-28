# TTS Single-Lookahead Prefetch Implementation Plan

> **For the current-session executor:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task. Do not use subagents; the user has
> explicitly prohibited them.

**Goal:** Prepare exactly one future TTS WAV during current playback so queued
items transition without waiting for normal lazy synthesis and HTTP transfer.

**Architecture:** Keep the existing FIFO and one persistent media element. Add
a browser-local preparation map whose entries settle errors instead of rejecting
unobserved, reuse the prepared promise when a job becomes current, and allow
only `queue[0]` to be prepared ahead. Reset paths abort and clear every entry.

**Tech Stack:** Embedded JavaScript, Fetch/AbortController, HTMLMediaElement,
pytest, Playwright, FastAPI broadcast TTS, user systemd service on port 8024.

---

## File Map

- Modify `voxbridge/tts/listener_page.py`: bounded preparation registry,
  lookahead trigger, prepared-byte reuse, and cancellation.
- Modify `tests/test_demo_streaming_ws_utils.py`: static prefetch invariants.
- Modify `tests/test_tts_listener_page_browser.py`: real JavaScript ordering,
  reuse, and cancellation tests.
- Modify `tests/test_release_docs.py`, `README.md`, `docs/API.md`, and
  `docs/DEPLOYMENT.md`: document bounded lookahead and residual-gap limits.

## Task 1: Define Bounded Lookahead Behavior

**Files:**
- Modify: `tests/test_demo_streaming_ws_utils.py:1791-1890`
- Modify: `tests/test_tts_listener_page_browser.py`

- [ ] **Step 1: Add failing static contract tests**

Add assertions for one preparation map and remove the obsolete claim that only
the currently playing FIFO head may be fetched:

```python
def test_listener_page_prefetches_only_one_future_fifo_item():
    assert "const audioPreparations = new Map();" in TTS_LISTENER_HTML
    assert "function prepareAudio(job)" in TTS_LISTENER_HTML
    assert "function prefetchNextAudio()" in TTS_LISTENER_HTML
    assert "const nextJob = queue[0];" in TTS_LISTENER_HTML
    assert "prepareAudio(nextJob);" in TTS_LISTENER_HTML
    assert "queue.slice" not in TTS_LISTENER_HTML


def test_listener_page_reuses_prepared_audio_and_cancels_on_reset():
    assert "async function consumePreparedAudio(job)" in TTS_LISTENER_HTML
    assert "const preparation = prepareAudio(job);" in TTS_LISTENER_HTML
    assert "audioPreparations.delete(jobId);" in TTS_LISTENER_HTML
    assert "function cancelAudioPreparations()" in TTS_LISTENER_HTML
    assert "preparation.controller.abort();" in TTS_LISTENER_HTML
    assert "cancelAudioPreparations();" in TTS_LISTENER_HTML
```

Update the existing local-stop test to require preparation cancellation rather
than the removed single `abortController`.

- [ ] **Step 2: Add a failing Playwright ordering test**

Add a helper that injects a fake WebSocket, a successful WAV fetch, and media
methods. Expose `window.__ttsSocket`, `window.__ttsFetchCalls`, and
`window.__ttsSentMessages`. Send three stable jobs and assert:

```python
assert fetch_job_ids(listener_page) == ["job-1"]
emit_job(listener_page, "job-2", 1)
emit_job(listener_page, "job-3", 2)
listener_page.wait_for_function("window.__ttsFetchCalls.length === 2")
assert fetch_job_ids(listener_page) == ["job-1", "job-2"]
listener_page.wait_for_timeout(100)
assert fetch_job_ids(listener_page) == ["job-1", "job-2"]
listener_page.dispatch_event("#ttsPlayback", "ended")
listener_page.wait_for_function("window.__ttsFetchCalls.length === 3")
assert fetch_job_ids(listener_page) == ["job-1", "job-2", "job-3"]
```

This proves the second fetch happens before first playback ends, the third is
bounded, and promotion opens exactly one new lookahead slot.

- [ ] **Step 3: Add a failing Playwright cancellation test**

Inject deferred fetch promises that count abort signals. Start job 1, enqueue job
2, wait until two fetches exist, select Stop, and require:

```python
listener_page.wait_for_function("window.__ttsAbortCount === 2")
assert listener_page.evaluate("window.__ttsAbortCount") == 2
```

- [ ] **Step 4: Run focused tests and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py -k 'listener_page and prefetch' \
  tests/test_tts_listener_page_browser.py -k 'prefetch or preparation'
```

Expected: static tests fail because the registry does not exist; ordering sees
only job 1 before `ended`; cancellation sees one abort instead of two.

## Task 2: Implement Single-Lookahead Preparation

**Files:**
- Modify: `voxbridge/tts/listener_page.py:340-525`
- Modify: `tests/test_demo_streaming_ws_utils.py`
- Modify: `tests/test_tts_listener_page_browser.py`

- [ ] **Step 1: Replace the single fetch controller with preparation state**

Replace `abortController` with:

```javascript
const audioPreparations = new Map();

function jobIdOf(job) {
  return String(job && job.job_id || "");
}

function prepareAudio(job) {
  const jobId = jobIdOf(job);
  const existing = audioPreparations.get(jobId);
  if (existing) return existing;
  const controller = new AbortController();
  const preparation = {
    controller,
    audioBytes: null,
    error: null,
    promise: null,
  };
  preparation.promise = fetchAudio(job, controller.signal)
    .then((audioBytes) => {
      preparation.audioBytes = audioBytes;
      return preparation;
    })
    .catch((error) => {
      preparation.error = error;
      return preparation;
    });
  audioPreparations.set(jobId, preparation);
  return preparation;
}
```

The catch must settle to the preparation object so an early failed prefetch
cannot create an unhandled rejection.

- [ ] **Step 2: Add bounded lookahead, consumption, and cancellation**

```javascript
function prefetchNextAudio() {
  if (!currentJob || queue.length === 0) return;
  const nextJob = queue[0];
  prepareAudio(nextJob);
}

async function consumePreparedAudio(job) {
  const jobId = jobIdOf(job);
  const preparation = prepareAudio(job);
  await preparation.promise;
  audioPreparations.delete(jobId);
  if (preparation.error) throw preparation.error;
  return preparation.audioBytes;
}

function cancelAudioPreparations() {
  for (const preparation of audioPreparations.values()) {
    preparation.controller.abort();
  }
  audioPreparations.clear();
}
```

- [ ] **Step 3: Wire preparation into FIFO promotion**

After `currentJob = queue.shift()`, start current consumption first and then open
one lookahead slot:

```javascript
const audioPromise = consumePreparedAudio(currentJob);
prefetchNextAudio();
const audioBytes = await audioPromise;
await playAudioBuffer(audioBytes, localGeneration);
```

When a new `tts_job` is pushed, call `prefetchNextAudio()` before `pumpQueue()`.
Because it returns unless `currentJob` exists and always addresses `queue[0]`, it
cannot start job 3 while job 2 is still the one future item.

- [ ] **Step 4: Wire cancellation into every reset**

Call `cancelAudioPreparations()` in `resetLocalPlayback()` before clearing the
queue. Remove creation, clearing, and aborting of the old `abortController`.
Existing Stop, socket close, and Start all reach this reset. Also call
`cancelAudioPreparations()` directly in `beforeunload` beside
`stopActivePlayback()` because the browser may not wait for the socket close
event during page destruction.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py -k 'listener_page' \
  tests/test_tts_listener_page_browser.py
```

Expected: all listener static and browser tests pass.

- [ ] **Step 6: Commit**

```bash
git add voxbridge/tts/listener_page.py \
  tests/test_demo_streaming_ws_utils.py tests/test_tts_listener_page_browser.py
git commit -m "feat: prefetch one queued TTS item"
```

## Task 3: Public Contract

**Files:**
- Modify: `tests/test_release_docs.py`
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`

- [ ] **Step 1: Add failing documentation assertions**

Extend the Kokoro listener contract test:

```python
assert "单条预取" in readme
assert "single-item lookahead" in api
assert "bounded" in deployment.lower()
```

- [ ] **Step 2: Run and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_release_docs.py::test_public_docs_describe_optional_kokoro_tts_contract
```

- [ ] **Step 3: Document behavior and limits**

Document that current playback triggers preparation of only the next FIFO WAV,
Stop cancels it, shared synthesis is unchanged, and a residual delay remains if
Kokoro cannot finish before current playback ends. Do not claim sample-accurate
gapless playback.

- [ ] **Step 4: Verify and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_release_docs.py
git add README.md docs/API.md docs/DEPLOYMENT.md tests/test_release_docs.py
git commit -m "docs: explain bounded TTS lookahead"
```

## Task 4: Full Verification And Port 8024 Deployment

**Files:**
- No systemd changes.

- [ ] **Step 1: Verify the feature worktree**

```bash
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
git diff --check
git status --short --branch
```

- [ ] **Step 2: Audit bounds and protocol**

Confirm one `queue[0]` lookahead, no queue-wide fetch loop, no new endpoint or
WebSocket message, no second media element, and reset cancellation.

- [ ] **Step 3: Fast-forward into clean unchanged main and rerun full tests**

```bash
git merge --ff-only feat/tts-single-lookahead-prefetch
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
```

- [ ] **Step 4: Restart only the user service**

```bash
systemctl --user restart voxbridge-8024.service
```

Wait for `8024`; verify old PIDs are gone, one backend and one EngineCore remain,
`NRestarts=0`, HTTPS root/login return `303/200`, and startup logs contain no
errors.

- [ ] **Step 5: Run authenticated deployed-page smoke**

Open `/listen` with Playwright, Start one listener, inject no synthetic meeting
content, and verify the page still connects, rate persistence works, and Stop
disconnects cleanly. Automated fake-job Playwright tests provide the deterministic
prefetch ordering evidence without exposing meeting text.

- [ ] **Step 6: Record evidence**

Report exact test count, commit, PID/topology, port, HTTPS, logs, and the bounded
prefetch ordering result. Do not push unless explicitly requested.
