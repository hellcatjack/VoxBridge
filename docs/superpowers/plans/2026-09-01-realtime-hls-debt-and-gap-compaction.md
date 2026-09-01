# Real-time HLS Debt and Waiting-gap Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep long translated-speech debt visible to global Auto while removing only wait-generated HLS carrier silence and preserving the natural inter-sentence gap.

**Architecture:** `FFmpegHLSEncoder` will use a bounded two-second `2.0x` startup burst and then sustain `1.0x`, so its pending PCM remains a truthful shared debt signal. The encoder will report exact carrier bytes inserted while waiting; the publisher will turn that into absolute caption resume metadata, and each fixed-`1.0x` listener may perform one guarded seek without any custom pause/play/retry loop. Prepared PCM retains the speed selected at synthesis time through stable release.

**Tech Stack:** Python 3.12, asyncio, FastAPI, FFmpeg HLS, Kokoro ONNX, vanilla JavaScript, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-01-realtime-hls-debt-and-gap-compaction-design.md`

## Global Constraints

- Work directly on the existing `main` checkout only after the user selects inline execution; preserve unrelated changes.
- Run Python from the repository with `../.venv/bin/python`.
- Keep the production service on port `8024`.
- Keep every browser at `defaultPlaybackRate = 1.0`, `playbackRate = 1.0`, and hls.js `maxLiveSyncPlaybackRate: 1`.
- Keep Auto thresholds exactly `<10s = 1.0x`, `10-<30s = 1.2x`, `30-<40s = 1.4x`, and `>=40s = 1.5x`.
- Use a `2.0x` encoder burst for at most the first 2.0 seconds of active PCM after genuine queue starvation, then sustain `1.0x`.
- Keep tail-finalization carrier bounded by the existing bootstrap requirement.
- Never seek over translated speech or the natural PCM gap.
- A listener compaction performs at most one seek per cue and never calls `pause()` or `play()` as part of seeking.
- Do not add client telemetry to the global speed controller.
- No listener means no active TTS synthesis, encoder, epoch, or public speech backlog.

## File Structure

- `voxbridge/tts/hls.py`: bounded encoder pacing, exact discardable-gap receipts, caption resume metadata, prepared-speed reuse, and logging.
- `voxbridge/cli/demo_streaming_ws.py`: additive caption JSON fields.
- `voxbridge/tts/listener_page.py`: one-shot buffered gap compaction while media rate remains fixed.
- `tests/test_tts_hls.py`: encoder pacing, gap accounting, cue metadata, prepared-speed behavior, and backlog thresholds.
- `tests/test_demo_streaming_ws_protocol.py`: public caption endpoint contract.
- `tests/test_demo_streaming_ws_utils.py`: static listener contract with no custom recovery loop.
- `tests/test_tts_listener_page_browser.py`: native-HLS and hls.js seek behavior, buffer guards, one-shot semantics, and playback stability.
- `README.md`, `docs/API.md`, `docs/DEPLOYMENT.md`, `tests/test_release_docs.py`: operator and API documentation.

---

### Task 1: Bound accelerated HLS publication and measure provisional carrier

**Files:**
- Modify: `voxbridge/tts/hls.py:24-25`
- Modify: `voxbridge/tts/hls.py:74-78`
- Modify: `voxbridge/tts/hls.py:399-524`
- Test: `tests/test_tts_hls.py:990-1185`

**Interfaces:**
- Produces: `HLSAppendReceipt(start_at_ms: int, end_at_ms: int, discardable_gap_before_ms: int = 0)`.
- Produces: `ACTIVE_PCM_BURST_RATE = 2.0` and `ACTIVE_PCM_BURST_MEDIA_SEC = 2.0`.
- Preserves: `FFmpegHLSEncoder.pending_audio_ms` as the full unsubmitted PCM duration.

- [ ] **Step 1: Replace the permanent-fast-writer test with failing bounded-burst tests**

Add a recorder that captures every requested sleep after appending three seconds of PCM. Assert that the first 20 100ms frames use 50ms sleeps and later frames use 100ms sleeps:

```python
@pytest.mark.asyncio
async def test_ffmpeg_encoder_limits_two_x_burst_to_two_seconds(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    delays: list[float] = []
    wrote_21_frames = asyncio.Event()

    async def record_delay(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 21:
            wrote_21_frames.set()
        await real_sleep(0)

    class FakeStdin:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    monkeypatch.setattr(asyncio, "sleep", record_delay)
    encoder = FFmpegHLSEncoder(tmp_path / "live", sample_rate=8000, frame_ms=100)
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    encoder._timeline_origin_at_ms = 100_000
    await encoder.append_pcm(bytes(8000 * 3 * 2))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        await asyncio.wait_for(wrote_21_frames.wait(), timeout=1)
        assert delays[:20] == pytest.approx([0.05] * 20)
        assert delays[20] == pytest.approx(0.1)
        assert encoder.pending_audio_ms > 0
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None
```

Add a second test with two adjacent 1.5-second queue entries; it must still have only one two-second burst. Add a third test that lets the writer reach an empty-queue wait, appends another item, and proves the second item receives a fresh burst.

- [ ] **Step 2: Add a failing exact-discardable-gap receipt test**

```python
@pytest.mark.asyncio
async def test_append_receipt_reports_only_carrier_submitted_after_scheduled_audio(tmp_path):
    encoder = FFmpegHLSEncoder(tmp_path / "live", sample_rate=8000)
    encoder._process = SimpleNamespace(returncode=None)
    encoder._timeline_origin_at_ms = 100_000
    encoder._scheduled_end_pcm_bytes = 32_000
    encoder._submitted_pcm_bytes = 48_000

    receipt = await encoder.append_pcm(bytes(8_000 * 2))

    assert receipt.discardable_gap_before_ms == 1_000
    encoder._process = None
```

Also append an immediately adjacent second clip and assert its discardable value is zero.

- [ ] **Step 3: Run the new encoder tests and verify RED**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py::test_ffmpeg_encoder_limits_two_x_burst_to_two_seconds \
  tests/test_tts_hls.py::test_append_receipt_reports_only_carrier_submitted_after_scheduled_audio -q
```

Expected: the burst test observes 50ms sleeps after frame 20, and the receipt test fails because `discardable_gap_before_ms` does not exist.

- [ ] **Step 4: Implement receipt accounting before advancing the schedule**

Extend the frozen receipt with a compatible default:

```python
@dataclass(frozen=True, slots=True)
class HLSAppendReceipt:
    start_at_ms: int
    end_at_ms: int
    discardable_gap_before_ms: int = 0
```

In `append_pcm()`, capture the previous scheduled end before computing the actual start:

```python
natural_start_pcm_bytes = self._scheduled_end_pcm_bytes
start_pcm_bytes = max(self._submitted_pcm_bytes, natural_start_pcm_bytes)
discardable_pcm_bytes = max(0, start_pcm_bytes - natural_start_pcm_bytes)
discardable_gap_before_ms = round(
    discardable_pcm_bytes * 1000.0 / bytes_per_second
)
```

Return this value in `HLSAppendReceipt` after updating `_scheduled_end_pcm_bytes`.

- [ ] **Step 5: Implement the bounded burst in the writer loop**

Replace the permanent active rate with named constants:

```python
ACTIVE_PCM_BURST_RATE = 2.0
ACTIVE_PCM_BURST_MEDIA_SEC = 2.0
TAIL_PUBLISH_RATE = 2.0
```

At the start of `_writer_loop()`, calculate the burst byte budget. Reset it only in the branch that awaited a new item after `_pcm_queue` was empty. For every active frame:

```python
publish_rate = 1.0
if writing_audio and burst_pcm_bytes_remaining > 0:
    publish_rate = ACTIVE_PCM_BURST_RATE
    burst_pcm_bytes_remaining = max(
        0,
        burst_pcm_bytes_remaining - consumed_bytes,
    )
await asyncio.sleep(frame_sec / publish_rate)
```

Use `TAIL_PUBLISH_RATE` only inside `_flush_tail_until_visible()` and its deadline calculation.

- [ ] **Step 6: Run the complete HLS unit suite**

Run:

```bash
../.venv/bin/python -m pytest tests/test_tts_hls.py -q
```

Expected: all HLS tests pass, including real FFmpeg segment generation.

- [ ] **Step 7: Commit the transport change**

```bash
git add voxbridge/tts/hls.py tests/test_tts_hls.py
git commit -m "fix: bound accelerated HLS speech publication"
```

---

### Task 2: Preserve prepared speed and publish exact natural-gap markers

**Files:**
- Modify: `voxbridge/tts/hls.py:80-86`
- Modify: `voxbridge/tts/hls.py:1090-1285`
- Test: `tests/test_tts_hls.py:220-275`
- Test: `tests/test_tts_hls.py:490-570`

**Interfaces:**
- Consumes: `HLSAppendReceipt.discardable_gap_before_ms` from Task 1.
- Produces: `HLSCaptionCue.discardable_gap_before_ms: int`.
- Produces: `HLSCaptionCue.resume_at_ms: int | None`.
- Preserves: `_PreparedAudio.displayed_multiplier` and `_PreparedAudio.effective_speed` through stable release.

- [ ] **Step 1: Change the prepared-speed test to the desired behavior and verify RED**

Rename `test_release_regenerates_prepared_audio_when_global_speed_changed` and assert one synthesis call, not two:

```python
@pytest.mark.asyncio
async def test_release_reuses_speed_selected_when_audio_was_prepared(tmp_path):
    synth = FakeSynthesizer(make_wav(duration_ms=250))
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 10_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    item = ready_item(text="Keep the selected accelerated voice.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.prepare(item)
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)
        encoder.pending_audio_ms = 0
        await publisher.publish(item)
        await publisher.wait_idle()

        assert [call[2] for call in synth.speed_calls] == pytest.approx([1.26])
        assert publisher.status.global_speed_multiplier == 1.2
        assert len(encoder.appended) == 1
    finally:
        await publisher.close()
```

Run this test. Expected: FAIL because stable release discards the prepared `1.2x` PCM and synthesizes again at baseline.

- [ ] **Step 2: Add a failing caption marker test**

Allow `FakeEncoder` to return a configured `discardable_gap_before_ms`. Publish two 100ms tone clips with the normal 300ms pause, place 1,000ms of carrier before the second receipt, and assert:

```python
assert first.discardable_gap_before_ms == 0
assert first.resume_at_ms is None
assert second.discardable_gap_before_ms == 1_000
assert second.start_at_ms - second.resume_at_ms == 300
```

The last assertion uses the synthetic WAV with no detected edge silence, so the preserved natural gap is exactly the configured sentence pause.

Run the new test. Expected: FAIL because caption cues do not expose either field.

- [ ] **Step 3: Reuse prepared speed without a release-time downgrade**

In `_worker_loop()`, after popping prepared PCM, choose its recorded speed directly:

```python
if prepared is not None:
    _, decision_backlog_ms, _, _ = self._backlog_snapshot()
    displayed_multiplier = prepared.displayed_multiplier
    effective_speed = prepared.effective_speed
    self._global_speed_multiplier = displayed_multiplier
    speed_source = "prepared"
else:
    displayed_multiplier, effective_speed, decision_backlog_ms = (
        self._select_synthesis_speed_locked(key, kind)
    )
    speed_source = "decision"
```

Remove the mismatch-discard branch. Add `speed_source` to the existing start and publication logs. Revision invalidation remains unchanged.

- [ ] **Step 4: Add cue metadata and clamp untrusted durations**

Extend the caption cue:

```python
@dataclass(frozen=True, slots=True)
class HLSCaptionCue:
    cue_id: str
    start_at_ms: int
    end_at_ms: int
    text: str
    discardable_gap_before_ms: int = 0
    resume_at_ms: int | None = None
```

Before appending the current cue, read the previous cue. Clamp the receipt value to the actual cue gap, derive the natural gap, and construct the resume marker:

```python
previous = self._caption_cues[-1] if self._caption_cues else None
actual_gap_ms = 0 if previous is None else max(0, cue_start_at_ms - previous.end_at_ms)
discardable_gap_ms = min(
    actual_gap_ms,
    max(0, int(receipt.discardable_gap_before_ms)),
)
natural_gap_ms = max(0, actual_gap_ms - discardable_gap_ms)
resume_at_ms = (
    cue_start_at_ms - natural_gap_ms
    if previous is not None and discardable_gap_ms > 0
    else None
)
```

Store both fields and include the discardable duration in the publication log.

- [ ] **Step 5: Run prepared-speed and caption tests, then the whole HLS suite**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py::test_release_reuses_speed_selected_when_audio_was_prepared \
  tests/test_tts_hls.py::test_caption_cue_marks_only_wait_generated_carrier -q
../.venv/bin/python -m pytest tests/test_tts_hls.py -q
```

Expected: both focused tests and the HLS suite pass.

- [ ] **Step 6: Commit publisher behavior**

```bash
git add voxbridge/tts/hls.py tests/test_tts_hls.py
git commit -m "fix: preserve prepared speed and natural speech gaps"
```

---

### Task 3: Expose additive gap metadata through the caption API

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py:5790-5812`
- Modify: `tests/test_demo_streaming_ws_protocol.py:2250-2285`

**Interfaces:**
- Consumes: `HLSCaptionCue.discardable_gap_before_ms` and `resume_at_ms` from Task 2.
- Produces: caption JSON fields `discardable_gap_before_ms: int` and `resume_at_ms: int | null`.

- [ ] **Step 1: Extend the endpoint test and verify RED**

Configure the protocol `FakeEncoder` receipt with a 1,000ms disposable gap, publish two cues, and require the second response object to contain:

```python
assert payload["cues"][1]["discardable_gap_before_ms"] == 1_000
assert payload["cues"][1]["resume_at_ms"] is not None
```

Also require the first cue to serialize zero and `null`:

```python
assert payload["cues"][0]["discardable_gap_before_ms"] == 0
assert payload["cues"][0]["resume_at_ms"] is None
```

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_demo_streaming_ws_protocol.py::test_public_hls_caption_feed_requires_matching_listener_lease -q
```

Expected: FAIL because the new keys are absent.

- [ ] **Step 2: Serialize the two additive fields**

Add to each cue dictionary:

```python
"discardable_gap_before_ms": int(cue.discardable_gap_before_ms),
"resume_at_ms": (
    int(cue.resume_at_ms) if cue.resume_at_ms is not None else None
),
```

- [ ] **Step 3: Run protocol and utility suites**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit the API contract**

```bash
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_protocol.py
git commit -m "feat: expose disposable HLS gap markers"
```

---

### Task 4: Compact a buffered historical gap exactly once

**Files:**
- Modify: `voxbridge/tts/listener_page.py:530-830`
- Modify: `voxbridge/tts/listener_page.py:930-1125`
- Modify: `tests/test_tts_listener_page_browser.py:45-275`
- Modify: `tests/test_tts_listener_page_browser.py:320-365`
- Modify: `tests/test_tts_listener_page_browser.py:825-870`
- Modify: `tests/test_demo_streaming_ws_utils.py:2335-2375`

**Interfaces:**
- Consumes: cue fields `discardable_gap_before_ms` and `resume_at_ms`.
- Produces: `compactBufferedWaitingGap(playheadAtMs: number, previousCue: object, nextCue: object): boolean` in the listener script.
- Preserves: fixed media rate and native HLS/hls.js timestamp mapping.

- [ ] **Step 1: Instrument the browser harness for seek calls**

Before production code changes, add:

```javascript
window.__ttsSeekCalls = [];
HTMLMediaElement.prototype.fastSeek = function(value) {
  window.__ttsSeekCalls.push(Number(value));
  this.currentTime = Number(value);
};
```

Keep the existing `__ttsPlayCalls` and `__ttsPauseCalls` so tests can prove the compactor does not touch playback lifecycle.

- [ ] **Step 2: Replace the obsolete no-seek test with a failing guarded-compaction test**

Use two cues with an actual 3,000ms gap, 2,500ms disposable carrier, a 500ms natural gap, and a playhead 100ms after the previous cue ended. Map program time 104,100ms to media time 50.0s and buffer through 54.0s. Assert:

```python
assert listener_page.evaluate("window.__ttsSeekCalls") == pytest.approx([52.5])
assert listener_page.evaluate("window.__ttsPauseCalls") == 0
assert len(listener_page.evaluate("window.__ttsPlayCalls")) == 1
assert listener_page.eval_on_selector(
    "#ttsPlayback", "node => [node.defaultPlaybackRate, node.playbackRate]"
) == [1, 1]
```

The 52.5s target retains the remaining 400ms of the natural gap and skips only historical carrier.

- [ ] **Step 3: Add failing safety tests**

Add separate Playwright cases proving:

- a buffer ending before `next.start_at_ms + 1,000` causes no seek;
- extending that buffer on a later poll permits one seek;
- repeated polls after the attempt do not add another seek;
- a zero or 499ms disposable gap is not compacted;
- a seek exception does not call `play()` or `pause()` and sequential playback remains active;
- Safari `getStartDate()` and hls.js `playingDate` produce the same target.

Run the new cases. Expected: the positive cases fail because no compactor exists; negative cases remain green.

- [ ] **Step 4: Implement range and target helpers without a recovery state machine**

Add constants and one state key:

```javascript
const MIN_DISCARDABLE_GAP_MS = 500;
const NEXT_SPEECH_BUFFER_GUARD_MS = 1000;
const attemptedGapCueKeys = new Set();
```

Add a range helper that requires both the target and guarded speech point inside the same buffered range. Implement `compactBufferedWaitingGap()` with these operations only:

```javascript
const naturalGapMs = Math.max(0, nextStartAtMs - resumeAtMs);
const heardGapMs = Math.max(0, playheadAtMs - previousEndAtMs);
const remainingNaturalMs = Math.max(0, naturalGapMs - heardGapMs);
const targetProgramAtMs = nextStartAtMs - remainingNaturalMs;
const currentTime = Number(playbackElement.currentTime);
const targetMediaTime = currentTime + (targetProgramAtMs - playheadAtMs) / 1000;
const guardedMediaTime = currentTime
  + (nextStartAtMs + NEXT_SPEECH_BUFFER_GUARD_MS - playheadAtMs) / 1000;
```

After validating `running && playbackStarted`, the cue, target ordering, and
shared buffered range, add the cue key before the operation. Do not require
`nowPlaying.dataset.playing === "true"`: a native `waiting` event is the primary
case this compactor must resolve.

```javascript
if (attemptedGapCueKeys.has(nextCueKey)) return false;
attemptedGapCueKeys.add(nextCueKey);
```

Then perform the single operation:

```javascript
try {
  if (typeof playbackElement.fastSeek === "function") {
    playbackElement.fastSeek(targetMediaTime);
  } else {
    playbackElement.currentTime = targetMediaTime;
  }
} catch (error) {}
```

Do not add a timeout, pending-seek object, `play()`, `pause()`, buffering copy, or retry.

- [ ] **Step 5: Call the compactor only between the selected and next cue**

In `applyCaptionSnapshot()`, select both the latest cue at/before the playhead and the earliest future cue. After retaining the selected caption, call:

```javascript
compactBufferedWaitingGap(playheadAtMs, selected, next);
```

Clear `attemptedGapCueKeys` when starting or stopping a listener epoch. Continue forcing normal media rate from all existing media events. Add one browser case that dispatches `waiting`, then makes the guarded next cue available; it must seek once without a second `play()` call and return to the existing native event flow.

- [ ] **Step 6: Add static assertions against the removed retry design**

In `test_demo_streaming_ws_utils.py`, require the one-shot helper and forbid the old recovery identifiers:

```python
assert "function compactBufferedWaitingGap(" in TTS_LISTENER_HTML
assert "playbackElement.fastSeek(targetMediaTime)" in TTS_LISTENER_HTML
for removed in (
    "pendingSilenceCompaction",
    "silenceCompactionRecoveryTimer",
    "requestPendingSilencePlayback",
):
    assert removed not in TTS_LISTENER_HTML
```

- [ ] **Step 7: Run browser and static listener suites**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_listener_page_browser.py \
  tests/test_demo_streaming_ws_utils.py -q
```

Expected: all listener tests pass with no document-scrollbar or caption regressions.

- [ ] **Step 8: Commit the listener compactor**

```bash
git add \
  voxbridge/tts/listener_page.py \
  tests/test_tts_listener_page_browser.py \
  tests/test_demo_streaming_ws_utils.py
git commit -m "fix: compact only buffered waiting gaps"
```

---

### Task 5: Document, verify, deploy, and inspect real behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `tests/test_release_docs.py`

**Interfaces:**
- Documents: bounded 2-second burst, sustained real-time pacing, prepared-speed ownership, caption gap fields, and one-shot compaction.
- Verifies: service `voxbridge-8024.service` on port `8024`.

- [ ] **Step 1: Update the public contract test first and verify RED**

Require all three documents to mention:

```python
assert "discardable_gap_before_ms" in api
assert "resume_at_ms" in api
assert "two-second" in deployment.lower()
assert "one-shot" in deployment.lower()
assert "不循环重试" in readme
```

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_release_docs.py::test_public_docs_describe_optional_kokoro_tts_contract -q
```

Expected: FAIL because the new behavior is not documented.

- [ ] **Step 2: Update README, API, and deployment documentation**

Document these exact semantics:

- active HLS PCM bursts at `2.0x` for no more than 2.0 seconds after starvation and then sustains `1.0x`;
- global Auto continues to use unpublished server PCM with unchanged tiers;
- prepared PCM keeps the speed selected when synthesis began;
- caption `discardable_gap_before_ms` identifies only wait-generated carrier;
- caption `resume_at_ms` preserves the continuous-speech natural gap;
- browsers seek only after a 1.0-second buffer guard, once per cue, with no custom pause/play/retry loop;
- all media elements remain at `1.0x`.

- [ ] **Step 3: Run focused regression suites**

Run:

```bash
git diff --check
../.venv/bin/python -m pytest \
  tests/test_kokoro_tts.py \
  tests/test_tts_hls.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py \
  tests/test_tts_listener_page_browser.py \
  tests/test_release_docs.py -q
```

Expected: zero failures.

- [ ] **Step 4: Run the complete repository test suite**

Run:

```bash
../.venv/bin/python -m pytest tests -q
```

Expected: zero failures. Record the exact passing count and elapsed time.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/API.md docs/DEPLOYMENT.md tests/test_release_docs.py
git commit -m "docs: explain real-time HLS debt and gap markers"
```

- [ ] **Step 6: Review the completed commit range**

Run:

```bash
git diff --check e5b3092..HEAD
git status --short --branch
git log --oneline e5b3092..HEAD
```

Expected: no whitespace errors, no uncommitted files, and five implementation commits after the design and plan commits.

- [ ] **Step 7: Restart and verify the production service**

Run:

```bash
systemctl --user restart voxbridge-8024.service
systemctl --user is-active voxbridge-8024.service
ss -lntp | rg ':8024'
curl -fsS http://127.0.0.1:8024/api/tts/live/status
```

Expected: active service on `8024`; idle status has an empty epoch, zero backlog, `global_speed_mode: "auto"`, multiplier `1.0`, effective speed `1.05`, encoder inactive, and empty `last_error`.

- [ ] **Step 8: Run two-listener HLS lifecycle smoke test**

Create two temporary listener IDs through their playlist endpoints, capture status after each, and delete only those IDs. Expected: both playlist requests return 200, the second listener retains the first listener's `speech_epoch_id`, and deleting the final temporary listener clears the epoch and encoder.

- [ ] **Step 9: Inspect post-deployment pacing and Auto evidence**

During the next active translation run, inspect `/data/Qwen3-ASR/logs/voxbridge_8024.log` for publication records. Verify:

- sustained pending PCM no longer disappears at a permanent `2.0x` publication rate;
- a prepared sentence logs `speed_source=prepared` and is not regenerated at a lower tier;
- when unpublished backlog crosses 30,000ms or 40,000ms, the next synthesized sentence logs multiplier `1.4` or `1.5` respectively;
- publication logs report nonzero `discardable_gap_before_ms` only after a real wait.

If the live sermon does not naturally cross a tier during the verification window, rely on the exact-boundary and long-PCM automated tests rather than injecting synthetic church audio into production.

- [ ] **Step 10: Push `main` without force only when origin has not diverged**

Run:

```bash
git fetch origin main
git rev-list --left-right --count origin/main...main
git push origin main
```

Expected before push: zero remote-only commits. Never force-push a diverged `main`.

---

## Final Verification Checklist

- [ ] Each new test was observed failing for the intended missing behavior before production edits.
- [ ] Long PCM keeps pending debt after the two-second HLS burst.
- [ ] Adjacent sentences do not receive repeated burst budgets.
- [ ] A real empty-queue wait resets the burst once.
- [ ] Prepared speed is reused through stable release.
- [ ] Continuous speech has zero disposable carrier.
- [ ] Wait-generated carrier is reported exactly once.
- [ ] Listener compaction retains the computed natural gap.
- [ ] Listener compaction never calls pause/play and never retries a cue.
- [ ] Native HLS and hls.js browser tests pass at media rate `1.0`.
- [ ] Focused and full suites pass with fresh output.
- [ ] Port `8024`, service state, status JSON, runtime logs, and two-listener lifecycle are verified after restart.
- [ ] Worktree is clean and `origin/main` matches the deployed commit.
