# Shared Speech-Only HLS Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the continuously encoded HLS idle carrier with a bounded startup bootstrap so every listener receives one shared, speech-only media timeline and hears newly released translation after only normal HLS slice buffering.

**Architecture:** `FFmpegHLSEncoder` will reserve and write a one-time bootstrap carrier, then block on its shared PCM queue instead of writing idle frames. Released sentences remain FIFO media with the existing 300 ms sentence pause. The Listen page will consume that shared timeline directly and remove all device-local historical-silence seeking.

**Tech Stack:** Python 3.11, asyncio, FFmpeg AAC/HLS, FastAPI, vanilla JavaScript, hls.js, pytest, pytest-asyncio, Playwright with Google Chrome, systemd user services.

**Spec:** `docs/superpowers/specs/2026-08-31-shared-speech-only-hls-timeline-design.md`

## Global Constraints

- The runtime project is `/data/Qwen3-ASR/VoxBridge`; run Python only through `/data/Qwen3-ASR/.venv/bin/python` (from the repo, `../.venv/bin/python`).
- The public service must remain on port `8024`; do not move it to 8000, 8001, or 8080.
- Preserve one shared TTS synthesizer and one FFmpeg encoder for all active listener leases; the configured listener capacity remains 128 and lease TTL remains 90 seconds.
- Preserve rollback-safe translation release and the existing 300 ms appended sentence pause.
- Preserve Auto tiers exactly: `<10s: 1.0x`, `<30s: 1.2x`, `<40s: 1.4x`, `>=40s: 1.5x`.
- Do not add periodic heartbeat audio; a platform-specific stalled-playlist problem must be handled by control-plane reload logic only after it is reproduced.
- Do not trim TTS leading/trailing PCM in this change.
- Follow strict red-green TDD: every behavior test must be observed failing for the intended reason before production code changes.

## File Structure

- Modify `voxbridge/tts/hls.py`: own bounded bootstrap accounting, queue-blocking idle behavior, PCM scheduling, FFmpeg lifecycle, and shared HLS receipts.
- Modify `tests/test_tts_hls.py`: exercise the writer state machine against a fake stdin and the real FFmpeg playlist across bootstrap, idle, and resume.
- Modify `voxbridge/tts/listener_page.py`: remove local historical-silence seeking and let ordinary media `waiting`/`playing` events represent the shared live edge.
- Modify `tests/test_tts_listener_page_browser.py`: replace compaction tests with observable no-seek shared-timeline behavior and retain existing wait/resume/Auto regressions.
- No API, database, dependency, or configuration file changes are planned.

---

### Task 1: Make the shared FFmpeg encoder stop advancing while idle

**Files:**
- Modify: `tests/test_tts_hls.py:940-1175`
- Modify: `voxbridge/tts/hls.py:253-430`

**Interfaces:**
- Consumes: `FFmpegHLSEncoder.start()`, `append_pcm(pcm: bytes) -> HLSAppendReceipt`, `wait_ready(timeout: float)`, `pending_audio_ms`, `playlist_text()`, and `parse_hls_live_edge_at_ms()`.
- Produces: unchanged public encoder interface; private integer state `_bootstrap_pcm_bytes_remaining` and `_bootstrap_pcm_bytes_total` reserve the startup region before any translated PCM receipt is issued.

- [ ] **Step 1: Replace the writer-loop idle-carrier test with a failing queue-blocking behavior test**

The production mutation this catches is restoring `chunk = silence` after translated PCM drains. Replace `test_ffmpeg_encoder_bursts_backlog_but_keeps_idle_carrier_realtime` with:

```python
@pytest.mark.asyncio
async def test_ffmpeg_encoder_bursts_backlog_then_blocks_without_idle_writes(
    tmp_path,
    monkeypatch,
):
    real_sleep = asyncio.sleep
    delays: list[float] = []
    writes: list[bytes] = []
    wrote_two_frames = asyncio.Event()

    async def record_delay(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    class FakeStdin:
        def write(self, data: bytes) -> None:
            writes.append(bytes(data))
            if len(writes) == 2:
                wrote_two_frames.set()

        async def drain(self) -> None:
            return None

    monkeypatch.setattr(asyncio, "sleep", record_delay)
    encoder = FFmpegHLSEncoder(
        tmp_path / "live",
        sample_rate=8000,
        frame_ms=100,
    )
    encoder._process = SimpleNamespace(returncode=None, stdin=FakeStdin())
    encoder._timeline_origin_at_ms = 100_000
    await encoder.append_pcm(bytes(round(8000 * 0.2) * 2))

    writer = asyncio.create_task(encoder._writer_loop())
    try:
        await asyncio.wait_for(wrote_two_frames.wait(), timeout=1)
        await real_sleep(0.02)

        assert len(writes) == 2
        assert delays == pytest.approx([0.1 / 1.4, 0.1 / 1.4])
        assert writer.done() is False
    finally:
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        encoder._process = None
```

- [ ] **Step 2: Run the writer test and verify RED**

Run:

```bash
../.venv/bin/python -m pytest tests/test_tts_hls.py::test_ffmpeg_encoder_bursts_backlog_then_blocks_without_idle_writes -q
```

Expected: FAIL because the current writer produces a third idle-carrier write and delay instead of blocking on `_pcm_queue`.

- [ ] **Step 3: Change the real-FFmpeg bootstrap test to prove the playlist becomes stable when idle**

Rename `test_ffmpeg_encoder_idle_carrier_produces_decodable_aac_hls_segment` to `test_ffmpeg_encoder_bootstrap_is_decodable_and_does_not_keep_advancing`. Keep its FFprobe assertions, then add this observable check before `finally`:

The bounded bootstrap may contain only one listed segment, so change the existing segment collection from `][:-1]` to the complete list:

```python
        bootstrap_segments = [
            line.strip()
            for line in playlist.splitlines()
            if line.strip().endswith(".ts")
        ]
        assert bootstrap_segments
```

Probe `bootstrap_segments[0]`, then add:

```python
        while encoder._bootstrap_pcm_bytes_remaining:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        idle_playlist = encoder.playlist_text()
        idle_edge = parse_hls_live_edge_at_ms(idle_playlist)

        await asyncio.sleep(0.8)

        assert encoder.playlist_text() == idle_playlist
        assert parse_hls_live_edge_at_ms(encoder.playlist_text()) == idle_edge
        assert encoder.pending_audio_ms == 0
```

- [ ] **Step 4: Add a real-FFmpeg idle/resume receipt test**

The production mutations this catches are adding wall-clock idle to receipt timestamps or failing to resume FFmpeg after the queue has been empty. Add:

```python
@pytest.mark.asyncio
async def test_ffmpeg_encoder_resumes_after_idle_without_timeline_gap(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")
    encoder = FFmpegHLSEncoder(
        tmp_path / "speech-only-live",
        sample_rate=24000,
        segment_sec=0.5,
        playlist_segments=8,
        frame_ms=50,
    )
    tone = decode_mono_pcm16_wav(
        make_wav(duration_ms=1200),
        expected_rate=24000,
    )
    await encoder.start()
    try:
        await encoder.wait_ready(timeout=5)
        while encoder._bootstrap_pcm_bytes_remaining:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        bootstrap_edge = parse_hls_live_edge_at_ms(encoder.playlist_text())
        assert bootstrap_edge is not None

        first = await encoder.append_pcm(tone)
        await wait_until(lambda: encoder.pending_audio_ms == 0, timeout=3)
        await wait_until(
            lambda: (
                parse_hls_live_edge_at_ms(encoder.playlist_text()) or 0
            ) > bootstrap_edge,
            timeout=3,
        )
        await asyncio.sleep(0.3)
        first_idle_playlist = encoder.playlist_text()
        first_idle_edge = parse_hls_live_edge_at_ms(first_idle_playlist)

        await asyncio.sleep(0.8)
        assert encoder.playlist_text() == first_idle_playlist

        second = await encoder.append_pcm(tone)
        assert second.start_at_ms == first.end_at_ms
        await wait_until(lambda: encoder.pending_audio_ms == 0, timeout=3)
        await wait_until(
            lambda: (
                parse_hls_live_edge_at_ms(encoder.playlist_text()) or 0
            ) > (first_idle_edge or 0),
            timeout=3,
        )
    finally:
        await encoder.close()
```

- [ ] **Step 5: Run both real-FFmpeg tests and verify RED**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py::test_ffmpeg_encoder_bootstrap_is_decodable_and_does_not_keep_advancing \
  tests/test_tts_hls.py::test_ffmpeg_encoder_resumes_after_idle_without_timeline_gap -q
```

Expected: the bootstrap test FAILS because the playlist continues gaining carrier segments. The resume test must also fail either on the unchanged-playlist assertion or because the old writer has already charged wall-clock carrier bytes into the second receipt.

- [ ] **Step 6: Implement bounded bootstrap reservation in `FFmpegHLSEncoder`**

In `__init__`, add zeroed bootstrap state without changing direct unit-test construction semantics:

```python
        self._bootstrap_pcm_bytes_total = 0
        self._bootstrap_pcm_bytes_remaining = 0
```

Add a private calculator next to `playlist_path`:

```python
    def _required_bootstrap_pcm_bytes(self) -> int:
        frame_sec = self._frame_bytes / (self.sample_rate * 2)
        aac_frame_sec = 1024 / self.sample_rate
        bootstrap_sec = self.segment_sec + max(frame_sec, aac_frame_sec)
        frame_count = max(1, math.ceil(bootstrap_sec / frame_sec))
        return frame_count * self._frame_bytes
```

In `start()`, immediately before the writer task is created, reserve the complete bootstrap region:

```python
        self._bootstrap_pcm_bytes_total = self._required_bootstrap_pcm_bytes()
        self._bootstrap_pcm_bytes_remaining = self._bootstrap_pcm_bytes_total
        self._scheduled_end_pcm_bytes = max(
            self._scheduled_end_pcm_bytes,
            self._bootstrap_pcm_bytes_total,
        )
```

Replace `_writer_loop()` with this state order: bootstrap frames first, then complete translated PCM items, then await the next queue item. Do not poll the queue and do not synthesize an idle frame after bootstrap.

```python
    async def _writer_loop(self) -> None:
        frame_bytes = self._frame_bytes
        frame_samples = frame_bytes // 2
        frame_sec = frame_samples / self.sample_rate
        active = b""
        try:
            while True:
                bootstrapping = self._bootstrap_pcm_bytes_remaining > 0
                if bootstrapping:
                    chunk = self._idle_carrier_pcm
                    self._bootstrap_pcm_bytes_remaining = max(
                        0,
                        self._bootstrap_pcm_bytes_remaining - frame_bytes,
                    )
                    writing_audio = False
                    consumed_bytes = 0
                else:
                    if not active:
                        active = await self._pcm_queue.get()
                    writing_audio = True
                    chunk = active[:frame_bytes]
                    consumed_bytes = len(chunk)
                    active = active[len(chunk):]
                    if len(chunk) < frame_bytes:
                        chunk += bytes(frame_bytes - len(chunk))

                process = self._process
                if process is None or process.returncode is not None or process.stdin is None:
                    raise HLSUnavailable("FFmpeg exited while streaming")
                process.stdin.write(chunk)
                self._submitted_pcm_bytes += len(chunk)
                await process.stdin.drain()
                if writing_audio:
                    self._pending_pcm_bytes = max(
                        0,
                        self._pending_pcm_bytes - consumed_bytes,
                    )
                    if not active:
                        self._pcm_queue.task_done()
                publish_rate = BACKLOG_PUBLISH_RATE if writing_audio else 1.0
                await asyncio.sleep(frame_sec / publish_rate)
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionResetError, HLSUnavailable) as exc:
            logger.warning("shared HLS encoder stopped: %s", type(exc).__name__)
```

Reset both bootstrap counters in `close()` alongside other timeline accounting.

- [ ] **Step 7: Run focused encoder tests and verify GREEN**

Run:

```bash
../.venv/bin/python -m pytest tests/test_tts_hls.py -q
```

Expected: every HLS test passes; FFmpeg integration tests are skipped only if FFmpeg/FFprobe is genuinely unavailable.

- [ ] **Step 8: Commit the encoder behavior**

```bash
git add voxbridge/tts/hls.py tests/test_tts_hls.py
git commit -m "fix: pause shared HLS timeline while translation is idle"
```

---

### Task 2: Remove device-local historical-silence seeking

**Files:**
- Modify: `tests/test_tts_listener_page_browser.py:460-510,1138-1250`
- Modify: `voxbridge/tts/listener_page.py:559-890,1120-1415`

**Interfaces:**
- Consumes: shared HLS media timeline, caption snapshot `cues`, native media `waiting`, `stalled`, and `playing` events.
- Produces: the existing Listen UI and playback-rate API with no local `currentTime` assignment for caption gaps.

- [ ] **Step 1: Replace compaction expectations with a failing no-seek behavior test**

Rename `_start_historical_silence_harness` to `_start_shared_timeline_gap_harness`. Remove the parameterized compaction tests and add:

```python
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
```

Delete these obsolete policy tests because the corresponding client policy no longer exists:

- `test_auto_compaction_requires_two_seconds_of_next_speech_buffer`
- `test_auto_compaction_returns_to_normal_rate`
- `test_auto_compaction_retries_playback_when_seek_stays_waiting`
- `test_manual_fast_compaction_retains_four_second_guard`

The replacement test is behavioral: current production seeks from 50.0 to 52.5, so it must fail before implementation.

- [ ] **Step 2: Run the browser test and verify RED**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_listener_page_browser.py::test_listener_keeps_shared_playhead_when_future_speech_is_buffered -q
```

Expected: FAIL with current time approximately 52.5 instead of 50.0.

- [ ] **Step 3: Remove the local compaction state machine from the Listen page**

Delete the constants:

```javascript
HISTORICAL_SILENCE_MIN_GAP_MS
HISTORICAL_SILENCE_KEEP_MS
AUTO_NEXT_SPEECH_BUFFER_GUARD_SEC
NEXT_SPEECH_BUFFER_GUARD_SEC
SILENCE_COMPACTION_RECOVERY_MS
```

Delete the state:

```javascript
compactedNextCueKey
pendingSilenceCompaction
silenceCompactionRecoveryTimer
```

Delete these functions in full:

```javascript
mediaTimeIsBuffered
clearPendingSilenceCompaction
scheduleSilenceCompactionRecovery
requestPendingSilencePlayback
compactHistoricalSilence
```

In `applyCaptionSnapshot()`, retain caption selection but remove both compaction branches:

```javascript
      if (selected === null) {
        if (!captionCueId) setLiveCaption("Waiting for translated speech");
        nowPlaying.dataset.speaking = "false";
        return;
      }
      setLiveCaption(selected.text, selected.cue_id);
```

Remove compaction cleanup from `markPlaying()` and `stopListening()`. Remove the `seeked` event listener. Reduce `waiting` and `stalled` handlers to their ordinary live-edge branches; neither handler may call a seek-recovery function or replace the media source.

- [ ] **Step 4: Run focused browser regressions and verify GREEN**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_listener_page_browser.py::test_listener_keeps_shared_playhead_when_future_speech_is_buffered \
  tests/test_tts_listener_page_browser.py::test_waiting_resets_fast_selection_to_normal_rate \
  tests/test_tts_listener_page_browser.py::test_waiting_status_remains_visible_after_playback_resumes \
  tests/test_tts_listener_page_browser.py::test_stalled_resets_fast_selection_to_normal_rate -q
```

Expected: all four pass. The wait/resume tests prove a short live-edge stall remains recoverable without a local seek.

- [ ] **Step 5: Run the complete listener browser file**

```bash
../.venv/bin/python -m pytest tests/test_tts_listener_page_browser.py -q
```

Expected: all tests pass with no unexpected browser console or playback errors.

- [ ] **Step 6: Commit the listener behavior**

```bash
git add voxbridge/tts/listener_page.py tests/test_tts_listener_page_browser.py
git commit -m "fix: play the shared HLS timeline without local silence seeks"
```

---

### Task 3: Verify concurrency, deploy, and measure the production timeline

**Files:**
- Verify only: `voxbridge/tts/hls.py`
- Verify only: `voxbridge/tts/listener_page.py`
- Verify only: `tests/test_tts_hls.py`
- Verify only: `tests/test_tts_listener_page_browser.py`
- Verify only: `tests/test_demo_streaming_ws_protocol.py`

**Interfaces:**
- Consumes: the completed Tasks 1 and 2, `voxbridge-8024.service`, `voxbridge-translation.service`, port 8024, and `/api/tts/live/status`.
- Produces: a deployed epoch with one encoder, listener-scoped access to shared media files, and evidence that idle wall-clock time no longer appears as multi-second HLS silence.

- [ ] **Step 1: Run the concurrency and HTTP protocol regression group**

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py \
  tests/test_demo_streaming_ws_protocol.py -q
```

Expected: all tests pass, including one synthesis/encoder for multiple listeners, public listener capability scoping, listener capacity, captions, and status.

- [ ] **Step 2: Run the full automated suite**

```bash
../.venv/bin/python -m pytest -q
```

Expected: zero failures. Record the exact passed/skipped count in the completion report.

- [ ] **Step 3: Inspect the final change set before deployment**

```bash
git status --short
git diff --check HEAD~2..HEAD
git diff --stat HEAD~2..HEAD
git log -3 --oneline
```

Expected: only the two implementation commits plus the approved design/plan commits are present; no whitespace errors or unrelated files appear.

- [ ] **Step 4: Restart only the public VoxBridge service and verify runtime health**

```bash
systemctl --user restart voxbridge-8024.service
systemctl --user is-active voxbridge-8024.service voxbridge-translation.service
ss -lntp | rg ':8024'
curl -fsS http://127.0.0.1:8024/api/tts/live/status
```

Expected: both services are `active`, port 8024 is listening, `available=true`, `producer_active=true` while translation is connected, and `last_error` is empty.

- [ ] **Step 5: Verify multiple listener leases still share one encoder**

Open two listener sessions with distinct IDs, keep both playlists polling, then run:

```bash
curl -fsS http://127.0.0.1:8024/api/tts/live/status
pgrep -af 'ffmpeg.*voxbridge-tts-hls-8024'
```

Expected: `listener_count` reports both sessions, exactly one FFmpeg process exists, and queue/error fields remain healthy. Normalize the listener ID inside each returned playlist URL and confirm both playlist bodies reference the same media sequence and segment filenames.

- [ ] **Step 6: Prove the production playlist stops advancing during an idle translation gap**

With at least one listener lease active, save the current playlist body and parsed final `EXT-X-PROGRAM-DATE-TIME`/`EXTINF` live edge, wait longer than two segment durations while no translated PCM is pending, and fetch again.

Expected: the playlist body and live edge remain unchanged, `pending_audio_ms=0`, the encoder process stays alive, and the listener lease remains valid.

- [ ] **Step 7: Prove the same epoch resumes on the next released sentence**

Continue the same listener sessions until the translation service releases another stable sentence. Fetch the playlists and status again.

Expected: media sequence/live edge advance in the same epoch, no FFmpeg restart occurs, both listeners receive the same new segment names, and playback resumes after normal slice/network buffering without a `currentTime` seek.

- [ ] **Step 8: Measure shared-stream silence and check service logs**

Decode the most recent shared HLS window with FFmpeg `silencedetect` and inspect service logs:

```bash
journalctl --user -u voxbridge-8024.service --since '10 minutes ago' --no-pager \
  | rg 'shared HLS|TTS audio published|ERROR|Traceback'
```

Expected: no repeated multi-second carrier spans are added between sentence releases, no traceback/error is present, and each released sentence has one publish event regardless of listener count. Synthesizer edge silence and the configured 300 ms sentence pause are allowed.

- [ ] **Step 9: Concurrent device acceptance**

Keep the iPhone native-HLS Listen page and Windows Chrome/HLS.js Listen page in Auto mode through several sentence releases and at least one long translation-idle interval.

Expected: both hear the same sentence order and encoded pauses; neither waits for a client-side two- or four-second next-speech guard; each may show one short live-edge buffering event bounded by normal segment/network behavior; no sentence-start interruption returns.

- [ ] **Step 10: Report evidence and leave rollback ready**

Report exact commit IDs, test counts, service state, listener count, FFmpeg process count, measured idle/resume behavior, and any remaining device latency difference. Do not claim the fix complete unless every automated check passes and the observable server timeline meets the acceptance criteria. If production playback does not resume after a long unchanged playlist, roll back both server and listener commits together and investigate control-plane playlist reload without reintroducing audio carrier.
