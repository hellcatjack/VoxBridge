# Global Adaptive TTS Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move listener acceleration from unreliable per-browser playback rates to one epoch-scoped server-side Kokoro Auto speed shared by every listener.

**Architecture:** `SharedHLSTTSPublisher` owns a speech epoch, conservative unpublished-speech backlog, and global Auto multiplier. It passes an absolute per-call speed to the existing serialized `KokoroOnnxSynthesizer`, publishes one HLS stream, and exposes read-only speed/backlog status while every browser plays at HTML media rate `1.0`.

**Tech Stack:** Python 3, asyncio, FastAPI, Kokoro ONNX, FFmpeg HLS, vanilla JavaScript, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-01-global-adaptive-tts-speed.md`

## Global Constraints

- Work directly on the existing `main` checkout, as previously authorized; preserve unrelated user changes.
- Run every Python command with `/data/Qwen3-ASR/.venv/bin/python`, normally written from the repository as `../.venv/bin/python`.
- The service remains on port `8024`; do not use `8000`, `8001`, or `8080`.
- With zero listeners, perform no TTS synthesis and report zero active speech backlog.
- On `0 -> 1` listeners, retain only the newest eligible stable item; its displayed multiplier is `1.0x`.
- Use exactly these Auto tiers: `<10s = 1.0x`, `10-<30s = 1.2x`, `30-<40s = 1.4x`, `>=40s = 1.5x`.
- Preserve the configured Kokoro baseline of `1.05`; displayed `1.0x` means effective Kokoro speed `1.05`.
- Select speed once per sentence and never retime PCM already synthesized, appended, or published.
- All listener HTML media elements use `defaultPlaybackRate = 1.0` and `playbackRate = 1.0`.
- Do not add custom PCM time stretching, parallel per-rate HLS streams, or per-listener TTS jobs.
- A listener joining, leaving, buffering, or expiring cannot change controller ownership while at least one listener remains.
- Keep hls.js live-sync playback-rate adjustment disabled with `maxLiveSyncPlaybackRate: 1`.

## File Structure

- `voxbridge/tts/kokoro_onnx.py`: validate and pass an absolute optional speed per serialized synthesis call.
- `voxbridge/tts/hls.py`: own epoch lifecycle, conservative backlog, Auto tier selection, speed-aware prepared audio, logging, and shared status.
- `voxbridge/cli/demo_streaming_ws.py`: pass baseline/feature configuration and serialize the new shared status fields.
- `voxbridge/tts/listener_page.py`: remove client rate control and render read-only global speed/backlog while forcing media rate `1.0`.
- `tests/test_kokoro_tts.py`: per-call Kokoro speed and validation tests.
- `tests/test_tts_hls.py`: epoch, backlog, threshold, multi-listener, speed application, and prepared-cache tests.
- `tests/test_demo_streaming_ws_protocol.py`: public status JSON and multi-listener continuity tests.
- `tests/test_demo_streaming_ws_utils.py`: CLI configuration and rollback-switch tests.
- `tests/test_tts_listener_page_browser.py`: fixed client rate, shared status copy, buffering, and HLS configuration tests.
- `README.md`, `docs/API.md`, `docs/DEPLOYMENT.md`: operator-visible global Auto behavior and rollback documentation.

---

### Task 1: Add a validated per-call Kokoro speed

**Files:**
- Modify: `voxbridge/tts/kokoro_onnx.py:1-186`
- Test: `tests/test_kokoro_tts.py:75-168`
- Add to Git: `docs/superpowers/specs/2026-09-01-global-adaptive-tts-speed.md`
- Add to Git: `docs/superpowers/plans/2026-09-01-global-adaptive-tts-speed.md`

**Interfaces:**
- Consumes: `KokoroTTSConfig.speed: float` as the fixed baseline and default.
- Produces: `KokoroOnnxSynthesizer.synthesize(text: str, target_language: str, *, speed: float | None = None) -> SynthesizedAudio`.
- Produces: one serialized `model.create(..., speed=effective_speed)` call; no model recreation.

- [ ] **Step 1: Write failing per-call speed tests**

Add tests that prove the override reaches Kokoro and the configured baseline remains the default:

```python
def test_synthesis_accepts_absolute_per_call_speed(tmp_path):
    factory = FakeFactory()
    synth = KokoroOnnxSynthesizer(
        config=make_config(tmp_path, speed=1.05),
        kokoro_factory=factory,
    )

    synth.synthesize("Catch up now.", "English", speed=1.575)
    synth.synthesize("Back at baseline.", "English")

    assert factory.models[0].calls[0].speed == pytest.approx(1.575)
    assert factory.models[0].calls[1].speed == pytest.approx(1.05)


@pytest.mark.parametrize("speed", [0.49, 2.01, float("inf"), float("nan")])
def test_synthesis_rejects_invalid_per_call_speed(tmp_path, speed):
    synth = KokoroOnnxSynthesizer(
        config=make_config(tmp_path),
        kokoro_factory=FakeFactory(),
    )

    with pytest.raises(TTSSynthesisError, match="speed"):
        synth.synthesize("Invalid speed.", "English", speed=speed)
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
../.venv/bin/python -m pytest \
  tests/test_kokoro_tts.py::test_synthesis_accepts_absolute_per_call_speed \
  tests/test_kokoro_tts.py::test_synthesis_rejects_invalid_per_call_speed -q
```

Expected: FAIL because `synthesize()` does not accept the `speed` keyword.

- [ ] **Step 3: Implement minimal per-call validation and forwarding**

Import `math`, extend the method signature, and derive the absolute speed before acquiring the existing inference lock:

```python
def synthesize(
    self,
    text: str,
    target_language: str,
    *,
    speed: float | None = None,
) -> SynthesizedAudio:
    effective_speed = self.config.speed if speed is None else float(speed)
    if not math.isfinite(effective_speed) or not 0.5 <= effective_speed <= 2.0:
        raise TTSSynthesisError("TTS speed must be between 0.5 and 2.0")
```

Change only the Kokoro call argument:

```python
samples, sample_rate = model.create(
    model_input,
    voice=voice,
    speed=effective_speed,
    lang=lang,
    is_phonemes=is_phonemes,
)
```

- [ ] **Step 4: Run the Kokoro adapter suite**

Run:

```bash
../.venv/bin/python -m pytest tests/test_kokoro_tts.py -q
```

Expected: all tests PASS and existing calls without `speed` still use `config.speed`.

- [ ] **Step 5: Commit the adapter and approved documents**

```bash
git add voxbridge/tts/kokoro_onnx.py tests/test_kokoro_tts.py
git add -f \
  docs/superpowers/specs/2026-09-01-global-adaptive-tts-speed.md \
  docs/superpowers/plans/2026-09-01-global-adaptive-tts-speed.md
git commit -m "feat: support per-sentence Kokoro speed"
```

---

### Task 2: Introduce the shared speech epoch and conservative Auto controller

**Files:**
- Modify: `voxbridge/tts/hls.py:79-124`
- Modify: `voxbridge/tts/hls.py:609-846`
- Modify: `voxbridge/tts/hls.py:973-999`
- Modify: `voxbridge/tts/hls.py:1188-1217`
- Test: `tests/test_tts_hls.py:23-294`
- Test: `tests/test_tts_hls.py:611-810`

**Interfaces:**
- Consumes: `baseline_tts_speed: float = 1.05` and `auto_speed_enabled: bool = True` constructor values.
- Produces: `select_global_tts_multiplier(backlog_ms: int) -> float`.
- Produces: `_retain_latest_idle_item_locked() -> tuple[int, ItemKey | None]`, where the second element identifies the retained first-epoch item.
- Produces: `HLSStreamStatus.speech_epoch_id`, `global_speed_mode`, `global_speed_multiplier`, and `tts_effective_speed`.
- Produces: `_backlog_snapshot() -> tuple[int, int, int, bool]`, returning encoder-pending milliseconds, conservative total milliseconds, unique item count, and whether any component is estimated.
- Produces: idle status with backlog/count zero regardless of retained idle queue contents.

- [ ] **Step 1: Write failing threshold and epoch-lifecycle tests**

Update `FakeSynthesizer` so later tests can capture the absolute speed:

```python
def synthesize(self, text: str, target_language: str, *, speed: float | None = None):
    self.calls.append((text, target_language, speed))
    return SimpleNamespace(
        wav_bytes=self.wav_bytes,
        sample_rate=24000,
        duration_ms=self.duration_ms,
    )
```

Add exact boundary coverage:

```python
@pytest.mark.parametrize(
    ("backlog_ms", "expected"),
    [
        (0, 1.0),
        (9_999, 1.0),
        (10_000, 1.2),
        (29_999, 1.2),
        (30_000, 1.4),
        (39_999, 1.4),
        (40_000, 1.5),
    ],
)
def test_global_tts_multiplier_boundaries(backlog_ms, expected):
    assert select_global_tts_multiplier(backlog_ms) == expected
```

Add one lifecycle test that proves idle history is not debt and ownership is not tied to the first listener:

```python
@pytest.mark.asyncio
async def test_speech_epoch_skips_idle_debt_and_survives_first_listener_exit(tmp_path):
    synth = FakeSynthesizer(make_wav())
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: FakeEncoder(root),
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    try:
        await publisher.publish(ready_item(0))
        await publisher.publish(ready_item(1))
        assert publisher.status.speech_epoch_id == ""
        assert publisher.status.translated_audio_backlog_ms == 0
        assert publisher.status.translated_audio_backlog_count == 0

        await publisher.touch_listener("iphone-a", "owner-a")
        epoch = publisher.status.speech_epoch_id
        assert epoch.startswith("epoch-")
        await publisher.touch_listener("chrome-b", "owner-b")
        await publisher.wait_idle()
        assert [(text, language) for text, language, _ in synth.calls] == [
            ("Stable translation 1.", "English")
        ]

        await publisher.remove_listener("iphone-a", "owner-a")
        assert publisher.status.speech_epoch_id == epoch
        assert publisher.status.listener_count == 1

        await publisher.remove_listener("chrome-b", "owner-b")
        assert publisher.status.speech_epoch_id == ""
        assert publisher.status.global_speed_multiplier == 1.0
        assert publisher.status.translated_audio_backlog_ms == 0
    finally:
        await publisher.close()
```

Update existing `synth.calls` assertions to compare `(text, language)` projections during this task. Task 3 will add assertions for the third, absolute-speed element after the worker starts passing it.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py::test_global_tts_multiplier_boundaries \
  tests/test_tts_hls.py::test_speech_epoch_skips_idle_debt_and_survives_first_listener_exit -q
```

Expected: FAIL because the selector, epoch fields, constructor arguments, and idle-status semantics do not exist.

- [ ] **Step 3: Add status fields and the pure tier selector**

Define an internal item-key alias and the public pure function in `hls.py`:

```python
ItemKey = tuple[str, int, str, str]


def select_global_tts_multiplier(backlog_ms: int) -> float:
    value = max(0, int(backlog_ms))
    if value >= 40_000:
        return 1.5
    if value >= 30_000:
        return 1.4
    if value >= 10_000:
        return 1.2
    return 1.0
```

Extend `HLSStreamStatus` with:

```python
speech_epoch_id: str
global_speed_mode: str
global_speed_multiplier: float
tts_effective_speed: float
```

Add publisher state initialized as:

```python
self._baseline_tts_speed = float(baseline_tts_speed)
self._auto_speed_enabled = bool(auto_speed_enabled)
self._speech_epoch_id = ""
self._global_speed_multiplier = 1.0
self._first_epoch_item_key: ItemKey | None = None
```

Validate the baseline as finite and between `0.5` and `2.0`.

- [ ] **Step 4: Make backlog epoch-scoped and conservative**

Implement `_backlog_snapshot() -> tuple[int, int, int, bool]` and start it with this idle guard:

```python
if not self._leases or self._encoder is None or not self._speech_epoch_id:
    return 0, 0, 0, False
```

For each unique `_known_items` entry, use exact `_PreparedAudio.audio_ms` when present; otherwise use `_estimate_item_audio_ms(item)`. Change `_estimate_item_audio_ms` to use at least the default language value and a safety-adjusted observation:

```python
observed_ms_per_char = self._audio_ms_per_char.get(language, 0.0)
ms_per_char = max(default_ms_per_char, observed_ms_per_char * 1.10)
```

The status property must calculate:

```python
effective_speed = self._baseline_tts_speed * self._global_speed_multiplier
```

and return zero debt when idle even if `_queue` retains unpublished translations.

- [ ] **Step 5: Create and clear epoch state atomically**

Change `_retain_latest_idle_item_locked()` to return both dropped count and retained key. During `touch_listener`, after the encoder starts successfully:

```python
self._active_root = root
self._speech_epoch_id = root.name
self._global_speed_multiplier = 1.0
self._first_epoch_item_key = retained_key
self._encoder = encoder
```

During `_stop_stream`, clear all four epoch/controller values under the same lock as encoder removal:

```python
self._speech_epoch_id = ""
self._global_speed_multiplier = 1.0
self._first_epoch_item_key = None
```

Drain retained queue entries on last-listener shutdown so the ended epoch cannot leak debt; later translations can repopulate the idle queue normally.

```python
while True:
    try:
        dropped_item = self._queue.get_nowait()
    except asyncio.QueueEmpty:
        break
    self._queue.task_done()
    self._known_items.pop(self._item_key(dropped_item), None)
```

- [ ] **Step 6: Run shared-publisher lifecycle and backlog tests**

```bash
../.venv/bin/python -m pytest tests/test_tts_hls.py -q
```

Expected: all publisher, lease, caption, backlog, and FFmpeg HLS tests PASS.

- [ ] **Step 7: Commit epoch and controller state**

```bash
git add voxbridge/tts/hls.py tests/test_tts_hls.py
git commit -m "feat: add shared TTS speed epoch controller"
```

---

### Task 3: Apply global speed per sentence and make prepared audio speed-safe

**Files:**
- Modify: `voxbridge/tts/hls.py:117-124`
- Modify: `voxbridge/tts/hls.py:730-757`
- Modify: `voxbridge/tts/hls.py:1001-1164`
- Test: `tests/test_tts_hls.py:267-590`
- Test: `tests/test_tts_hls.py:611-688`

**Interfaces:**
- Consumes: `select_global_tts_multiplier(backlog_ms)` and Task 2 epoch state.
- Consumes: Task 1 `synthesize(..., speed=absolute_speed)`.
- Produces: `_PreparedAudio.displayed_multiplier: float` and `_PreparedAudio.effective_speed: float`.
- Produces: `_select_synthesis_speed_locked(key: ItemKey, kind: str) -> tuple[float, float, int]`, returning displayed multiplier, absolute Kokoro speed, and backlog used for the decision.
- Produces: speed-mismatched prepared audio is discarded and regenerated before release.

- [ ] **Step 1: Write failing worker speed tests**

Add a test for server-side `1.5x` selection with an active epoch:

```python
@pytest.mark.asyncio
async def test_worker_applies_global_multiplier_as_absolute_kokoro_speed(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 40_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.05,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.publish(ready_item(text="Accelerated together."))
        await publisher.wait_idle()

        assert synth.calls == [("Accelerated together.", "English", 1.575)]
        assert publisher.status.global_speed_multiplier == 1.5
    finally:
        await publisher.close()
```

Add cache mismatch coverage:

```python
@pytest.mark.asyncio
async def test_release_regenerates_prepared_audio_when_global_speed_changed(tmp_path):
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
    item = ready_item(text="Prepared at the old speed.")
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        assert await publisher.prepare(item) is True
        await wait_until(lambda: publisher.status.prepared_audio_count == 1)
        assert synth.calls[-1][2] == pytest.approx(1.26)

        encoder.pending_audio_ms = 0
        assert await publisher.publish(item) is True
        await publisher.wait_idle()

        assert [call[2] for call in synth.calls] == pytest.approx([1.26, 1.05])
        assert len(encoder.appended) == 1
    finally:
        await publisher.close()
```

Add invalid-effective-speed fallback coverage using `baseline_tts_speed=1.5` and a `>=40s` backlog:

```python
@pytest.mark.asyncio
async def test_invalid_effective_speed_falls_back_without_stopping_epoch(tmp_path):
    synth = FakeSynthesizer(make_wav())
    encoder = FakeEncoder(tmp_path / "stream")
    encoder.pending_audio_ms = 40_000
    publisher = SharedHLSTTSPublisher(
        synthesizer=synth,
        encoder_factory=lambda root: encoder,
        root_dir=tmp_path,
        baseline_tts_speed=1.5,
        clock=FakeClock(),
    )
    try:
        await publisher.touch_listener("iphone-a", "owner-a")
        await publisher.publish(ready_item(text="Stay in range."))
        await publisher.wait_idle()

        assert synth.calls[-1][2] == pytest.approx(1.5)
        assert publisher.status.global_speed_multiplier == 1.0
        assert publisher.status.speech_epoch_id.startswith("epoch-")
    finally:
        await publisher.close()
```

- [ ] **Step 2: Run focused worker tests and verify failure**

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_hls.py::test_worker_applies_global_multiplier_as_absolute_kokoro_speed \
  tests/test_tts_hls.py::test_release_regenerates_prepared_audio_when_global_speed_changed \
  tests/test_tts_hls.py::test_invalid_effective_speed_falls_back_without_stopping_epoch -q
```

Expected: FAIL because the worker neither chooses nor passes a per-item speed and prepared audio has no speed metadata.

- [ ] **Step 3: Store synthesis speed with prepared PCM**

Extend `_PreparedAudio`:

```python
displayed_multiplier: float
effective_speed: float
```

When observing generated duration, normalize the speech portion back toward the displayed `1.0x` baseline before updating the language estimate:

```python
baseline_speech_ms = speech_audio_ms * max(1.0, displayed_multiplier)
observed = baseline_speech_ms / text_chars
```

This prevents fast synthesized sentences from teaching the conservative baseline estimator that future speech is artificially short.

- [ ] **Step 4: Select one safe speed per worker item**

Import `math`. Under the publisher lock, calculate the conservative backlog while the selected item remains in `_known_items`. Implement:

```python
def _select_synthesis_speed_locked(
    self,
    key: ItemKey,
    kind: str,
) -> tuple[float, float, int]:
    _, backlog_ms, _, _ = self._backlog_snapshot()
    force_join_baseline = kind == "release" and key == self._first_epoch_item_key
    multiplier = 1.0 if force_join_baseline else (
        select_global_tts_multiplier(backlog_ms)
        if self._auto_speed_enabled
        else 1.0
    )
    effective = self._baseline_tts_speed * multiplier
    if not math.isfinite(effective) or not 0.5 <= effective <= 2.0:
        self._last_error = "TTSSpeedRangeError"
        multiplier = 1.0
        effective = self._baseline_tts_speed
    self._global_speed_multiplier = multiplier
    if force_join_baseline:
        self._first_epoch_item_key = None
    return multiplier, effective, backlog_ms
```

Use the returned absolute value in the existing serialized thread call:

```python
audio = await asyncio.to_thread(
    self._synthesizer.synthesize,
    item.text,
    item.target_language,
    speed=effective_speed,
)
```

- [ ] **Step 5: Enforce prepared-cache speed identity**

Before marking `cache_hit`, compare the prepared result's effective speed with the selected speed. On mismatch, discard it and synthesize again:

```python
if prepared is not None and not math.isclose(
    prepared.effective_speed,
    effective_speed,
    rel_tol=0.0,
    abs_tol=1e-9,
):
    prepared = None
cache_hit = prepared is not None
```

Store the new multiplier and effective speed in `_PreparedAudio`.

Expand preparation and publication log messages with:

```text
epoch=%s backlog_ms=%d multiplier=%.1f effective_speed=%.3f speed_cache_hit=%s
```

Log the idle-skipped count at epoch creation and the stop reason at final lease removal or expiry.

- [ ] **Step 6: Run all Kokoro and HLS tests**

```bash
../.venv/bin/python -m pytest tests/test_kokoro_tts.py tests/test_tts_hls.py -q
```

Expected: all tests PASS, including revision invalidation and preparation reuse at an unchanged speed.

- [ ] **Step 7: Commit speed-aware synthesis and caching**

```bash
git add voxbridge/tts/hls.py tests/test_tts_hls.py
git commit -m "feat: synthesize shared HLS speech at global speed"
```

---

### Task 4: Expose shared speed through application configuration and status

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py:5365-5405`
- Modify: `voxbridge/cli/demo_streaming_ws.py:5707-5734`
- Modify: `voxbridge/cli/demo_streaming_ws.py:12869-12882`
- Test: `tests/test_demo_streaming_ws_protocol.py:94-100`
- Test: `tests/test_demo_streaming_ws_protocol.py:2296-2333`
- Test: `tests/test_demo_streaming_ws_utils.py:1526-1559`

**Interfaces:**
- Consumes: Task 2 `SharedHLSTTSPublisher(..., baseline_tts_speed, auto_speed_enabled)`.
- Produces: CLI flag `--disable-tts-global-auto-speed`, default false.
- Produces: `/api/tts/live/status` fields `speech_epoch_id`, `global_speed_mode`, `global_speed_multiplier`, and `tts_effective_speed`.
- Produces: idle status with empty epoch, `1.0` multiplier, and zero translated-audio debt.

- [ ] **Step 1: Write failing application-contract tests**

Update `_FakeTTSSynthesizer.synthesize` to accept `*, speed=None` and record it. Extend the existing status test:

```python
assert status["speech_epoch_id"].startswith("epoch-")
assert status["global_speed_mode"] == "auto"
assert status["global_speed_multiplier"] == 1.0
assert status["tts_effective_speed"] == pytest.approx(1.05)
```

Add an idle endpoint assertion before the first playlist request:

```python
idle = client.get("/api/tts/live/status").json()
assert idle["speech_epoch_id"] == ""
assert idle["translated_audio_backlog_ms"] == 0
assert idle["translated_audio_backlog_count"] == 0
```

Add this explicit application configuration test so `tts_speed=1.1` becomes the publisher baseline and the rollback switch produces fixed mode:

```python
def test_shared_hls_uses_configured_baseline_and_global_auto_rollback(tmp_path):
    args = _args()
    args.tts_hls_root_dir = str(tmp_path)
    args.tts_speed = 1.1
    args.disable_tts_global_auto_speed = True
    args.tts_hls_encoder_factory = lambda root: _FakeHLSEncoder(root)

    app = _create_app(args, _FakeASR(), tts_synthesizer=_FakeTTSSynthesizer())

    status = app.state.tts_hls.status
    assert status.global_speed_mode == "fixed"
    assert status.global_speed_multiplier == 1.0
    assert status.tts_effective_speed == pytest.approx(1.1)
```

- [ ] **Step 2: Run protocol and configuration tests and verify failure**

```bash
../.venv/bin/python -m pytest \
  tests/test_demo_streaming_ws_protocol.py::test_removing_one_hls_listener_does_not_stop_shared_encoder \
  tests/test_demo_streaming_ws_utils.py::test_build_tts_synthesizer_is_optional_and_maps_cli_config -q
```

Expected: FAIL because the new status fields and rollback configuration are absent.

- [ ] **Step 3: Wire publisher configuration**

Add the argument:

```python
p.add_argument(
    "--disable-tts-global-auto-speed",
    action="store_true",
    help="Use the configured fixed Kokoro speed for shared listener audio",
)
```

Pass configuration into the publisher:

```python
baseline_tts_speed=float(getattr(args, "tts_speed", 1.05)),
auto_speed_enabled=not bool(
    getattr(args, "disable_tts_global_auto_speed", False)
),
```

- [ ] **Step 4: Serialize the shared status fields**

Add to the JSON response without changing listener-specific endpoints:

```python
"speech_epoch_id": str(status.speech_epoch_id),
"global_speed_mode": str(status.global_speed_mode),
"global_speed_multiplier": float(status.global_speed_multiplier),
"tts_effective_speed": float(status.tts_effective_speed),
```

- [ ] **Step 5: Run application tests**

```bash
../.venv/bin/python -m pytest \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py -q
```

Expected: all tests PASS and removing the first of two listeners leaves the same active epoch.

- [ ] **Step 6: Commit configuration and API changes**

```bash
git add \
  voxbridge/cli/demo_streaming_ws.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py
git commit -m "feat: expose global TTS speed status"
```

---

### Task 5: Make the listener a fixed-rate consumer of shared Auto audio

**Files:**
- Modify: `voxbridge/tts/listener_page.py:490-590`
- Modify: `voxbridge/tts/listener_page.py:598-673`
- Modify: `voxbridge/tts/listener_page.py:832-935`
- Modify: `voxbridge/tts/listener_page.py:954-985`
- Modify: `voxbridge/tts/listener_page.py:1027-1057`
- Modify: `voxbridge/tts/listener_page.py:1164-1273`
- Test: `tests/test_tts_listener_page_browser.py:46-230`
- Replace obsolete tests: `tests/test_tts_listener_page_browser.py:271-449`
- Replace obsolete tests: `tests/test_tts_listener_page_browser.py:733-996`

**Interfaces:**
- Consumes: Task 4 shared status JSON.
- Produces: read-only element `#globalSpeedStatus` with text `Auto - 1.0x`, `Auto - 1.2x`, `Auto - 1.4x`, or `Auto - 1.5x`.
- Produces: `forceNormalPlaybackRate()` which always writes `1` to both media-rate properties.
- Produces: Live Audio copy based on server backlog only; client lag remains caption/playhead metadata, not speech debt.

- [ ] **Step 1: Replace obsolete browser-rate tests with failing fixed-rate tests**

Extend the harness status payload:

```javascript
speech_epoch_id: "epoch-test",
global_speed_mode: "auto",
global_speed_multiplier: 1.5,
tts_effective_speed: 1.575,
```

Update the fake Hls constructor to retain options:

```javascript
constructor(options = {}) {
  this.options = options;
  // retain the existing harness fields
}
```

Add these browser tests:

```python
def test_listener_renders_read_only_global_auto_speed(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=40_000,
        global_speed_multiplier=1.5,
    )
    _start_hls_harness(listener_page)

    assert listener_page.locator("#playbackRate").count() == 0
    assert listener_page.text_content("#globalSpeedStatus") == "Auto - 1.5x"
    assert listener_page.eval_on_selector(
        "#ttsPlayback", "node => [node.defaultPlaybackRate, node.playbackRate]"
    ) == [1, 1]


def test_listener_never_accelerates_media_for_backlog_or_progress(listener_page):
    _install_hls_harness(
        listener_page,
        translated_audio_backlog_ms=61_000,
        global_speed_multiplier=1.5,
    )
    _start_hls_harness(listener_page)
    _set_live_lag(listener_page, current_time=50, live_edge=100)
    _set_buffered_range(listener_page, start=0, end=100)

    playback = listener_page.locator("#ttsPlayback")
    playback.dispatch_event("timeupdate")
    playback.dispatch_event("progress")
    assert playback.evaluate("node => node.playbackRate") == 1
    assert "Speech backlog: 1m 1s" in listener_page.text_content("#playbackStatus")
    assert "Global speed: Auto - 1.5x" in listener_page.text_content("#playbackStatus")


def test_hls_js_live_rate_adjustment_is_disabled(listener_page):
    _install_hls_harness(
        listener_page,
        native_hls=False,
        hls_js_supported=True,
    )
    _start_hls_harness(listener_page)

    assert listener_page.evaluate(
        "window.__ttsHlsInstances[0].options.maxLiveSyncPlaybackRate"
    ) == 1
```

Add harness parameter `global_speed_multiplier: float = 1.0`. Delete tests that select or persist `#playbackRate`; retain layout, caption, lease, native-HLS, hls.js fallback, Media Session, and start/stop coverage.

- [ ] **Step 2: Run the focused browser tests and verify failure**

```bash
../.venv/bin/python -m pytest \
  tests/test_tts_listener_page_browser.py::test_listener_renders_read_only_global_auto_speed \
  tests/test_tts_listener_page_browser.py::test_listener_never_accelerates_media_for_backlog_or_progress \
  tests/test_tts_listener_page_browser.py::test_hls_js_live_rate_adjustment_is_disabled -q
```

Expected: FAIL because the page still has a selectable client rate and applies backlog-derived playback rates.

- [ ] **Step 3: Replace the selector with read-only shared status**

Replace the `<select>` with:

```html
<div class="playback-settings" aria-live="polite">
  <span>Playback speed</span>
  <strong id="globalSpeedStatus">Auto - 1.0x</strong>
</div>
```

Remove local-storage rate state, supported manual rates, client Auto thresholds, catch-up flags, and forward-buffer hysteresis used only for rate selection.

- [ ] **Step 4: Force browser media rate to one**

Implement and call this during initialization, source setup, start, resume, `playing`, `waiting`, `stalled`, `progress`, and `timeupdate`:

```javascript
function forceNormalPlaybackRate() {
  playbackElement.defaultPlaybackRate = 1;
  playbackElement.playbackRate = 1;
}
```

Instantiate hls.js with:

```javascript
hlsController = new Hls({ maxLiveSyncPlaybackRate: 1 });
```

Keep current native-HLS preference and fallback selection unchanged.

- [ ] **Step 5: Render unambiguous shared and local status**

In `pollStatus`, read only the server backlog and global multiplier:

```javascript
serverTranslatedAudioBacklogSec = Math.max(0, serverBacklogMs) / 1000;
globalSpeedMultiplier = Number.isFinite(Number(status.global_speed_multiplier))
  ? Number(status.global_speed_multiplier)
  : 1;
globalSpeedStatus.textContent = `Auto - ${globalSpeedMultiplier.toFixed(1)}x`;
```

Use exact Live Audio copy:

```javascript
`Speech backlog: ${formatDurationSec(serverTranslatedAudioBacklogSec)}`
  + ` · Global speed: Auto - ${globalSpeedMultiplier.toFixed(1)}x`
```

Do not add `liveLagSec()` to server speech backlog. Keep `Buffering live audio` for startup before successful playback; keep `Preparing next translated sentence` for a real speech gap when server debt exists; keep `Waiting for translated speech` when it does not.

- [ ] **Step 6: Run the complete listener browser suite**

```bash
../.venv/bin/python -m pytest tests/test_tts_listener_page_browser.py -q
```

Expected: all fixed-rate, layout, caption, HLS fallback, lifecycle, and Media Session tests PASS.

- [ ] **Step 7: Commit the listener change**

```bash
git add voxbridge/tts/listener_page.py tests/test_tts_listener_page_browser.py
git commit -m "fix: share server-side Auto speed across listeners"
```

---

### Task 6: Document, verify, deploy, and observe the complete flow

**Files:**
- Modify: `README.md:145-160`
- Modify: `docs/API.md` near `/api/tts/live/status`
- Modify: `docs/DEPLOYMENT.md:574-589`
- Verify: all modified production and test files from Tasks 1-5

**Interfaces:**
- Consumes: complete shared epoch, speed, status, and fixed-rate listener flow.
- Produces: operator instructions for status interpretation and `--disable-tts-global-auto-speed` rollback.
- Produces: deployed `voxbridge-8024.service` verified on port `8024`.

- [ ] **Step 1: Write the operator-facing documentation**

Document these exact semantics:

- No listener means no TTS synthesis and zero active speech backlog.
- The first listener skips old translations and retains only the newest stable sentence.
- All listeners hear one global Auto rate selected from `10s`, `30s`, and `40s` boundaries.
- `Speech backlog` is unpublished shared audio, not one phone's network delay.
- Client media playback remains `1.0`; Kokoro generates the faster shared speech.
- `--disable-tts-global-auto-speed` keeps fixed baseline server synthesis for rollback.

Add the four new status fields to `docs/API.md` with their idle values.

- [ ] **Step 2: Run formatting and focused regression checks**

```bash
git diff --check
../.venv/bin/python -m pytest \
  tests/test_kokoro_tts.py \
  tests/test_tts_hls.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py \
  tests/test_tts_listener_page_browser.py -q
```

Expected: `git diff --check` is silent and every focused test passes.

- [ ] **Step 3: Run the full project test suite**

```bash
../.venv/bin/python -m pytest tests -q
```

Expected: the entire suite passes with no new failures. If an environment-only test is skipped, record its exact name and skip reason in the execution notes.

- [ ] **Step 4: Review the branch diff against the approved spec**

```bash
git status --short
git diff --stat cc7f5b0..HEAD
git log --oneline -5
```

Check that production changes are limited to the files named in this plan, the global thresholds are exact, and no per-listener TTS or alternate stream was added.

- [ ] **Step 5: Commit documentation corrections, if any**

```bash
git add README.md docs/API.md docs/DEPLOYMENT.md
git commit -m "docs: explain shared adaptive TTS speed"
```

If Step 1 documentation was already included in a prior commit and the working tree is clean, do not create an empty commit.

- [ ] **Step 6: Restart the managed service**

```bash
systemctl --user restart voxbridge-8024.service
systemctl --user is-active voxbridge-8024.service
ss -lntp | rg ':8024'
```

Expected: service state is `active` and exactly one managed VoxBridge process owns port `8024`.

- [ ] **Step 7: Verify the live status and page contract**

```bash
curl -fsS http://127.0.0.1:8024/api/tts/live/status
curl -fsS http://127.0.0.1:8024/listen | rg \
  'globalSpeedStatus|maxLiveSyncPlaybackRate|forceNormalPlaybackRate'
```

Expected while idle: empty `speech_epoch_id`, zero translated backlog, multiplier `1.0`, and effective speed `1.05`. The listener HTML contains the fixed-rate guard and no selectable `#playbackRate` control.

- [ ] **Step 8: Run a two-listener live smoke check**

Open one Windows Chrome listener and one iPhone listener against the same service, then verify:

1. Both show the same `Auto - N.Nx` value.
2. Both receive the same sentence sequence from one epoch.
3. The browser media element stays at rate `1.0` on the instrumented Windows check.
4. Removing the original listener does not reset the other listener's epoch or displayed speed.
5. A local iPhone `waiting` event does not change the shared server multiplier.

- [ ] **Step 9: Inspect runtime logs for epoch and speed evidence**

```bash
journalctl --user -u voxbridge-8024.service --since '-15 min' --no-pager | \
  rg 'epoch=|multiplier=|effective_speed=|speed_cache_hit=|last_error|Traceback'
```

Expected: one epoch for both listeners, one synthesis per published sentence, selected speed evidence at sentence boundaries, and no traceback or repeated TTS failure.

- [ ] **Step 10: Confirm a clean final repository state**

```bash
git status --short
git log -1 --oneline
```

Expected: clean working tree and the final commit describes the completed global adaptive TTS change.
