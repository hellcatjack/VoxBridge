# TTS Translation Revision Stability Gate Implementation Plan

> **For the current-session executor:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task. Do not use subagents; the user has
> explicitly prohibited them.

**Goal:** Delay translated-speech publication until the latest source sentence revision
has remained unchanged for a backend-controlled 3.0-second quiet window, while keeping
text subtitles and translation generation responsive.

**Architecture:** Replace `OrderedTTSBuffer` with one revision-aware ordered stability
buffer, then drive it with one bounded scheduler task per ASR WebSocket session. Source
updates and translation callbacks mutate the buffer under the existing TTS transition
lock; the scheduler wakes at the next monotonic deadline and publishes only current,
mature revisions. Normal finalization force-releases final reconciled ready translations,
while abrupt disconnect cancels pending speech.

**Tech Stack:** Python 3.12, asyncio, FastAPI/Starlette WebSockets, pytest, FastAPI
TestClient, systemd user services, Qwen3-ASR through vLLM, Kokoro ONNX CPU TTS.

---

## File Map

- Modify `voxbridge/tts/jobs.py`: replace `OrderedTTSBuffer` with the single
  `RevisionStableTTSBuffer` state machine and deterministic transition metadata.
- Modify `voxbridge/cli/demo_streaming_ws.py`: CLI option, session scheduler, revision
  resets, final force-drain, cancellation, and structured trace events.
- Modify `tests/test_tts_jobs.py`: fake-clock unit coverage for quiet windows, revisions,
  ordering, failures, force-drain, reset, and late revisions.
- Modify `tests/test_demo_streaming_ws_utils.py`: transition-helper and parser tests.
- Modify `tests/test_demo_streaming_ws_protocol.py`: scheduler, revision replacement,
  finalization, disconnect, ordering, and trace-privacy tests.
- Modify `tests/test_release_docs.py`, `README.md`, `CHANGELOG.md`, `docs/API.md`, and
  `docs/DEPLOYMENT.md`: public behavior and operations.
- Modify `/home/hellcat/.config/systemd/user/voxbridge-8024.service` only during
  deployment: append `--tts-revision-stable-sec 3.0` without changing credentials,
  model, memory, VAD, translation endpoint, `.venv`, or port 8024.

## Invariants

- One ASR WebSocket owns at most one TTS stability scheduler task.
- One sentence ID keeps one immutable source order and produces at most one spoken job.
- A newer source revision clears the older ready translation and restarts the quiet
  window from the revision time; translation completion does not restart it.
- Strict source order remains authoritative.
- No incoming audio or model decode is required for deadline release.
- Only orderly finalization may bypass the remaining quiet-window duration.
- Abrupt disconnect never force-releases unreconciled speech.
- No frontend, `/listen`, `/ws/tts`, or broadcast-route behavior changes.
- New trace events contain fingerprints and numeric metadata, never text or raw IDs.

## Task 1: Revision-Stable Ordered Buffer

**Files:**
- Modify: `voxbridge/tts/jobs.py:31-147`
- Modify: `tests/test_tts_jobs.py:1-150`

- [ ] **Step 1: Replace old buffer tests with fake-clock quiet-window tests**

Keep registry tests unchanged. Import `RevisionStableTTSBuffer`, add
`FakeClock.advance(seconds)`, and add these tests:

```python
def test_stability_buffer_withholds_ready_revision_until_quiet_window():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    result = buffer.register("s1", revision=1, source_order=0)
    assert result.accepted is True
    assert buffer.mark_ready("s1", 1, "first", "English") is True
    assert buffer.drain() == []
    assert buffer.next_deadline() == pytest.approx(103.0)
    clock.advance(2.999)
    assert buffer.drain() == []
    clock.advance(0.001)
    ready = buffer.drain()
    assert [(item.sentence_id, item.revision, item.text) for item in ready] == [
        ("s1", 1, "first")
    ]
    assert ready[0].release_reason == "quiet_window"


def test_revision_update_discards_old_translation_and_restarts_window():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "old", "English")
    clock.advance(2.9)
    update = buffer.register("s1", 2, 0)
    assert update.reset is True
    assert update.previous_revision == 1
    assert update.previous_ready is True
    assert update.previous_quiet_age_ms == 2900
    assert buffer.mark_ready("s1", 1, "stale", "English") is False
    assert buffer.mark_ready("s1", 2, "new", "English") is True
    clock.advance(2.9)
    assert buffer.drain() == []
    clock.advance(0.1)
    assert [item.text for item in buffer.drain()] == ["new"]


def test_translation_finishing_after_source_deadline_releases_immediately():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    clock.advance(4.0)
    buffer.mark_ready("s1", 1, "late translation", "English")
    assert [item.text for item in buffer.drain()] == ["late translation"]


def test_stability_buffer_preserves_order_and_skips_failed_head():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    buffer.mark_ready("s2", 1, "second", "English")
    clock.advance(3.0)
    assert buffer.drain() == []
    assert buffer.mark_failed("s1", 1) is True
    assert [item.sentence_id for item in buffer.drain()] == ["s2"]


def test_force_drain_releases_only_current_ready_revisions():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=60.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    buffer.register("s2", 2, 1)
    buffer.mark_ready("s1", 1, "first", "English")
    assert buffer.mark_ready("s2", 1, "stale", "English") is False
    buffer.mark_ready("s2", 2, "second", "English")
    ready = buffer.drain(force=True)
    assert [(item.revision, item.text) for item in ready] == [(1, "first"), (2, "second")]
    assert {item.release_reason for item in ready} == {"final_force"}


def test_revision_after_release_is_reported_and_never_emitted_twice():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=0.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "spoken", "English")
    assert len(buffer.drain()) == 1
    clock.advance(1.25)
    late = buffer.register("s1", 2, 0)
    assert late.late_after_release is True
    assert late.released_revision == 1
    assert late.elapsed_since_release_ms == 1250
    assert buffer.mark_ready("s1", 2, "changed", "English") is False
    assert buffer.drain() == []


def test_stability_buffer_reset_discards_all_session_state():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "old", "English")
    buffer.reset()
    buffer.register("s2", 1, 0)
    buffer.mark_ready("s2", 1, "new", "English")
    clock.advance(3.0)
    assert [item.sentence_id for item in buffer.drain()] == ["s2"]
```

- [ ] **Step 2: Run focused tests to verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_jobs.py
```

Expected: collection fails because `RevisionStableTTSBuffer` does not exist.

- [ ] **Step 3: Add transition metadata and the new entry type**

Retain `TTSReadyItem` compatibility fields and append release metadata with defaults:

```python
@dataclass(frozen=True, slots=True)
class TTSReadyItem:
    sentence_id: str
    revision: int
    source_order: int
    target_language: str
    text: str
    release_reason: str = "quiet_window"
    source_quiet_age_ms: int = 0
    translation_ready_age_ms: int = 0


@dataclass(frozen=True, slots=True)
class TTSRevisionRegistration:
    accepted: bool
    reset: bool
    late_after_release: bool
    sentence_id: str
    revision: int
    source_order: int
    previous_revision: int | None = None
    previous_quiet_age_ms: int = 0
    previous_ready: bool = False
    released_revision: int | None = None
    elapsed_since_release_ms: int = 0


@dataclass(frozen=True, slots=True)
class TTSWaitState:
    sentence_id: str
    revision: int
    source_order: int
    quiet_age_ms: int
    required_quiet_ms: int
    remaining_ms: int
    blocked_by_earlier: bool


@dataclass(slots=True)
class _RevisionStableEntry:
    sentence_id: str
    revision: int
    source_order: int
    changed_at: float
    status: str = "waiting"
    target_language: str | None = None
    text: str | None = None
    translation_ready_at: float | None = None
```

- [ ] **Step 4: Implement the single authoritative buffer**

Replace `OrderedTTSBuffer` with this implementation:

```python
class RevisionStableTTSBuffer:
    def __init__(
        self,
        *,
        stable_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stable_sec < 0:
            raise ValueError("stable_sec must not be negative")
        self._stable_sec = float(stable_sec)
        self._clock = clock
        self._entries: dict[int, _RevisionStableEntry] = {}
        self._sentence_orders: dict[str, int] = {}
        self._order_sentences: dict[int, str] = {}
        self._released: dict[str, tuple[int, int, float]] = {}
        self._next_order = 0
        self._lock = threading.RLock()

    @staticmethod
    def _require_sentence_id(sentence_id: str) -> str:
        sid = str(sentence_id or "").strip()
        if not sid:
            raise ValueError("sentence_id must be a non-empty string")
        return sid

    @staticmethod
    def _elapsed_ms(start: float, end: float) -> int:
        return int(round(max(0.0, float(end) - float(start)) * 1000.0))

    def register(
        self,
        sentence_id: str,
        revision: int,
        source_order: int,
    ) -> TTSRevisionRegistration:
        sid = self._require_sentence_id(sentence_id)
        if revision < 0 or source_order < 0:
            raise ValueError("revision and source_order must not be negative")
        now = self._clock()
        with self._lock:
            known_order = self._sentence_orders.get(sid)
            if known_order is not None and known_order != source_order:
                raise ValueError("sentence_id cannot change source_order")
            known_sentence = self._order_sentences.get(source_order)
            if known_sentence is not None and known_sentence != sid:
                raise ValueError("source_order is already registered")

            released = self._released.get(sid)
            if released is not None:
                released_revision, released_order, released_at = released
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=revision > released_revision,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(released_order),
                    released_revision=int(released_revision),
                    elapsed_since_release_ms=self._elapsed_ms(released_at, now),
                )

            if source_order < self._next_order:
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                )

            current = self._entries.get(source_order)
            if current is not None and revision <= current.revision:
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                    previous_revision=int(current.revision),
                )

            self._sentence_orders[sid] = int(source_order)
            self._order_sentences[int(source_order)] = sid
            if current is None:
                self._entries[source_order] = _RevisionStableEntry(
                    sid, int(revision), int(source_order), float(now)
                )
                return TTSRevisionRegistration(
                    accepted=True,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                )

            previous_revision = int(current.revision)
            previous_ready = current.status == "ready"
            previous_quiet_age_ms = self._elapsed_ms(current.changed_at, now)
            self._entries[source_order] = _RevisionStableEntry(
                sid, int(revision), int(source_order), float(now)
            )
            return TTSRevisionRegistration(
                accepted=True,
                reset=True,
                late_after_release=False,
                sentence_id=sid,
                revision=int(revision),
                source_order=int(source_order),
                previous_revision=previous_revision,
                previous_quiet_age_ms=previous_quiet_age_ms,
                previous_ready=previous_ready,
            )

    def _current_entry(
        self,
        sentence_id: str,
        revision: int,
    ) -> _RevisionStableEntry | None:
        order = self._sentence_orders.get(sentence_id)
        if order is None:
            return None
        entry = self._entries.get(order)
        if entry is None or entry.revision != revision:
            return None
        return entry

    def mark_ready(
        self,
        sentence_id: str,
        revision: int,
        text: str,
        target_language: str,
    ) -> bool:
        sid = self._require_sentence_id(sentence_id)
        translated = str(text or "").strip()
        language = str(target_language or "").strip()
        if translated and not language:
            raise ValueError("target_language must be a non-empty string")
        now = self._clock()
        with self._lock:
            entry = self._current_entry(sid, int(revision))
            if entry is None:
                return False
            if not translated:
                entry.status = "failed"
                entry.text = None
                entry.target_language = None
                entry.translation_ready_at = None
                return True
            entry.status = "ready"
            entry.text = translated
            entry.target_language = language
            entry.translation_ready_at = float(now)
            return True

    def mark_failed(self, sentence_id: str, revision: int) -> bool:
        sid = self._require_sentence_id(sentence_id)
        with self._lock:
            entry = self._current_entry(sid, int(revision))
            if entry is None:
                return False
            entry.status = "failed"
            entry.text = None
            entry.target_language = None
            entry.translation_ready_at = None
            return True

    def wait_state(self, sentence_id: str) -> TTSWaitState | None:
        sid = self._require_sentence_id(sentence_id)
        now = self._clock()
        with self._lock:
            order = self._sentence_orders.get(sid)
            entry = self._entries.get(order) if order is not None else None
            if entry is None or entry.status != "ready":
                return None
            quiet_age_ms = self._elapsed_ms(entry.changed_at, now)
            required_ms = int(round(self._stable_sec * 1000.0))
            return TTSWaitState(
                sentence_id=sid,
                revision=int(entry.revision),
                source_order=int(entry.source_order),
                quiet_age_ms=quiet_age_ms,
                required_quiet_ms=required_ms,
                remaining_ms=max(0, required_ms - quiet_age_ms),
                blocked_by_earlier=entry.source_order != self._next_order,
            )

    def next_deadline(self) -> float | None:
        with self._lock:
            entry = self._entries.get(self._next_order)
            if entry is None or entry.status != "ready":
                return None
            return float(entry.changed_at + self._stable_sec)

    def drain(self, *, force: bool = False) -> list[TTSReadyItem]:
        now = self._clock()
        ready: list[TTSReadyItem] = []
        with self._lock:
            while True:
                entry = self._entries.get(self._next_order)
                if entry is None or entry.status == "waiting":
                    break
                if entry.status == "failed":
                    del self._entries[self._next_order]
                    self._next_order += 1
                    continue
                if not force and now < entry.changed_at + self._stable_sec:
                    break
                del self._entries[self._next_order]
                self._next_order += 1
                self._released[entry.sentence_id] = (
                    int(entry.revision),
                    int(entry.source_order),
                    float(now),
                )
                ready.append(
                    TTSReadyItem(
                        sentence_id=entry.sentence_id,
                        revision=int(entry.revision),
                        source_order=int(entry.source_order),
                        target_language=str(entry.target_language or ""),
                        text=str(entry.text or ""),
                        release_reason="final_force" if force else "quiet_window",
                        source_quiet_age_ms=self._elapsed_ms(entry.changed_at, now),
                        translation_ready_age_ms=self._elapsed_ms(
                            entry.translation_ready_at or now,
                            now,
                        ),
                    )
                )
        return ready

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sentence_orders.clear()
            self._order_sentences.clear()
            self._released.clear()
            self._next_order = 0
```

Implement these exact rules:

1. Reject negative `stable_sec`, revision, and order; validate non-empty IDs/languages.
2. Each public operation that computes timing samples the injected clock once.
3. Treat same/older revisions as idempotent no-ops; reject identity/order changes.
4. A higher revision resets status/text/language/ready time and reports prior timing.
5. Ready/failed transitions mutate only the exact current revision.
6. `next_deadline` returns a deadline only for the current ready head.
7. `drain` consumes failed heads immediately and releases consecutive mature heads.
8. `force=True` bypasses only quiet age and labels items `final_force`.
9. Retain released revision/order/time to reject and report late revisions.
10. `reset` clears pending, released, order, and identity state.

- [ ] **Step 5: Verify GREEN and broadcast compatibility**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_tts_jobs.py tests/test_tts_broadcast.py
```

Expected: both modules pass.

- [ ] **Step 6: Commit**

```bash
git add voxbridge/tts/jobs.py tests/test_tts_jobs.py
git commit -m "feat: gate TTS on stable source revisions"
```

## Task 2: CLI Control and Independent Scheduler

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py:63,4435-4444,5000-5275,5740-5985,9920-10030`
- Modify: `tests/test_demo_streaming_ws_utils.py:1-85,790-1025`
- Modify: `tests/test_demo_streaming_ws_protocol.py:90-110,570-765`

- [ ] **Step 1: Write parser and scheduler tests first**

Set `tts_revision_stable_sec=0.0` in protocol `_args()` so existing tests remain fast.
Add parser tests for default `3.0`, override `1.75`, and rejection of `-0.1`. Replace the
old utility-test buffer import with `RevisionStableTTSBuffer(stable_sec=0.0)`.

Add `_drain_listener_events(subscription)` beside protocol helpers and this test:

```python
def test_ws_tts_stability_scheduler_releases_without_more_audio():
    args = _args()
    args.final_redecode_on_stop = False
    args.tts_revision_stable_sec = 0.12
    app = _create_app(args, _StableTTSSentenceASR(), translator=_FakeTranslator(),
                      tts_synthesizer=_FakeTTSSynthesizer())
    listener = app.state.tts_broadcast.register("anonymous")
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        ws.send_bytes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
        _receive_until_type(ws, "sentence_translation")
        assert not [e for e in _drain_listener_events(listener) if e.get("type") == "tts_job"]
        time.sleep(0.16)
        jobs = [e for e in _drain_listener_events(listener) if e.get("type") == "tts_job"]
        assert jobs
```

- [ ] **Step 2: Verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py -k 'tts_revision_stability' \
  tests/test_demo_streaming_ws_protocol.py::test_ws_tts_stability_scheduler_releases_without_more_audio
```

Expected: parser attribute failure and no delayed publication.

- [ ] **Step 3: Add the non-negative CLI option**

Add:

```python
def _non_negative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


p.add_argument(
    "--tts-revision-stable-sec",
    type=_non_negative_float_arg,
    default=3.0,
    help="Require this long without a source revision before publishing translated speech",
)
```

- [ ] **Step 4: Construct the new buffer and bounded scheduler state**

Import `RevisionStableTTSBuffer`, read the option with fallback `3.0`, and replace the
runtime buffer. Add only these scheduler fields:

```python
ordered=RevisionStableTTSBuffer(stable_sec=tts_revision_stable_sec),
stability_task=None,
stability_wake=asyncio.Event(),
stability_stopping=False,
last_wait_key=None,
```

- [ ] **Step 5: Implement one deadline-driven scheduler**

Add:

```python
def _wake_tts_stability_scheduler() -> None:
    if not bool(tts_runtime.stability_stopping):
        tts_runtime.stability_wake.set()


async def _drain_tts_stability(*, force: bool = False) -> None:
    async with tts_transition_lock:
        ready = tts_runtime.ordered.drain(force=force)
        if ready:
            await _publish_tts_ready(ready)


async def _tts_stability_scheduler() -> None:
    try:
        while not bool(tts_runtime.stability_stopping):
            tts_runtime.stability_wake.clear()
            await _drain_tts_stability()
            async with tts_transition_lock:
                deadline = tts_runtime.ordered.next_deadline()
            if deadline is None:
                await tts_runtime.stability_wake.wait()
                continue
            timeout = max(0.0, float(deadline) - time.monotonic())
            try:
                await asyncio.wait_for(tts_runtime.stability_wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        raise
```

Create exactly one task after nested TTS helpers are defined. Clear the event before the
locked state read to avoid lost wakeups.

- [ ] **Step 6: Route ready/failed callbacks through the gate**

In each transition closure, verify generation, call `mark_ready` or `mark_failed`, wake
the scheduler for accepted transitions, then return `ordered.drain()`. Keep
`_run_ordered_tts_transition` so transition and whole-batch publication remain under the
same lock.

- [ ] **Step 7: Verify scheduler and existing ordering behavior**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py \
  tests/test_demo_streaming_ws_protocol.py -k 'tts and (scheduler or source_order or failed_earlier)'
```

Expected: parser, independent release, strict order, and failed-head tests pass.

- [ ] **Step 8: Commit**

```bash
git add voxbridge/cli/demo_streaming_ws.py \
  tests/test_demo_streaming_ws_utils.py tests/test_demo_streaming_ws_protocol.py
git commit -m "feat: schedule stable TTS publication independently"
```

## Task 3: Revision Reset, Final Force-Drain, and Cancellation

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py:5740-6025,8060-8350,9020-9258,9440-9700`
- Modify: `tests/test_demo_streaming_ws_protocol.py:760-1045,3110-3255`

- [ ] **Step 1: Write lifecycle tests first**

Add a fast translator revision test derived from the existing superseded-revision test.
Do not block revision 1 translation. Set the window to `0.20`, trigger revision 2 inside
the window, wait `0.24` without more audio, and assert the listener receives exactly one
matching job at revision 2.

Add an orderly final test with `tts_revision_stable_sec=60.0`: produce a translation,
assert no job before finish, send `finish`, collect through `final`, and assert ready jobs
were published before producer inactive.

Add an abrupt-disconnect test: produce a ready translation with a `0.20` window, exit
without `finish`, wait `0.24`, and assert no `tts_job` appears.

Add `test_ws_translation_direction_change_discards_pending_tts`: create a ready
translation with a `60.0` window, send `set_translation_direction` with the opposite
direction before release, wait briefly, and assert no old-direction job is published.
Then send fresh audio and finish; assert only jobs using the new canonical target
language are published.

- [ ] **Step 2: Verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_protocol.py -k \
  'fast_translation_waits_for_latest_revision or finish_force_releases_latest or abrupt_disconnect_discards_pending_tts or direction_change_discards_pending_tts'
```

Expected: early revision, missing force-release, or post-disconnect publication failure.

- [ ] **Step 3: Make registration and reset lock-safe**

Convert `_register_tts_source` to async, preserve source-order allocation, call
`ordered.register` under `tts_transition_lock`, wake for accepted registrations, and
await it at both sentence commit/update call sites.

Convert `_reset_tts_ordering` to async. Under the same lock increment generation, reset
source order/identity/buffer/wait key, and wake the scheduler. Await it from TTS config,
direction change, start, canonical final re-decode, and final reconciliation. Never call
it while already holding the transition lock.

- [ ] **Step 4: Add explicit scheduler shutdown**

```python
async def _stop_tts_stability_scheduler(*, reason: str) -> None:
    if bool(tts_runtime.stability_stopping):
        return
    tts_runtime.stability_stopping = True
    tts_runtime.stability_wake.set()
    task = tts_runtime.stability_task
    if task is not None and not task.done():
        task.cancel()
    if task is not None:
        with suppress(asyncio.CancelledError):
            await task
    tts_runtime.stability_task = None
```

Require `not stability_stopping` in `_tts_output_active()`. In outer `finally`, stop the
scheduler before cancelling translations, preventing deadline publication after abrupt
disconnect.

- [ ] **Step 5: Force-release after final reconciliation and translation drain**

Immediately after `_drain_tts_translation_task("before_final")`, call
`await _drain_tts_stability(force=True)` before final payload and producer-inactive
publication. This bypasses only quiet age; stale/unready entries remain rejected.

Import `inspect` in the lifecycle test module and add this source-level assertion so
VAD, hard cut, and ordinary commits cannot acquire a second force-release path:

```python
def test_tts_quiet_window_is_bypassed_only_by_orderly_finalization():
    source = inspect.getsource(demo_streaming_ws._create_app)
    assert source.count("_drain_tts_stability(force=True)") == 1
    assert '_drain_tts_stability(force=True)' not in source[source.index("async def _maybe_vad_silence_cut"):source.index("async def _audio_consumer")]
```

- [ ] **Step 6: Verify lifecycle and all TTS protocol tests**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_protocol.py -k 'tts or superseded_sentence_revision'
```

Expected: only revision 2 is spoken, final force-drains, abrupt disconnect discards, and
existing TTS behavior passes.

- [ ] **Step 7: Commit**

```bash
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_protocol.py
git commit -m "fix: speak only final stable sentence revisions"
```

## Task 4: Structured Diagnostics and Failure Isolation

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py:5260-5330,5820-5990`
- Modify: `tests/test_demo_streaming_ws_protocol.py:3110-3255`

- [ ] **Step 1: Write trace-contract tests first**

Enable `subtitle_trace_log` in the fast-revision test and assert:

```python
event_names = {row.get("event") for row in trace_rows}
assert "tts_stability_wait" in event_names
assert "tts_stability_reset" in event_names
assert "tts_stability_release" in event_names

private_events = {
    "tts_stability_wait",
    "tts_stability_reset",
    "tts_stability_release",
    "tts_late_revision_after_release",
}
for row in trace_rows:
    if row.get("event") in private_events:
        assert "text" not in row
        assert "translation" not in row
        assert "sentence_id" not in row
        assert "job_id" not in row
        assert len(str(row.get("sentence_hash8", ""))) == 8
```

Add a zero-window case that emits revision 1 and later receives revision 2. Assert one
`tts_late_revision_after_release` event with released/incoming revision and elapsed
milliseconds, while the listener still receives only one job.

Add scheduler-failure isolation using a one-shot monkeypatch:

```python
def test_ws_tts_scheduler_failure_does_not_stop_asr(monkeypatch):
    original = RevisionStableTTSBuffer.next_deadline
    failed = False

    def fail_once(self):
        nonlocal failed
        if self.pending_count and not failed:
            failed = True
            raise RuntimeError("synthetic scheduler failure")
        return original(self)

    monkeypatch.setattr(RevisionStableTTSBuffer, "next_deadline", fail_once)
    args = _args()
    args.tts_revision_stable_sec = 1.0
    app = _create_app(args, _StableTTSSentenceASR(), translator=_FakeTranslator(),
                      tts_synthesizer=_FakeTTSSynthesizer())
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        _receive_until_type(ws, "started")
        frame = np.array([0, 1000, -1000], dtype="<i2").tobytes()
        ws.send_bytes(frame)
        unavailable = _receive_until_type(ws, "tts_status")
        assert unavailable["status"] == "unavailable"
        ws.send_bytes(frame)
        assert _receive_until_type(ws, "partial")["type"] == "partial"
```

- [ ] **Step 2: Verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_protocol.py -k \
  'tts_stability_trace_is_redacted or late_revision_after_tts_release_is_traced or tts_scheduler_failure_does_not_stop_asr'
```

Expected: required events are absent.

- [ ] **Step 3: Emit deduplicated wait and revision diagnostics**

After accepted `mark_ready`, inspect `wait_state(sentence_id)`. Emit
`tts_stability_wait` only when `remaining_ms > 0` or `blocked_by_earlier` is true, and
only once per `(sentence_id, revision, blocked_by_earlier)` key:

```python
_trace_event(
    "tts_stability_wait",
    sentence_hash8=_opaque_identifier_hash8(sentence_id),
    revision=int(wait.revision),
    source_order=int(wait.source_order),
    quiet_age_ms=int(wait.quiet_age_ms),
    required_quiet_ms=int(wait.required_quiet_ms),
    remaining_ms=int(wait.remaining_ms),
    blocked_by_earlier=bool(wait.blocked_by_earlier),
)
```

Use `TTSRevisionRegistration` to emit `tts_stability_reset` for accepted higher
revisions and `tts_late_revision_after_release` for rejected post-release revisions.
Use only `sentence_hash8` and numeric metadata specified by the design.

- [ ] **Step 4: Emit release timing before publication**

At the beginning of each `_publish_tts_ready` iteration emit:

```python
_trace_event(
    "tts_stability_release",
    sentence_hash8=_opaque_identifier_hash8(str(item.sentence_id)),
    revision=int(item.revision),
    source_order=int(item.source_order),
    release_reason=str(item.release_reason),
    source_quiet_age_ms=int(item.source_quiet_age_ms),
    translation_ready_age_ms=int(item.translation_ready_age_ms),
    ordered_backlog_depth=int(tts_runtime.ordered.pending_count),
)
```

Clear `last_wait_key` when a revision resets, an item releases, or ordering resets.

- [ ] **Step 5: Isolate unexpected scheduler failures**

Wrap the scheduler loop with a non-cancellation exception handler. Set
`stability_stopping=True`, emit `tts_stability_scheduler_failed` with exception type and
pending count, log a redacted warning, and send `tts_status: unavailable`. Do not close
the ASR WebSocket or stop text translation. Emit scheduler start/cancel events with
generation and pending count; do not repeat unchanged wait events.

- [ ] **Step 6: Verify trace privacy and text-pool compatibility**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_protocol.py -k 'tts_stability or late_revision' \
  tests/test_text_pool_trace_report.py
```

Expected: stability events are present and redacted; text-pool tests pass.

- [ ] **Step 7: Commit**

```bash
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_protocol.py
git commit -m "feat: trace TTS revision stability decisions"
```

## Task 5: Public Documentation and Deployment Contract

**Files:**
- Modify: `tests/test_release_docs.py`
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write release-document assertions first**

```python
def test_docs_publish_tts_revision_stability_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "--tts-revision-stable-sec" in api
    assert "--tts-revision-stable-sec 3.0" in deployment
    assert "source revision" in api.lower()
    assert "字幕" in readme and "朗读" in readme
```

- [ ] **Step 2: Verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_release_docs.py
```

Expected: the option and behavior are undocumented.

- [ ] **Step 3: Document API timing and finalization**

In `docs/API.md`, state explicitly:

- Sentence source/translation events remain immediate.
- TTS waits from the latest source revision, not translation completion.
- Default is 3.0 seconds; `0` disables only the delay.
- A revision inside the window invalidates old speech text.
- Normal finish force-releases only final ready revisions after translation drain.
- Abrupt disconnect discards pending speech.
- New stability traces contain no text.

- [ ] **Step 4: Document operations and release behavior**

Append `--tts-revision-stable-sec 3.0` to the TTS block in `docs/DEPLOYMENT.md` and
explain tuning from `tts_late_revision_after_release`, without punctuation or
language-specific heuristics. Preserve port 8024 and `.venv` commands.

Update `README.md` to distinguish responsive visible subtitles from deliberately
delayed spoken translations. Add an Unreleased `Fixed` entry in `CHANGELOG.md` for
preventing speech of superseded revisions.

- [ ] **Step 5: Verify docs and parser tests**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_release_docs.py tests/test_demo_streaming_ws_utils.py
```

Expected: both modules pass.

- [ ] **Step 6: Commit**

```bash
git add README.md CHANGELOG.md docs/API.md docs/DEPLOYMENT.md tests/test_release_docs.py
git commit -m "docs: explain stable translated speech timing"
```

## Task 6: Full Verification, Integration, and Port 8024 Deployment

**Files:**
- Verify all tracked files from Tasks 1-5.
- Modify during deployment:
  `/home/hellcat/.config/systemd/user/voxbridge-8024.service`

- [ ] **Step 1: Run full regression verification**

```bash
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
git diff --check
git status --short
```

Expected: compile exits 0, all tests pass, diff check is empty, and no unintended files
are present.

- [ ] **Step 2: Audit the implementation against invariants**

```bash
rg -n 'OrderedTTSBuffer|RevisionStableTTSBuffer|tts-revision-stable-sec|tts_stability_' \
  voxbridge tests README.md CHANGELOG.md docs
rg -n 'sentence_id=|job_id=|text=|translation=' voxbridge/cli/demo_streaming_ws.py \
  | rg 'tts_stability' || true
git log --oneline --decorate -8
```

Expected: no runtime `OrderedTTSBuffer`; one stability state machine; no raw text/ID in
new diagnostics; scoped commits.

- [ ] **Step 3: Integrate into the service working tree**

Invoke `superpowers:finishing-a-development-branch`. Because the service runs from
`/data/Qwen3-ASR/VoxBridge`, use a non-interactive fast-forward only if `main` has not
moved. If unexpected changes exist, stop and ask instead of overwriting or rebasing.

After integration run:

```bash
git -C /data/Qwen3-ASR/VoxBridge status --short --branch
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  /data/Qwen3-ASR/VoxBridge/tests/test_tts_jobs.py \
  /data/Qwen3-ASR/VoxBridge/tests/test_demo_streaming_ws_protocol.py
```

Expected: main is clean and focused tests pass from the service tree.

- [ ] **Step 4: Append only the production quiet-window flag**

```bash
unit=/home/hellcat/.config/systemd/user/voxbridge-8024.service
test "$(rg -o -- '--tts-final-translation-drain-sec 30' "$unit" | wc -l)" -eq 1
test "$(rg -o -- '--tts-revision-stable-sec' "$unit" | wc -l)" -eq 0
sed -i \
  's/--tts-final-translation-drain-sec 30/--tts-final-translation-drain-sec 30 --tts-revision-stable-sec 3.0/' \
  "$unit"
systemd-analyze --user verify "$unit"
```

Expected: verify exits 0. Do not print or alter the authentication hash. Do not change
Python, model, memory, VAD, translation API, host, or port.

- [ ] **Step 5: Restart one managed service and verify topology**

```bash
systemctl --user daemon-reload
systemctl --user restart voxbridge-8024.service
systemctl --user is-active voxbridge-8024.service
systemctl --user show voxbridge-8024.service \
  -p MainPID -p NRestarts -p MemoryCurrent --no-pager
ss -lntp | rg ':8024'
ps -eo pid,ppid,cmd | rg '[v]oxbridge\.cli\.demo_streaming_ws|VLLM::EngineCore'
```

Expected: active service, port 8024, one VoxBridge process, and one EngineCore. If extra
processes appear, stop and diagnose rather than starting another backend.

- [ ] **Step 6: Verify HTTPS and startup logs**

```bash
curl -k -sS -o /dev/null -w '%{http_code}\n' \
  https://ushome.amycat.com:18024/
journalctl --user -u voxbridge-8024.service --since '-5 min' --no-pager \
  | rg 'Application startup complete|ws open|EngineCore|Traceback|ERROR|orphan' || true
```

Expected: the existing authenticated/redirect HTTPS response, clean startup, no restart
loop, traceback, or orphan warning.

- [ ] **Step 7: Perform the end-to-end smoke test**

Use the authenticated main page and `/listen` on another browser/device. Speak or play
one sentence that receives a streaming revision, then stop. Confirm visible text updates
before speech, only the final revision is spoken once, silence requires no extra audio
to trigger speech, and Stop publishes final ready speech before producer inactive.

Inspect redacted metadata only:

```bash
tail -n 500 /data/Qwen3-ASR/logs/voxbridge_subtitle_trace.jsonl \
  | rg 'tts_stability_(wait|reset|release)|tts_late_revision_after_release'
```

Expected: wait then release, reset when a revision changes, no text fields, and no late
revision in the smoke sample.

- [ ] **Step 8: Record evidence**

Report exact test count, service MainPID, EngineCore count, port 8024 listener, configured
3.0-second value, and observed event sequence. Do not claim completion if evidence is
missing.

## Rollback

First set the installed value to `0` and restart the same service. This disables only
the delay while retaining order/revision checks:

```bash
unit=/home/hellcat/.config/systemd/user/voxbridge-8024.service
sed -i 's/--tts-revision-stable-sec 3\.0/--tts-revision-stable-sec 0/' "$unit"
systemctl --user daemon-reload
systemctl --user restart voxbridge-8024.service
ss -lntp | rg ':8024'
```

If code rollback is required, use normal non-interactive reverts of scoped commits.
Never use `git reset --hard`, alter unrelated user changes, or run a second backend.
