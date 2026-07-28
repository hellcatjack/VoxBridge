# Multi-Listener TTS Broadcast Implementation Plan

> **For the current-session executor:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task. Do not use subagents; the user has
> explicitly prohibited them.

**Goal:** Move translated-speech playback to an authenticated `/listen` page where
multiple devices independently receive future stable translations and play them in
strict FIFO order.

**Architecture:** Add a process-local broadcast hub that snapshots online listeners,
stores one shared TTS job per stable translation, and releases it after every intended
listener acknowledges receipt or disconnects. Keep the existing ordered translation
producer, synthesize one cached WAV under the global CPU lock, and retain legacy
owner-scoped TTS APIs only for clients that explicitly request them.

**Tech Stack:** Python 3.12, FastAPI/Starlette WebSockets, asyncio, Kokoro ONNX CPU TTS,
vanilla HTML/CSS/JavaScript, pytest, FastAPI TestClient.

---

## File Map

- Create `voxbridge/tts/broadcast.py`: listeners, jobs, audio leases, acknowledgement,
  expiry, and overflow isolation.
- Create `voxbridge/tts/listener_page.py`: standalone listener HTML/CSS/JavaScript.
- Modify `voxbridge/tts/__init__.py`: export broadcast types.
- Modify `voxbridge/cli/demo_streaming_ws.py`: hub assembly, routes, stable translation
  publication, main-page decoupling, authentication return path, and CLI bounds.
- Create `tests/test_tts_broadcast.py`: deterministic hub tests.
- Modify `tests/test_demo_streaming_ws_protocol.py`: routes, authentication, producer,
  shared synthesis, and stop behavior.
- Modify `tests/test_demo_streaming_ws_utils.py`: parser and frontend contracts.
- Modify `tests/test_release_docs.py`, `docs/API.md`, `docs/DEPLOYMENT.md`, `README.md`,
  and `CHANGELOG.md`: public behavior and operations.

## Task 1: Broadcast Hub Core

**Files:**
- Create: `voxbridge/tts/broadcast.py`
- Create: `tests/test_tts_broadcast.py`
- Modify: `voxbridge/tts/__init__.py`

- [ ] **Step 1: Write publication tests first**

Create tests using an injected clock and token factory:

```python
def test_publish_fans_one_job_to_current_listeners_only():
    hub = create_hub(tokens=iter(["listener-a", "listener-b", "job-1"]))
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item("Stable translation."))
    assert first.queue.get_nowait()["job_id"] == job.job_id
    assert second.queue.get_nowait()["job_id"] == job.job_id
    late = hub.register("owner-c")
    with pytest.raises(asyncio.QueueEmpty):
        late.queue.get_nowait()


def test_publish_without_listener_retains_nothing():
    hub = create_hub()
    assert hub.publish(ready_item("Nobody is listening.")) is None
    assert hub.job_count == 0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_broadcast.py
```

Expected: import failure because `voxbridge.tts.broadcast` does not exist.

- [ ] **Step 3: Implement registration and publication**

Implement these public interfaces:

```python
@dataclass(frozen=True, slots=True)
class BroadcastTTSJob:
    job_id: str
    sentence_id: str
    revision: int
    source_order: int
    target_language: str
    text: str
    created_at: float
    expires_at: float
    audio_bytes: bytes | None = None
    sample_rate: int | None = None
    duration_ms: int | None = None


@dataclass(slots=True)
class TTSListenerSubscription:
    listener_id: str
    owner_key: str
    queue: asyncio.Queue[dict[str, object]]
    overflowed: asyncio.Event


```

Implement `TTSBroadcastHub.register(owner_key) -> TTSListenerSubscription`,
`unregister(listener_id, owner_key) -> int`, and
`publish(item) -> BroadcastTTSJob | None`. Use internal `_BroadcastJobState` for mutable
pending-listener, in-flight, and cached audio fields. Validate all identifiers/text and
create cryptographically random tokens by default. Publish metadata only, never
translated text.

- [ ] **Step 4: Write lifecycle tests before lifecycle code**

Add:

```python
def test_acknowledgement_is_per_listener_and_last_ack_deletes_job():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item("Stable translation."))
    assert hub.acknowledge(job.job_id, first.listener_id, "owner-a") is True
    assert hub.job_count == 1
    assert hub.acknowledge(job.job_id, second.listener_id, "owner-b") is True
    assert hub.job_count == 0


def test_disconnect_releases_only_that_listener():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item("Stable translation."))
    assert hub.unregister(first.listener_id, "owner-a") == 1
    claimed = hub.claim_audio(job.job_id, second.listener_id, "owner-b")
    assert claimed.job_id == job.job_id
    hub.release_audio(job.job_id)


def test_audio_lease_prevents_disconnect_from_deleting_inflight_job():
    hub = create_hub()
    listener = hub.register("owner-a")
    job = hub.publish(ready_item("Stable translation."))
    hub.claim_audio(job.job_id, listener.listener_id, "owner-a")
    hub.unregister(listener.listener_id, "owner-a")
    assert hub.job_count == 1
    hub.release_audio(job.job_id)
    assert hub.job_count == 0


def test_foreign_owner_cannot_claim_or_acknowledge_job():
    hub = create_hub()
    listener = hub.register("owner-a")
    job = hub.publish(ready_item("Private translation."))
    with pytest.raises(TTSBroadcastNotFound):
        hub.claim_audio(job.job_id, listener.listener_id, "owner-b")
    assert hub.acknowledge(job.job_id, listener.listener_id, "owner-b") is False


def test_expired_job_is_not_exposed():
    clock = FakeClock(100.0)
    hub = create_hub(clock=clock, ttl_sec=30)
    listener = hub.register("owner-a")
    job = hub.publish(ready_item("Expiring translation."))
    clock.value = 130.01
    with pytest.raises(TTSBroadcastNotFound):
        hub.claim_audio(job.job_id, listener.listener_id, "owner-a")
    assert hub.job_count == 0


def test_overflow_disconnects_only_the_slow_listener():
    hub = create_hub(listener_queue_size=1)
    slow = hub.register("owner-a")
    fast = hub.register("owner-b")
    hub.publish(ready_item("First."))
    fast.queue.get_nowait()
    second = hub.publish(ready_item("Second.", source_order=1, sentence_id="s2"))
    assert slow.overflowed.is_set()
    assert fast.queue.get_nowait()["job_id"] == second.job_id
    assert hub.listener_count == 1
```

- [ ] **Step 5: Verify lifecycle RED**

Expected: missing claim, release, cache, acknowledgement, pruning, and overflow behavior.

- [ ] **Step 6: Implement lifecycle and bounds**

Implement `claim_audio(job_id, listener_id, owner_key) -> BroadcastTTSJob`,
`release_audio(job_id) -> None`, `cache_audio(job_id, audio_bytes, sample_rate,
duration_ms) -> BroadcastTTSJob`, `acknowledge(job_id, listener_id, owner_key) -> bool`,
and `prune() -> int`. Protect state with one `threading.RLock`. Delete a job only when
pending listeners and in-flight leases are both zero. On queue overflow unregister only
that listener and signal its WebSocket; never evict an unread event.

- [ ] **Step 7: Run Task 1 tests and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_broadcast.py tests/test_tts_jobs.py
git add voxbridge/tts/broadcast.py voxbridge/tts/__init__.py tests/test_tts_broadcast.py
git commit -m "feat: add bounded TTS broadcast hub"
```

## Task 2: Standalone Listener Page

**Files:**
- Create: `voxbridge/tts/listener_page.py`
- Modify: `tests/test_demo_streaming_ws_utils.py`

- [ ] **Step 1: Write listener-template tests first**

```python
def test_listener_page_requires_explicit_start_and_uses_fifo():
    assert 'id="startListening"' in TTS_LISTENER_HTML
    assert 'id="stopListening"' in TTS_LISTENER_HTML
    assert 'new WebSocket(wsUrl("/ws/tts"))' in TTS_LISTENER_HTML
    assert "queue.push(job)" in TTS_LISTENER_HTML
    assert "currentJob = queue.shift()" in TTS_LISTENER_HTML
    assert 'type: "tts_received"' in TTS_LISTENER_HTML
    assert "X-TTS-Listener-ID" in TTS_LISTENER_HTML
```

- [ ] **Step 2: Verify RED**

Run the focused test and expect import failure for `listener_page.py`.

- [ ] **Step 3: Implement the mobile-first page**

Build a low-glare light interface with Start, Stop, connection/producer/playback status,
and backlog count. Start must create/resume one AudioContext from the user gesture and
open one `/ws/tts` connection. Incoming jobs append to one FIFO. The pump fetches only
the head item, reads the full WAV, acknowledges receipt, decodes, waits for `onended`,
then advances. Stop aborts the fetch, stops the source, clears only the local queue, and
closes the listener socket. Do not automatically reconnect or log translated text.

- [ ] **Step 4: Run tests and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_utils.py
git add voxbridge/tts/listener_page.py tests/test_demo_streaming_ws_utils.py
git commit -m "feat: add standalone TTS listener page"
```

## Task 3: Authenticated Listener and Audio Routes

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py`
- Modify: `tests/test_demo_streaming_ws_protocol.py`

- [ ] **Step 1: Write authentication tests first**

Add tests that unauthenticated `/listen` redirects to `/login?next=%2Flisten`, successful
login returns to `/listen`, external or protocol-relative `next` values return to `/`,
and unauthenticated `/ws/tts` closes with policy violation.

- [ ] **Step 2: Verify RED**

Expected: listener routes are absent and login ignores `next`.

- [ ] **Step 3: Implement safe return and listener WebSocket**

Add `_safe_login_next` accepting only same-origin relative paths beginning with one `/`.
Render an escaped hidden `next` form field. Initialize `app.state.tts_broadcast` and add
authenticated `GET /listen` plus `/ws/tts`. The listener socket registers an auth-owner
hash, sends `tts_listener_ready`, forwards subscription events, handles
`tts_received`/`ping`, and always unregisters on disconnect. Listener sockets must not
count against the ASR `--max-connections` limit.

- [ ] **Step 4: Write shared-audio tests first**

Register two authenticated listener clients, publish one item, and assert both receive
the same job ID, both can fetch using their own listener ID, a foreign session gets
`404`, one acknowledgement preserves the other assignment, and the fake synthesizer is
called once.

- [ ] **Step 5: Implement shared audio endpoint**

Add `POST /api/tts/broadcast/jobs/{job_id}/audio`. Read `X-TTS-Listener-ID`, validate the
auth owner, claim a lease, synthesize/cache once under `tts_synthesis_lock`, return WAV
with `no-store` and metadata headers, and release the lease in `finally`. Collapse all
missing/foreign failures into `404`.

- [ ] **Step 6: Run route tests and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_protocol.py -k 'listener or broadcast'
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_protocol.py
git commit -m "feat: serve authenticated TTS broadcast"
```

## Task 4: Stable Translation Producer Integration

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py`
- Modify: `tests/test_demo_streaming_ws_protocol.py`

- [ ] **Step 1: Write producer tests first**

Add tests proving that stable translations broadcast without main-page `tts_enabled` or
`tts_client_id`, no listener retains no job, out-of-order translation completion still
publishes source order, stale revisions never publish, and inactive producer status is
sent after the final stable job without clearing listener queues.

- [ ] **Step 2: Verify RED**

Expected: no broadcast publication because current source registration is private-TTS
gated.

- [ ] **Step 3: Separate broadcast and legacy state**

Keep `tts_runtime.enabled` as the deprecated private-client switch and add
`broadcast_enabled = available`. Run source registration, ordered transitions, direction
reset, and final drain when either output is active. `_publish_tts_ready` must publish
once through the hub and only create/send an owner-scoped legacy job when the producer
explicitly supplied legacy fields. Publish producer active/inactive state through the
hub. Do not wait for synthesis or listener playback.

- [ ] **Step 4: Verify legacy compatibility and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_protocol.py tests/test_tts_broadcast.py tests/test_tts_jobs.py
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_protocol.py
git commit -m "feat: broadcast stable translations to listeners"
```

Explicit legacy clients must still receive private `tts_job` events and use the old
audio/ack routes for one compatibility cycle.

## Task 5: Remove Main-Page Playback

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py`
- Modify: `tests/test_demo_streaming_ws_utils.py`

- [ ] **Step 1: Replace old frontend tests with decoupling tests**

```python
def test_main_page_links_listener_without_audio_playback():
    assert 'href="/listen"' in INDEX_HTML_TEMPLATE
    assert "打开译文朗读页" in INDEX_HTML_TEMPLATE
    assert 'id="ttsEnabledInput"' not in INDEX_HTML_TEMPLATE
    assert "AudioContext" not in INDEX_HTML_TEMPLATE
    assert "ttsQueue" not in INDEX_HTML_TEMPLATE
    assert "tts_client_id" not in INDEX_HTML_TEMPLATE
    assert "tts_enabled:" not in INDEX_HTML_TEMPLATE
```

- [ ] **Step 2: Verify RED**

Expected: the main page still contains the checkbox and player.

- [ ] **Step 3: Remove playback and add listener entry**

Remove the checkbox/status, feedback warning, AudioContext/source state, fetch/ack/cancel
functions, FIFO pump, TTS event handlers, client ID, toggle listener, and start payload
fields. Add an `/listen` link opening a new tab with `rel="noopener"`. Preserve all ASR,
translation, context, subtitle, scroll, and font behavior.

- [ ] **Step 4: Run tests and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_utils.py tests/test_demo_streaming_ws_protocol.py
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_utils.py
git commit -m "refactor: move TTS playback out of subtitle page"
```

## Task 6: Bounds, Tracing, and Documentation

**Files:**
- Modify: `voxbridge/cli/demo_streaming_ws.py`
- Modify: `tests/test_demo_streaming_ws_utils.py`
- Modify: `tests/test_release_docs.py`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write parser/docs tests first**

Require `--tts-listener-queue-size 128` and public documentation of `/listen`, `/ws/tts`,
future-only joins, multi-listener FIFO, shared synthesis, authentication, and main-page
decoupling.

- [ ] **Step 2: Verify RED**

Expected: parser rejects the option and documentation lacks the contract.

- [ ] **Step 3: Add bounds and safe tracing**

Add the positive integer CLI option. Trace listener connect/disconnect, publish,
received, overflow, and prune using only counts plus opaque hashes. Never log translated
text, raw listener/job IDs, or auth owner keys.

- [ ] **Step 4: Update public docs**

Document other-device login and `/listen`, local-only Start/Stop, future-only stable
translations, independent FIFO, shared synthesis, deprecated private TTS APIs, TTL/job/
listener bounds, CPU serialization, and slow-listener behavior. Preserve `.venv`, port
`8024`, user systemd, and restart-safety instructions.

- [ ] **Step 5: Run tests and commit**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_demo_streaming_ws_utils.py tests/test_release_docs.py
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_utils.py tests/test_release_docs.py docs/API.md docs/DEPLOYMENT.md README.md CHANGELOG.md
git commit -m "docs: document multi-device TTS listener"
```

## Task 7: Verification and Deployment

**Files:**
- Modify only if verification exposes a defect.

- [ ] **Step 1: Run full automated verification**

```bash
git diff --check
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
```

- [ ] **Step 2: Run a two-listener stress script**

Publish at least 200 jobs to two TestClient listener sockets, acknowledge at different
rates, disconnect one, and verify both preserve order, disconnect is isolated, retained
jobs return to zero, and synthesis count equals unique fetched jobs rather than listener
fetch count.

- [ ] **Step 3: Self-review against the design**

Check every section of
`docs/superpowers/specs/2026-07-28-tts-listener-broadcast-design.md`. Scan changed files
for translated-text logging, hard-coded language behavior, placeholder code, and stale
main-page player references.

- [ ] **Step 4: Merge locally and restart `8024` safely**

Stop `voxbridge-8024.service`; confirm the old main PID and EngineCore exit and `8024`
is free; start once; verify `ActiveState=active`, `NRestarts=0`, one main process, one
EngineCore, and one `8024` listener.

- [ ] **Step 5: Browser-test through the deployed HTTPS endpoint**

Open `https://ushome.amycat.com:18024/listen` in two independent authenticated browser
contexts. Click Start in both, run one main-page ASR session, verify future jobs arrive
and play FIFO, stop one listener without affecting the other, rejoin without replay,
and stop ASR while already queued browser audio finishes. Check desktop and narrow
mobile viewport.

- [ ] **Step 6: Final health checks**

```bash
systemctl --user show voxbridge-8024.service -p ActiveState -p SubState -p MainPID -p NRestarts
ss -lntp | rg ':8024'
systemd-cgls --user-unit voxbridge-8024.service --no-pager
git status --short --branch
```

Expected: active service, one EngineCore, clean repository, and no runtime artifacts.
