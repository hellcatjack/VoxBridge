# Runtime Stability and TTS Finality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this project, so execution remains inline in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce bounded vLLM CPU cache and unbounded log growth while preventing the newest revisable source sentence from being published to TTS prematurely.

**Architecture:** Add an explicit vLLM multimodal cache option, install a user-level logrotate timer, and extend the FIFO TTS stability buffer with newest-source grace plus monotonic backend sealing. Segment finalization seals sources only after final ASR reconciliation; ordinary sentences retain the existing three-second latency.

**Tech Stack:** Python 3.12, argparse, vLLM 0.14, FastAPI/WebSocket, pytest, systemd user units, logrotate 3.21.

---

### Task 1: Expose the vLLM multimodal processor cache budget

**Files:**
- Modify: `tests/test_demo_streaming_ws_utils.py`
- Modify: `voxbridge/cli/demo_streaming_ws.py`

- [ ] **Step 1: Write failing CLI and kwargs tests**

Add tests that require the default and override behavior and a small helper that
constructs the kwargs passed to `Qwen3ASRModel.LLM`:

```python
def test_parse_args_uses_bounded_vllm_mm_processor_cache_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    assert parse_args().mm_processor_cache_gb == 0.5


def test_parse_args_accepts_vllm_mm_processor_cache_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--mm-processor-cache-gb", "0.25"])
    assert parse_args().mm_processor_cache_gb == 0.25


def test_parse_args_rejects_negative_vllm_mm_processor_cache(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--mm-processor-cache-gb", "-0.1"])
    with pytest.raises(SystemExit):
        parse_args()


def test_vllm_model_kwargs_include_bounded_processor_cache():
    args = SimpleNamespace(
        gpu_memory_utilization=0.08,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_new_tokens=32,
        mm_processor_cache_gb=0.5,
    )
    assert _vllm_model_kwargs(args)["mm_processor_cache_gb"] == 0.5
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py \
  -k 'mm_processor_cache or vllm_model_kwargs'
```

Expected: failures because the argument and helper do not exist.

- [ ] **Step 3: Implement the argument and helper**

Add:

```python
def _vllm_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enforce_eager": True,
        "max_new_tokens": int(args.max_new_tokens),
        "mm_processor_cache_gb": float(args.mm_processor_cache_gb),
    }
```

Add the parser option with `type=_non_negative_float_arg`, default `0.5`, and
replace the inline vLLM keyword list with `**_vllm_model_kwargs(args)`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command and expect all selected tests to pass.

- [ ] **Step 5: Commit the cache change**

```bash
git add voxbridge/cli/demo_streaming_ws.py tests/test_demo_streaming_ws_utils.py
git commit -m "perf: bound vllm multimodal processor cache"
```

### Task 2: Add newest-source grace and source sealing to the TTS buffer

**Files:**
- Modify: `tests/test_tts_jobs.py`
- Modify: `voxbridge/tts/jobs.py`

- [ ] **Step 1: Write failing buffer tests**

Add tests for these exact behaviors:

```python
def test_newest_source_uses_additional_revision_grace():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(
        stable_sec=3.0,
        latest_revision_grace_sec=4.0,
        clock=clock,
    )
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "first", "English")
    clock.advance(3.0)
    assert buffer.drain() == []
    clock.advance(4.0)
    ready = buffer.drain()
    assert [item.text for item in ready] == ["first"]
    assert ready[0].release_reason == "latest_revision_grace"


def test_successor_removes_latest_grace_from_preceding_source():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(
        stable_sec=3.0,
        latest_revision_grace_sec=4.0,
        clock=clock,
    )
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "first", "English")
    clock.advance(3.0)
    buffer.register("s2", 1, 1)
    ready = buffer.drain()
    assert [item.text for item in ready] == ["first"]
    assert ready[0].release_reason == "quiet_window"


def test_sealed_newest_source_releases_without_arbitrary_timer():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(
        stable_sec=3.0,
        latest_revision_grace_sec=4.0,
        clock=clock,
    )
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "final", "English")
    assert buffer.seal_through(0) is True
    ready = buffer.drain()
    assert [item.text for item in ready] == ["final"]
    assert ready[0].release_reason == "source_sealed"


def test_translation_ready_after_seal_releases_current_revision():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(
        stable_sec=3.0,
        latest_revision_grace_sec=4.0,
        clock=clock,
    )
    buffer.register("s1", 1, 0)
    buffer.seal_through(0)
    assert buffer.drain() == []
    buffer.mark_ready("s1", 1, "final", "English")
    assert [item.release_reason for item in buffer.drain()] == ["source_sealed"]
```

Also add negative grace validation, monotonic/idempotent sealing, deadline, wait
state, and revision-reset coverage.

- [ ] **Step 2: Run buffer tests and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_tts_jobs.py
```

Expected: failures because `latest_revision_grace_sec` and `seal_through` do not
exist.

- [ ] **Step 3: Implement finality-aware FIFO release**

Extend the constructor:

```python
def __init__(
    self,
    *,
    stable_sec: float,
    latest_revision_grace_sec: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
```

Track `_latest_revision_grace_sec`, `_highest_source_order`, and
`_sealed_through`. Implement:

```python
def seal_through(self, source_order: int) -> bool:
    if source_order < 0:
        raise ValueError("source_order must not be negative")
    with self._lock:
        next_value = max(self._sealed_through, int(source_order))
        changed = next_value != self._sealed_through
        self._sealed_through = next_value
        return changed
```

Centralize the effective deadline and release reason. A sealed entry is
immediately eligible, the highest unsealed order uses
`stable_sec + latest_revision_grace_sec`, and all other orders use `stable_sec`.
Preserve FIFO, failed-head skipping, force drain, and revision reset behavior.

- [ ] **Step 4: Run buffer tests and verify GREEN**

Run the Step 2 command and expect all tests to pass.

- [ ] **Step 5: Commit the buffer change**

```bash
git add voxbridge/tts/jobs.py tests/test_tts_jobs.py
git commit -m "fix: gate TTS on backend source finality"
```

### Task 3: Integrate finality with streaming segment lifecycle

**Files:**
- Modify: `tests/test_demo_streaming_ws_utils.py`
- Modify: `tests/test_demo_streaming_ws_protocol.py`
- Modify: `voxbridge/cli/demo_streaming_ws.py`

- [ ] **Step 1: Write failing CLI and protocol tests**

Require `--tts-latest-revision-grace-sec` to default to `4.0`, accept overrides,
and reject negatives. Extend `_args()` with a zero grace so legacy timing tests
remain intentional.

Add one protocol test where a ready newest source remains unpublished after the
ordinary quiet window, then a segment finalization publishes exactly its latest
revision with `source_sealed`. Add a source-order test asserting a successor can
release its predecessor without waiting the extra grace. Assert the segment
sealing call appears after final reconciliation and before the old candidate
cursor is reset.

- [ ] **Step 2: Run focused integration tests and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_demo_streaming_ws_utils.py \
  tests/test_demo_streaming_ws_protocol.py \
  -k 'latest_revision_grace or segment_seals_tts or successor_releases_tts'
```

Expected: failures because the CLI option and integration do not exist.

- [ ] **Step 3: Implement CLI wiring and segment sealing**

Construct the buffer with:

```python
RevisionStableTTSBuffer(
    stable_sec=tts_revision_stable_sec,
    latest_revision_grace_sec=tts_latest_revision_grace_sec,
)
```

Add an async `_seal_tts_sources_through_current_segment(reason)` helper. Under
`tts_transition_lock`, seal through `tts_runtime.next_source_order - 1`, trace
`tts_source_sealed`, drain, and publish ready entries. Call it only after
`_update_sentence_commits(... final_reconcile=True)` succeeds in
`_finalize_segment_and_rotate`, before resetting `candidate_sentence_ids` and
switching to the new state.

Extend `tts_stability_wait` traces with the effective required quiet time and
whether newest-source grace is active. Wake the scheduler when sealing changes
eligibility.

- [ ] **Step 4: Run focused and full TTS protocol tests**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_tts_jobs.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit streaming integration**

```bash
git add voxbridge/cli/demo_streaming_ws.py \
  tests/test_demo_streaming_ws_protocol.py \
  tests/test_demo_streaming_ws_utils.py
git commit -m "fix: seal TTS sources after segment finalization"
```

### Task 4: Add bounded user-level log rotation

**Files:**
- Create: `deploy/logrotate/voxbridge.conf`
- Create: `deploy/systemd/voxbridge-logrotate.service`
- Create: `deploy/systemd/voxbridge-logrotate.timer`
- Modify: `tests/test_release_docs.py`

- [ ] **Step 1: Write failing deployment-contract tests**

Add a test that requires both production logs, `size 512M`, `rotate 21`,
`compress`, and `copytruncate` in the logrotate template. Require the timer to
contain `OnUnitActiveSec=1h`, `Persistent=true`, and the service to use a
user-owned state path.

- [ ] **Step 2: Run the deployment test and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q \
  tests/test_release_docs.py -k log_rotation
```

Expected: failure because deployment templates do not exist.

- [ ] **Step 3: Add the templates**

Create `deploy/logrotate/voxbridge.conf`:

```text
/data/Qwen3-ASR/logs/voxbridge_8024.log /data/Qwen3-ASR/logs/voxbridge_subtitle_trace.jsonl {
    size 512M
    rotate 21
    compress
    missingok
    notifempty
    copytruncate
}
```

Create a oneshot user service that runs:

```ini
ExecStartPre=/usr/bin/mkdir -p %h/.local/state/voxbridge
ExecStart=/usr/sbin/logrotate --state %h/.local/state/voxbridge/logrotate.status %h/.config/voxbridge/logrotate.conf
```

Create an hourly persistent timer targeting that service.

- [ ] **Step 4: Validate templates and verify GREEN**

Run the Step 2 test and:

```bash
/usr/sbin/logrotate --debug \
  --state /tmp/voxbridge-logrotate-test.status \
  deploy/logrotate/voxbridge.conf
```

Expected: valid parse and no live rotation in debug mode.

- [ ] **Step 5: Commit rotation templates**

```bash
git add deploy tests/test_release_docs.py
git commit -m "ops: add bounded user log rotation"
```

### Task 5: Document runtime budgets and TTS finality

**Files:**
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/test_release_docs.py`

- [ ] **Step 1: Add failing documentation assertions**

Require the docs to mention:

- `--mm-processor-cache-gb 0.5`;
- `--tts-latest-revision-grace-sec 4.0`;
- only the newest sentence receives extra grace and there is no global seven-second delay;
- backend segment sealing;
- installation and verification of `voxbridge-logrotate.timer`.

- [ ] **Step 2: Run docs tests and verify RED**

```bash
/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_release_docs.py
```

- [ ] **Step 3: Update documentation**

Document the CLI contracts, memory multiplication behavior, release reasons,
rotation retention, installation commands, and rollback procedure. Do not include
authentication hashes or local secrets.

- [ ] **Step 4: Run docs tests and verify GREEN**

Run the Step 2 command and expect all tests to pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/API.md docs/DEPLOYMENT.md CHANGELOG.md tests/test_release_docs.py
git commit -m "docs: explain bounded cache logs and TTS finality"
```

### Task 6: Full verification and production deployment

**Files:**
- Modify outside Git: `~/.config/systemd/user/voxbridge-8024.service`
- Install outside Git: `~/.config/voxbridge/logrotate.conf`
- Install outside Git: `~/.config/systemd/user/voxbridge-logrotate.service`
- Install outside Git: `~/.config/systemd/user/voxbridge-logrotate.timer`

- [ ] **Step 1: Run complete branch verification**

```bash
/data/Qwen3-ASR/.venv/bin/python -m compileall -q voxbridge tests
/data/Qwen3-ASR/.venv/bin/python -m pytest -q
git diff --check
test -z "$(git status --porcelain)"
```

- [ ] **Step 2: Fast-forward main and re-run the full suite**

Verify main is still the branch merge-base, then use `git merge --ff-only` and
repeat Step 1 from `/data/Qwen3-ASR/VoxBridge`.

- [ ] **Step 3: Install user rotation configuration**

Copy the tracked logrotate and systemd files to the user configuration paths,
run `systemctl --user daemon-reload`, enable the timer, and verify
`logrotate --debug` parses the installed config.

- [ ] **Step 4: Perform the one controlled service interruption**

Record the old VoxBridge and EngineCore PIDs. Stop
`voxbridge-8024.service`, confirm both old processes exit, move each oversized
active log to a collision-free numbered archive, add these service flags, and
start the service:

```text
--mm-processor-cache-gb 0.5
--tts-latest-revision-grace-sec 4.0
```

Do not launch a manual backend. Wait for the managed service to bind `8024`.

- [ ] **Step 5: Verify production topology and endpoints**

Require one `voxbridge.cli.demo_streaming_ws`, one `VLLM::EngineCore`, no old
PIDs, `NRestarts=0`, one `8024` listener, HTTPS root `303`, login `200`, and no
startup traceback, OOM, or fatal error.

- [ ] **Step 6: Verify production behavior**

Use authenticated Playwright to load the main page and `/listen`, start and stop
the listener, and confirm the WebSocket statuses. Confirm startup logs include
`mm_processor_cache_gb: 0.5`. Run the focused TTS finality protocol tests once
more from the deployed tree.

- [ ] **Step 7: Verify rotation and resource baseline**

Confirm the timer is active, the installed config passes logrotate debug mode,
new active logs are small, archived logs remain readable, and initial GTT/RSS
show one engine with no duplicate process.

- [ ] **Step 8: Clean the feature worktree**

After all checks pass, remove only
`/data/Qwen3-ASR/.worktrees/voxbridge-runtime-stability` and delete
`opt/runtime-stability-tts-finality`.
