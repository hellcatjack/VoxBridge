# vLLM Memory and Three-Hour Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents for this project.

**Goal:** Reduce the memory charged to `voxbridge-8024.service` while proving that one-microphone ASR, translation, and TTS remain stable for three continuous real-time hours.

**Architecture:** Add read-only cgroup/process/GTT probes and a bounded-memory WebSocket soak client, then use them for one-variable-at-a-time vLLM cold-start experiments. Install the winning flags in the existing user service and add systemd containment only after the measured peak is known.

**Tech Stack:** Python 3.12 from `/data/Qwen3-ASR/.venv`, asyncio, websockets, Linux cgroup v2, procfs DRM fdinfo, pytest, systemd user services, vLLM 0.14, ROCm/GTT.

---

### Task 1: Runtime memory snapshot model

**Files:**
- Create: `voxbridge/debug/runtime_memory.py`
- Test: `tests/test_runtime_memory.py`

- [ ] Write parser tests using temporary cgroup, proc status, smaps, and DRM fdinfo fixtures. Assert byte conversion, exact EngineCore counting, and `None` for unavailable kernel files.
- [ ] Run `/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_runtime_memory.py`; expect import failure.
- [ ] Implement immutable `ProcessMemorySnapshot` and `RuntimeMemorySnapshot` records plus `read_process_memory`, `read_drm_gtt_bytes`, and `read_runtime_memory`. Enumerate only `cgroup.procs`, and identify EngineCore from `/proc/<pid>/comm`.
- [ ] Rerun the focused test; expect all cases to pass.
- [ ] Commit with `git commit -m "feat: add cgroup and GTT memory probes"`.

### Task 2: Streaming JSONL memory sampler

**Files:**
- Create: `tools/runtime_memory_probe.py`
- Test: `tests/test_runtime_memory_probe_cli.py`

- [ ] Write failing tests for `--help`, one-sample JSONL output, per-line flushing, elapsed time, health/TTS status fields, and a missing cgroup.
- [ ] Run `/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_runtime_memory_probe_cli.py`; expect failure.
- [ ] Implement `--cgroup`, `--output`, `--interval-sec`, `--duration-sec`, `--health-url`, and `--tts-status-url`. Flush every row and exit cleanly on SIGINT/SIGTERM without modifying the service.
- [ ] Run the tests and a 60-second live sample against the `voxbridge-8024.service` cgroup. Expect 12 JSON rows, one EngineCore in every row, and HTTP 200.
- [ ] Commit with `git commit -m "feat: add streaming runtime memory sampler"`.

### Task 3: Bounded-memory real-time soak client

**Files:**
- Create: `tools/streaming_soak.py`
- Test: `tests/test_streaming_soak.py`

- [ ] Write failing fake-server tests for WAV validation, real-time pacing, looping without reconnecting, exact duration cutoff, incremental event writes, cookie forwarding, clean `finish`, and final timeout. Assert no growing in-memory event list exists.
- [ ] Run `/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_streaming_soak.py`; expect failure.
- [ ] Implement WAV input, duration, chunk size, language, translation direction, context terms, event JSONL, and environment-only authentication. Never log credentials or cookies.
- [ ] Run the tests and a two-minute live replay of `/data/Qwen3-ASR/audios/2mins_16k.wav` at `1.0x`. Expect a final event and zero server-side queue drops.
- [ ] Commit with `git commit -m "feat: add bounded real-time streaming soak client"`.

### Task 4: One-variable vLLM A/B matrix

**Files:**
- Modify locally: `~/.config/systemd/user/voxbridge-8024.service`
- Record outside git: `/tmp/voxbridge-vllm-memory-ab.jsonl`

- [ ] Save the production unit as `/tmp/voxbridge-8024.service.baseline` and record the current service/cgroup snapshot.
- [ ] Change only `--mm-processor-cache-gb 0.5` to `0`; restart through systemd, verify old PIDs exited, then record cold-start peak and a two-minute replay.
- [ ] With cache zero, test `--max-num-batched-tokens` in order: `4096`, `2048`, `1024`. Keep `--max-model-len 8192` and `--gpu-memory-utilization 0.09`. Reject on startup error, queue drop, new final-suffix loss, or more than 10 percent worse partial cadence.
- [ ] For the lowest safe batch value, test `--gpu-memory-utilization` at `0.08`, `0.075`, `0.07`, `0.065`, and `0.06`. Stop at the first cold-start/KV failure. Select the lowest passing value plus `0.005` safety and cold-start it three times.
- [ ] Require at least a 10 percent cgroup reduction unless every lower candidate fails quality. Restore the baseline unit immediately if no candidate passes.

### Task 5: systemd containment and deployment documentation

**Files:**
- Create: `deploy/systemd/voxbridge-8024-memory.conf`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `tests/test_release_docs.py`
- Install locally: `~/.config/systemd/user/voxbridge-8024.service.d/memory.conf`

- [ ] Add failing release tests requiring `MemoryHigh=16G`, `MemoryMax=20G`, `TasksMax=512`, `OOMPolicy=stop`, cgroup inspection commands, and port `8024`.
- [ ] Run `/data/Qwen3-ASR/.venv/bin/python -m pytest -q tests/test_release_docs.py`; expect failure.
- [ ] Add the tracked and installed drop-in with `MemoryAccounting=yes`, `MemoryHigh=16G`, `MemoryMax=20G`, `TasksMax=512`, and `OOMPolicy=stop`. Document that it does not constrain the external HY-MT process.
- [ ] Reload systemd, restart, and verify the limits, one EngineCore, port `8024`, and `/listen` HTTP 200.
- [ ] Run release tests and commit with `git commit -m "ops: contain VoxBridge runtime memory"`.

### Task 6: Regression matrix

**Files:**
- Output outside git: `/tmp/voxbridge-regression-*.jsonl`

- [ ] Run focused memory-tool, soak-client, subtitle, protocol, and CLI tests.
- [ ] Replay `audios/2mins_16k.wav`, `audios/BREAKING_16k.wav`, and `audios/repeat22_16k.wav` at `1.0x`; save events separately and run subtitle self-check on each.
- [ ] Run `/data/Qwen3-ASR/.venv/bin/python -m pytest -q`; expect zero failures.

### Task 7: Three-hour full-stack soak and final deployment

**Files:**
- Output outside git: `/tmp/voxbridge-soak-3h-events.jsonl`
- Output outside git: `/tmp/voxbridge-soak-3h-memory.jsonl`

- [ ] Start the memory probe for `11,400` seconds at 15-second intervals, including five minutes after stream stop.
- [ ] Keep one persistent HLS listener active and verify `listener_count=1`, one FFmpeg encoder, and no per-listener Kokoro worker.
- [ ] Loop `/data/Qwen3-ASR/audios/fun0ZaornRg_16k.wav` in one WebSocket session for `10,800` seconds at `1.0x` with translation and TTS enabled.
- [ ] Verify zero restart/OOM/limit events, one EngineCore, no queue drop, final and translation completeness, and no new sentence rollback/final-suffix gaps. Require post-warm-up memory slope at or below `32 MiB/hour` and final versus initial 30-minute means within `128 MiB`.
- [ ] Restore the baseline on any failure; otherwise verify active service, one backend, one EngineCore, `8024`, and HTTP 200.

### Task 8: Documentation and final commit

**Files:**
- Modify: `README.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `CHANGELOG.md`

- [ ] Record baseline/current/peak cgroup memory, EngineCore GTT, selected flags, memory slope, queue-drop count, restart count, and rollback command. Do not commit audio, credentials, logs, screenshots, or subtitle traces.
- [ ] Run release tests, the full suite, and `git diff --check`.
- [ ] Commit with `git commit -m "docs: record VoxBridge memory stability profile"`.
