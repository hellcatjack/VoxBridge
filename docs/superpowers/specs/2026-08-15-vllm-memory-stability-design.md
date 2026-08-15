# vLLM Memory and Three-Hour Stability Design

## Objective

Reduce the memory charged to `voxbridge-8024.service` without degrading
single-microphone streaming recognition, sentence completeness, translation,
or TTS behavior. The selected configuration must survive a three-hour
real-time full-stack run on port `8024` with one EngineCore process and no
monotonic memory growth.

## Baseline

The 2026-08-15 production baseline uses Qwen3-ASR-1.7B with:

- `--max-model-len 8192`
- `--max-num-batched-tokens 8192`
- `--mm-processor-cache-gb 0.5`
- `--gpu-memory-utilization 0.09`
- one allowed ASR connection

After approximately ten hours, the service cgroup held `13.290 GiB`, with a
`13.466 GiB` peak. EngineCore held `11.815 GiB` of GTT. A 30-second idle sample
showed no growth, and the cgroup reported no OOM, swap, or pressure events.

vLLM 0.14 sets both `max_num_encoder_input_tokens` and `encoder_cache_size` to
`max_num_batched_tokens`. At `8192`, startup profiles about 21 maximum-sized
audio items even though VoxBridge admits only one ASR connection. The
multimodal processor cache is also mirrored between the API process and the
EngineCore, so the current `0.5 GiB` setting has a theoretical `1 GiB` host
memory budget.

## Considered Approaches

### Recommended: staged cache reduction with measured GTT floor

Keep `max_model_len=8192` initially. Disable the low-value mirrored processor
cache, lower `max_num_batched_tokens` one step at a time, then lower
`gpu_memory_utilization` only after the reduced startup workspace has been
measured. This preserves long-sequence support and isolates every source of
memory reduction.

### Aggressive context reduction

Lower `max_model_len` to `4096` together with the cache changes. This can save
more KV memory but raises the risk that a long 60-second segment is rejected or
chunked differently. It is reserved for a second phase only if the recommended
path cannot produce a material reduction.

### Worker/process redesign

Move ASR, TTS, and the HTTP gateway into separate processes and load models
lazily. This improves fault isolation but does not inherently reduce the total
model footprint and adds queues, copies, and restart states. It is not justified
until configuration-level optimization is exhausted.

## Optimization Sequence

1. Record a fresh cold-start baseline using the existing production unit.
2. Test `mm_processor_cache_gb=0` while retaining all other baseline settings.
3. Test `max_num_batched_tokens` at `4096`, `2048`, and `1024`, stopping when
   startup fails, a maximum audio item is rejected, or replay latency regresses.
4. For the lowest safe batch value, lower `gpu_memory_utilization` in small
   steps. The selected value must cold-start three consecutive times and retain
   enough KV cache for one `8192`-token request.
5. Test the winning candidate with local Chinese and English audio at real-time
   speed and compare committed text, translation IDs, queue drops, partial
   cadence, and final suffix coverage against the baseline.
6. Add systemd containment after the steady-state value is known. The memory
   ceiling must leave at least 25 percent above the measured cold-start peak,
   and `TasksMax` must permit one current process tree but reject a duplicate
   EngineCore tree.
7. Run a three-hour full-stack soak and retain machine-readable metrics.

Only one variable changes in each short A/B round. Any failed round restores
the last known-good production unit before continuing.

## Benchmark and Monitoring Design

A repository tool will stream a 16 kHz mono PCM WAV to the existing WebSocket
protocol at real-time speed. It will write received events incrementally rather
than retaining three hours of subtitles in its own memory. A separate sampler
will write JSONL rows containing:

- timestamp and elapsed time
- systemd active state, PID, restart count, cgroup current and peak memory
- main-process and EngineCore RSS/PSS, file descriptor count, and thread count
- EngineCore GTT from DRM `fdinfo`
- EngineCore process count
- `8024` health status
- TTS listener, queue, preparation, and pending-audio status

The sampler must not depend on a process name truncated by `ps`; it will obtain
the service cgroup and enumerate its PIDs directly.

## Three-Hour Workload

The soak uses the existing long Chinese sermon WAV as the continuous source and
loops audio without reconnecting until three hours have elapsed. Audio is sent
at `1.0x`, not accelerated. Translation remains enabled, and one shared HLS
listener is kept active so Kokoro synthesis, FFmpeg, caption publication, and
listener cleanup are exercised rather than remaining idle.

Short preflight runs use `audios/2mins_16k.wav`,
`audios/BREAKING_16k.wav`, and `audios/repeat22_16k.wav` to cover English,
Chinese/repetition behavior, and rapid speech before the long soak begins.

## Acceptance Criteria

- `voxbridge-8024.service` remains active for the entire three hours with zero
  restart, OOM, memory-high, or memory-max events.
- Exactly one EngineCore exists throughout the run and after client shutdown.
- Port `8024` remains reachable and the health endpoint never fails twice in a
  row.
- After the first 15-minute warm-up, cgroup memory has no sustained positive
  slope greater than `32 MiB/hour`; the final 30-minute mean may not exceed the
  first post-warm-up 30-minute mean by more than `128 MiB`.
- The selected configuration reduces steady cgroup memory by at least 10
  percent from `13.290 GiB`, unless every lower candidate fails the quality or
  cold-start gates.
- No audio queue frames are dropped during the controlled real-time replay.
- Every committed sentence receives either a translation or an explicitly
  logged translation error; silent translation gaps are rejected.
- Subtitle self-check reports no completed-sentence rollback introduced by the
  candidate, and no new final-suffix gap appears relative to baseline.
- Stop completes, all pending final translations drain, and the service returns
  to a stable idle memory level within five minutes.

## Rollback and Safety

Before each variant, save the exact user-unit contents and current process/GTT
snapshot. Restart only through `systemctl --user`; never launch a second manual
backend. After every stop, verify that the previous main PID and EngineCore PID
have exited before starting the next variant. A failed health check, duplicate
EngineCore, queue drop, missing final event, or memory-limit event immediately
restores the previous configuration.

The local service remains fixed on port `8024`, and all Python commands use
`/data/Qwen3-ASR/.venv/bin/python`.
