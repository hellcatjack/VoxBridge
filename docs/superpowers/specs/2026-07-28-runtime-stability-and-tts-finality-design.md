# Runtime Stability and TTS Finality Design

## Scope

This change improves three runtime properties of the single-session VoxBridge
deployment on port `8024`:

1. Bound vLLM's duplicated multimodal processor cache at a deployment-appropriate
   size.
2. Rotate the two continuously growing diagnostic logs without restarting the
   service.
3. Prevent translated speech from being published while the newest source
   sentence is still inside the ASR revision horizon.

The frontend protocol and subtitle rendering remain unchanged. No language,
word, punctuation, or transcript-specific rule is introduced.

## Observed Evidence

During a 49-minute production session, every ASR and translation queue remained
healthy, but the following bounded or cumulative risks were measured:

- vLLM used its default `mm_processor_cache_gb=4`. vLLM duplicates this cache in
  the API process and EngineCore, allowing about 8 GiB of CPU cache for unique
  streaming audio inputs.
- `voxbridge_8024.log` and `voxbridge_subtitle_trace.jsonl` grew together at
  about 3.05 GB per day and had no rotation policy.
- 81 `tts_late_revision_after_release` events represented 24 unique source
  sentences out of 369 published sentences. Every first late revision happened
  while that sentence was still the newest registered TTS source. The median
  first late revision arrived 206 ms after release and the latest arrived
  3.62 seconds after release.

The current three-second TTS quiet window therefore measures inactivity, not
backend finality. Increasing the global quiet window to seven seconds would
penalize every sentence and is explicitly prohibited by this design.

## vLLM Cache Design

Add `--mm-processor-cache-gb` as a non-negative CLI option with a default of
`0.5`. Pass it unchanged to `Qwen3ASRModel.LLM`, which forwards keyword arguments
to `vllm.LLM`.

The managed `8024` service will set the option explicitly to `0.5`. With one API
process and one EngineCore, the configured multimodal processor cache budget is
about 1 GiB in total instead of about 8 GiB. A value of zero remains available
for a future benchmark, but is not the production default because vLLM itself
does not recommend disabling the cache without workload validation.

## Log Rotation Design

Install a user-level `voxbridge-logrotate.timer` that invokes logrotate hourly
with a user-owned state file. The policy covers:

- `/data/Qwen3-ASR/logs/voxbridge_8024.log`
- `/data/Qwen3-ASR/logs/voxbridge_subtitle_trace.jsonl`

Each file rotates independently after reaching 512 MiB. Keep 21 compressed
rotations, use `copytruncate` so the running systemd stdout descriptor and the
open subtitle trace handle remain valid, and tolerate missing or empty files.
At the measured rate, this preserves roughly one week of diagnostics while
bounding disk use.

For the first deployment, stop the managed service, rename the two oversized
active files to numbered archives, and restart the service. This avoids copying
about 6.6 GB while the application is writing. Subsequent rotations use the
timer and do not restart VoxBridge.

## TTS Finality State Machine

`RevisionStableTTSBuffer` gains two independent concepts:

- **Quiet stability:** the existing three-second interval since the current
  source revision was registered.
- **Source finality:** whether the source order is no longer the newest source,
  or the backend explicitly sealed it after completing a segment flush.

The release rules are:

1. A non-newest source keeps the existing three-second quiet window.
2. The newest unsealed source receives an additional four-second revision grace.
   Its maximum inactivity threshold is therefore seven seconds, but this applies
   only to that one newest source. It is not a global seven-second delay.
3. Registering a newer source removes the extra grace from preceding sources.
4. After `finish_streaming_transcribe` and final sentence reconciliation complete
   for a VAD or hard-cut segment, the backend seals all source orders produced by
   that segment. A sealed source can publish as soon as its current translation
   is ready, without an arbitrary post-finalization timer.
5. Orderly session finish retains the existing force drain. Abrupt disconnect
   still discards pending speech.
6. Any new revision resets the quiet/grace clock and invalidates an older pending
   translation exactly as it does today.

The buffer exposes `seal_through(source_order)`. Sealing is monotonic and cannot
be reversed. Drain reasons distinguish `quiet_window`, `latest_revision_grace`,
`source_sealed`, and `final_force` for diagnostics.

## Scheduler Behavior

The scheduler's next deadline must use the effective threshold of its FIFO head:

- three seconds for a source with a successor;
- seven seconds only for the newest unsealed source;
- immediate eligibility for a sealed source.

Registering a successor or sealing a segment wakes the scheduler because either
transition can make the FIFO head immediately releasable. Translation completion
continues to wake it as before.

## Failure Handling

- Negative cache or grace values fail argument parsing.
- Log rotation runs independently of VoxBridge. A logrotate failure does not
  restart or stop `8024` and is visible in the timer service status.
- Segment sealing occurs only after successful final ASR reconciliation. A
  failed segment finalize does not seal or publish the newest source.
- TTS publication remains FIFO. A failed translation still advances the failed
  head according to existing behavior.

## Tests

Unit tests must prove:

- CLI defaults and overrides pass `mm_processor_cache_gb=0.5` to vLLM kwargs.
- Negative cache and newest-grace values are rejected.
- A newest ready source is not released at three seconds.
- Registering a successor releases the preceding source after its ordinary
  three-second window.
- The newest source releases after its four-second additional grace.
- Segment sealing releases a ready newest source immediately and releases a
  translation that becomes ready after sealing.
- A revision resets both the ordinary window and newest grace.
- Existing FIFO, failure, final-force, and disconnect behavior remains intact.
- Deployment documentation and tracked rotation templates retain the 512 MiB,
  21-rotation, compressed, user-timer contract.

Protocol tests must demonstrate that a segment finalization seals the final
source only after final reconciliation, while a normal streaming source remains
held. Trace assertions must verify release reasons without exposing source text.

## Deployment Verification

After the full suite passes:

1. Fast-forward the clean main branch.
2. Install the logrotate configuration and user timer.
3. Stop `voxbridge-8024.service` once, archive current oversized logs, update the
   service with `--mm-processor-cache-gb 0.5` and
   `--tts-latest-revision-grace-sec 4.0`, then start it.
4. Confirm only one VoxBridge process and one EngineCore exist, `8024` listens,
   HTTPS login works, and `NRestarts=0`.
5. Confirm startup reports the new vLLM cache argument.
6. Run authenticated Playwright start/stop verification and a focused streaming
   protocol test.
7. Force a dry-run rotation check, without rotating the new small live logs.
