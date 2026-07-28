# TTS Translation Revision Stability Gate Design

## Objective

Prevent translated speech from starting while the source sentence can still be
revised. Text subtitles must remain responsive, but TTS may publish only the latest
translation of a sentence after that source revision has remained unchanged for a
short, configurable quiet window.

The initial production value is 3.0 seconds. This is intentionally separate from the
existing subtitle and early-translation stability thresholds: a sentence can be
stable enough to display and translate before it is safe enough to speak.

## Problem Statement

The current producer registers a source sentence when it emits
`sentence_committed`, starts translation, and publishes the translation to the TTS
broadcast hub as soon as that translation completes. The same sentence may later
receive a `sentence_updated` event with a higher revision. `OrderedTTSBuffer` prevents
one sentence ID from being emitted twice, so an already published revision cannot be
retracted or replaced for listeners that are already playing it.

Recent production tracing showed 30 TTS-issued sentences in one session. Eight were
published at revision 1 and then revised to revision 2 between 759 ms and 2919 ms
later. Therefore, 26.7% of issued jobs in that sample were provably published before
their source sentence stopped changing. This is a backend lifecycle defect rather
than a listener playback-order defect.

## Scope

This change adds a backend-only TTS revision stability gate between translated text
completion and TTS publication. It applies to both translation directions and all
listener devices.

The following behavior remains unchanged:

- Partial and committed source subtitles continue to reach the main page immediately.
- Translation requests may still begin at the existing early-translation point.
- The standalone `/listen` page, `/ws/tts` protocol, synthesis cache, and per-device
  FIFO playback queues do not change.
- The source text, translation text, VAD, sentence segmentation, and ASR model are not
  modified by this feature.
- A listener continues to receive only jobs published after it joins.

This design does not attempt to retract audio that has already been published or
started. A revision arriving after the speech-stability window is logged for tuning,
but it does not generate a second spoken version of the same sentence.

## Selected Approach

Add a revision-aware quiet window to the existing ordered TTS pipeline.

For each source sentence, the backend records the latest revision-change time. A
completed translation becomes eligible for speech only when all of these conditions
are true:

1. It belongs to the sentence's current revision.
2. That revision has not changed for `--tts-revision-stable-sec`.
3. Every earlier source order is ready, failed, or otherwise resolved.
4. The translation contains non-empty output for the configured target language.

The initial default for `--tts-revision-stable-sec` is `3.0`. The quiet window is
measured from the latest source revision, not from translation completion. If the
translation finishes after the quiet window has already elapsed, it can be released
immediately after it becomes ready.

### Rejected Alternatives

- **Speak immediately after translation:** Lowest latency, but it preserves the
  demonstrated race where revision 1 is spoken before revision 2 arrives.
- **Wait for the next sentence:** Strong natural confirmation, but the final sentence
  can wait indefinitely during silence and speech latency depends on the next speaker.
- **Use a fixed delay from translation completion:** Simple, but translation latency is
  unrelated to source stability. A slow translation would wait unnecessarily, while a
  fast translation could still precede a later source revision.
- **Synthesize then delay playback:** Wastes CPU on translations that may be invalidated
  and complicates listener cancellation. No synthesis should start before publication.

## Architecture

### TTS Revision Stability Gate

Introduce `RevisionStableTTSBuffer` in `voxbridge/tts/jobs.py` as a focused,
independently testable backend component responsible only for speech eligibility. It
owns one entry per unresolved source order and accepts these state transitions:

- `register(sentence_id, revision, source_order, changed_at)` records a new sentence or
  a newer revision. A newer revision clears any ready translation for the old revision
  and resets the quiet-window deadline.
- `mark_ready(sentence_id, revision, text, target_language)` records a completed
  translation only if its revision is still current. Superseded translation results
  are discarded.
- `mark_failed(sentence_id, revision)` resolves a current translation failure without
  blocking all later source orders indefinitely.
- `drain(now)` returns current, mature translations in strict source order.
- `force_drain(now)` is reserved for orderly producer finalization after ASR flushing
  and translation draining are complete.
- `reset()` clears all session-local state on disconnect, restart, direction change,
  or failed startup.

The component uses an injected monotonic clock so timing behavior can be tested without
wall-clock sleeps. It must not know about WebSockets, HTTP routes, browser listeners,
or Kokoro synthesis.

`RevisionStableTTSBuffer` replaces `OrderedTTSBuffer` in the ASR WebSocket session and
absorbs its strict-order and exactly-once behavior. The old class and its direct tests
are migrated rather than wrapped. This leaves one authoritative ordering and revision
state machine instead of two nested buffers with overlapping responsibilities.

### Session-Local Scheduler

Each ASR WebSocket session owns at most one asynchronous stability scheduler task. The
scheduler sleeps until the earliest unresolved quiet-window deadline, wakes, acquires
the existing TTS transition lock, drains newly eligible items, and publishes them to
the broadcast hub.

The scheduler must be independent of incoming audio, VAD events, translation
completion callbacks, and model decode calls. Otherwise a speaker who stops and sends
only filtered silence could leave the last sentence pending forever.

Registering a newer revision or resolving an earlier source order signals the
scheduler to recalculate its next deadline. The design must not create one unmanaged
task per sentence; a single task plus an event provides bounded lifecycle management.

### Existing Broadcast Layer

The broadcast hub remains downstream of the stability gate. Only drained stable items
are published. The hub continues to fan out one immutable job to the listener snapshot
and synthesize one shared WAV on demand. The listener page cannot observe pending or
superseded translations.

## State Model

Each unresolved ordered entry contains:

- `sentence_id`: stable source identity.
- `revision`: latest registered source revision.
- `source_order`: immutable meeting order.
- `changed_at`: monotonic time of the latest registered revision.
- `status`: `waiting`, `ready`, or `failed`.
- `text` and `target_language`: populated only for a ready current revision.
- `released`: terminal marker retained only as needed to reject late duplicate work.

State transitions are:

```text
unseen -> waiting(revision N)
waiting(revision N) -> ready(revision N)
waiting/ready(revision N) -> waiting(revision N+1)
waiting(revision N) -> failed(revision N)
ready(revision N) + quiet window elapsed + order unblocked -> released
failed(revision N) + order unblocked -> resolved without speech
```

Registering the same or an older revision is idempotent. A sentence ID cannot change
source order, and a source order cannot be reused by another sentence ID. Invalid
transitions raise or log a non-sensitive invariant error rather than silently
reordering speech.

## Runtime Data Flow

### New Sentence

1. The backend emits `sentence_committed` to the subtitle page as it does today.
2. It registers revision 1 and records `changed_at` using the monotonic clock.
3. Translation starts according to the existing translation policy.
4. Translation completion marks revision 1 ready but does not publish it immediately.
5. The scheduler releases it only after revision 1 has remained unchanged for 3.0
   seconds and all earlier source orders are resolved.

### Sentence Revision

1. The backend emits `sentence_updated` to the subtitle page immediately.
2. It registers the higher revision and updates `changed_at`.
3. Any old ready translation becomes superseded and ineligible for speech.
4. The scheduler deadline is reset from the new revision time.
5. The latest revision is translated. Stale completion from an older request is
   ignored by revision comparison.
6. Only the latest translation can be released after the renewed quiet window.

### Translation Completes After the Deadline

If the source revision has already been unchanged for at least 3.0 seconds when its
translation finishes, the translation callback signals the scheduler and the entry is
eligible immediately. It does not wait for another full 3.0 seconds.

### Strict Ordering

An earlier waiting translation blocks later ready translations so listeners hear the
meeting in source order. When that earlier entry matures or fails, one drain operation
may release a batch of consecutive eligible entries. A slow or failed translation
continues to use the existing bounded translation timeout and failure path; the new
gate must not introduce an unbounded ordering stall.

## Finalization and Cancellation

### Normal Stop and End of Stream

Orderly finalization follows this sequence:

1. Flush the current ASR streaming state and reconcile the final source revision.
2. Start or update translation for that final revision.
3. Drain pending translation tasks using the existing final translation drain policy.
4. Under the TTS transition lock, force-release only the latest ready revisions in
   source order because no further ASR revisions are expected.
5. Mark the producer inactive after all released jobs have been published.
6. Cancel and await the stability scheduler task.

`force_drain` does not publish stale revisions, empty translations, or unresolved
translation requests. It bypasses only the remaining quiet-window duration.

### Abrupt Disconnect or Failed Session

An abrupt WebSocket disconnect, authentication failure, startup failure, or exception
does not force speech. The scheduler is cancelled and pending gate state is discarded.
This avoids speaking text after an abandoned session whose final revision was never
reconciled.

### Direction or Session Reset

A new start and any accepted translation-direction change create a fresh gate and
scheduler. Source ordering, deadlines, and emitted sentence IDs never cross session
boundaries.

## Configuration

Add one backend CLI option:

```text
--tts-revision-stable-sec 3.0
```

Semantics:

- Non-negative seconds measured with a monotonic clock from the latest source revision.
- `0` disables the quiet-window delay for controlled tests or compatibility, while
  preserving revision and source-order checks.
- The value is independent of VAD silence, ASR stable-hit thresholds, translation
  polling intervals, and listener queue settings.
- The user service should set the production value explicitly to `3.0` so behavior is
  visible in deployment configuration rather than relying only on a parser default.

No frontend setting is added. Speech stability is a backend correctness policy, not a
per-listener presentation preference.

## Trace and Diagnostics

Add structured events to the existing TTS trace topic. Normal logs contain IDs only as
the existing redacted short fingerprints and never include source or translated text.

### `tts_stability_wait`

Emitted when a current translation is ready but its source revision is not yet mature:

- sentence fingerprint, revision, and source order
- current quiet age in milliseconds
- required quiet age in milliseconds
- remaining delay in milliseconds
- whether it is blocked by an earlier source order

### `tts_stability_reset`

Emitted when a newer source revision invalidates a pending translation or restarts the
quiet window:

- sentence fingerprint and source order
- previous and new revision
- previous quiet age in milliseconds
- whether the previous revision already had a ready translation

### `tts_stability_release`

Emitted immediately before publication:

- sentence fingerprint, revision, and source order
- release reason: `quiet_window` or `final_force`
- source quiet age in milliseconds
- translation-ready age in milliseconds
- ordered backlog depth

### `tts_late_revision_after_release`

Emitted if a newer revision arrives after the sentence has already been released:

- sentence fingerprint and source order
- released and incoming revision
- elapsed milliseconds since release

This event is the primary signal for deciding whether 3.0 seconds is sufficient. It
does not trigger replacement audio because already-playing speech cannot be recalled.

Scheduler start, cancellation, and unexpected failure are also logged with the session
fingerprint and pending-entry count. Repeated `wait` events for the same unchanged
deadline should be suppressed to avoid noisy logs.

## Error Handling

- A superseded translation callback is ignored and traced; it cannot overwrite the
  latest entry.
- Scheduler cancellation is expected during session cleanup and is not logged as an
  error.
- An unexpected scheduler exception disables TTS publication for that session and
  reports TTS unavailable without stopping ASR or text translation.
- A translation failure resolves its source order through the existing failure path so
  later speech can continue.
- Broadcast or synthesis errors retain their existing listener-isolated behavior.
- Clock values and deadlines use `time.monotonic()` exclusively and are unaffected by
  wall-clock adjustments.

## Testing Strategy

### Unit Tests

Use a fake monotonic clock to verify:

- A ready revision is withheld before 3.0 seconds and released at 3.0 seconds.
- A revision update at 2.9 seconds clears the old translation and restarts the window.
- A stale translation completion after an update is ignored.
- Multiple revisions produce exactly one ready item containing only the final text.
- A translation completing after the source deadline releases immediately.
- Later source orders remain blocked until earlier orders mature or fail.
- Consecutive mature entries drain in source order.
- `force_drain` releases only latest ready revisions.
- Reset clears deadlines, ordering, and emitted history.
- A revision after release is rejected and reported as a late revision.

### WebSocket Integration Tests

Use deterministic fake ASR and translation implementations to verify:

- `sentence_committed` and `sentence_updated` remain immediately visible to the main
  client while `/ws/tts` receives no early job.
- A fast revision-1 translation followed by revision 2 within the quiet window emits
  only one TTS job for revision 2.
- Filtered silent audio does not need to trigger inference for the scheduler to release
  a mature sentence.
- A normal stop flushes final ASR state, waits for final translation, force-releases the
  latest revision, and then sends producer inactive status.
- An abrupt disconnect cancels pending speech rather than force-releasing it.
- Two listeners receive the same single stable job and preserve existing independent
  FIFO playback behavior.
- A translation failure does not permanently block later source orders.

### Regression and Runtime Verification

- Run the complete test suite with `/data/Qwen3-ASR/.venv/bin/python -m pytest -q`.
- Restart only `voxbridge-8024.service` and verify exactly one main process and one
  EngineCore process.
- Confirm port 8024 with `ss -lntp | rg ':8024'`.
- Reproduce a sentence that receives revision 1 and revision 2 approximately 0.7 to
  3.0 seconds apart. Verify subtitles update immediately but only revision 2 appears in
  `tts_stability_release` and on `/listen`.
- Stop speaking after the last sentence and verify the listener receives it after the
  quiet window without requiring another audio/model event.
- Stop the producer before the quiet window expires and verify orderly finalization
  speaks only the final reconciled translation.

## Acceptance Criteria

- No TTS job is published for a source revision superseded before the configured quiet
  window expires.
- A sentence produces at most one spoken job per ASR session.
- The spoken job uses the latest known source revision and translation at release time.
- Text subtitle timing and frontend behavior do not regress.
- A final sentence is eventually spoken during silence without additional inference.
- Orderly stop releases the final stable translation; abrupt disconnect does not speak
  unreconciled text.
- TTS source order remains strict across translation latency differences and failures.
- Trace logs identify waits, resets, releases, and late revisions without leaking text
  or opaque raw job identifiers.
- The default deployment adds approximately 3.0 seconds to first-sentence speech
  eligibility, while continuous playback throughput remains governed primarily by
  synthesis and each listener's FIFO backlog.

## Rollout and Tuning

Deploy with `--tts-revision-stable-sec 3.0` and observe
`tts_late_revision_after_release` for representative Chinese-to-English and
English-to-Chinese sessions. If late revisions still occur, increase the value based on
the measured tail distribution rather than adding new punctuation or language-specific
rules. If no late revisions occur and latency is unacceptable, lower the value in
small steps while retaining the same state machine.

Rollback requires only setting the option to `0`; no frontend, protocol, or stored-data
migration is involved.
