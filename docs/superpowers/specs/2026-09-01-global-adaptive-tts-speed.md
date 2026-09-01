# Global Adaptive TTS Speed Design

## Status

Approved product direction; implementation pending.

## Problem

The shared listener service currently asks each browser to apply its own playback
rate. Desktop Chrome can accelerate live HLS playback, while iPhone Safari stays
near `1.0x` and can repeatedly report buffering. This makes two listeners hear
the same translated sermon with different timing and undermines the requirement
that every listener receive the same experience.

TTS work is intentionally demand-driven: no listener means no active TTS/FFmpeg
pipeline. The backlog controller therefore cannot belong to the first listener
or inherit all translations produced while nobody was listening.

## Goals

- Produce one shared audio stream with the same audible speed for every listener.
- Use server-side Kokoro synthesis speed instead of browser playback-rate control.
- Define backlog independently of any listener identity or client playhead.
- Skip stale translations when the first listener joins a new live session.
- Keep the current speech character at displayed `1.0x`.
- Change speed only between sentences so that a sentence never changes speed
  after playback begins.
- Preserve current multi-listener fan-out and on-demand resource usage.

## Non-goals

- Per-listener speed preferences.
- Replaying the full translation history to a late listener.
- Retiming PCM that has already been synthesized or published.
- Replacing HLS, FFmpeg, or Kokoro with a new transport or TTS engine.
- Using client-local HLS latency to control the global synthesis rate.
- Adding Low-Latency HLS as part of this change.

## Selected approach

Kokoro's native `speed` argument will set the audible rate for each sentence.
The HTML media element will play at `1.0` on every device. A controller owned by
`SharedHLSTTSPublisher` will select one global Auto multiplier from the shared
server-side speech backlog.

This reuses the existing shared HLS publisher and Kokoro's supported synthesis
API. It does not implement a custom time-stretching algorithm.

## Shared speech epoch

A speech epoch is the lifetime of one active shared TTS/HLS stream.

### No listeners

When `listener_count == 0`:

- No encoder or synthesis worker is active.
- Translation may continue, but those translations do not constitute speech
  backlog for a future listener.
- The public global backlog is `0`.
- The public global multiplier is `1.0x`.
- There is no active speech epoch identifier.

Eligible stable items may remain temporarily in the idle publication queue only
so that the newest live item can be selected when somebody joins. They must not
be counted as an active epoch's debt.

### First listener joins (`0 -> 1`)

The publisher will:

1. Discard all idle queued speech except the newest eligible stable item.
2. Remove stale matching entries from known-item and preparation caches.
3. Create a new unique speech epoch.
4. Start the single shared encoder and worker.
5. Synthesize the retained join sentence at displayed `1.0x`.
6. Start normal global Auto decisions with subsequent sentences.

The first listener is an epoch trigger, not the owner of the epoch or controller.

### Additional listeners

When `listener_count` changes from one positive value to another:

- The epoch identifier is unchanged.
- The encoder, worker, global backlog, and selected speed are unchanged.
- No duplicate TTS work is created.
- Every listener receives the same HLS epoch and the same synthesized audio.

If the listener that originally triggered the epoch leaves while another lease
remains, nothing resets.

### Last listener leaves (`1 -> 0`)

After explicit removal or lease expiry of the last listener:

- Stop the encoder, synthesis worker, and reaper.
- Clear queued, inflight, prepared, and known-item state belonging to the epoch.
- Clear caption state belonging to the epoch.
- Clear the global rate-controller state.
- Return public backlog to `0` and multiplier to `1.0x`.

The next listener creates a new epoch and does not inherit this epoch's debt.

## Backlog definition

The controller uses `conservative_backlog_ms`, representing the maximum
reasonable amount of media time still waiting to reach the shared HLS stream.

It is the sum of:

1. Exact PCM media duration currently pending in the encoder.
2. Exact PCM media duration synthesized or prepared but not yet appended.
3. A conservative baseline-speed estimate for the inflight item when exact PCM
   is not yet available.
4. Conservative baseline-speed estimates for known items waiting for synthesis.

Items must appear exactly once in this sum. In particular, an inflight or
prepared item must not be counted a second time through the known-item map.

### Conservative estimates

For unsynthesized speech, use the greater of:

- The existing language-specific default duration estimate.
- A duration derived from observed language-specific speech time per character,
  including a safety margin.

Add the configured inter-sentence pause exactly once. Estimates are made at the
displayed `1.0x` baseline, even when the next sentence may be synthesized faster.
This intentionally avoids under-reporting and avoids a circular dependency
between the backlog and the speed selected from that backlog.

Exact synthesized PCM duration replaces the estimate as soon as it is known.
Encoder pending duration is not divided by the encoder's internal feed capacity;
using media duration is the conservative value visible to listeners.

### Excluded values

The global backlog must not include:

- Translations produced before the active epoch.
- Any individual browser's `currentTime`, buffered ranges, or network delay.
- Audio already published into the shared HLS live window.
- Caption age.
- The identity or connection age of the first listener.

Client-local buffering remains a separate health signal and cannot change the
global TTS speed.

## Auto speed policy

The speed for the next sentence is selected from the conservative backlog at
the moment synthesis begins:

| Conservative backlog | Displayed multiplier |
| ---: | ---: |
| `< 10s` | `1.0x` |
| `>= 10s` and `< 30s` | `1.2x` |
| `>= 30s` and `< 40s` | `1.4x` |
| `>= 40s` | `1.5x` |

The selected multiplier remains fixed for the entire sentence. Boundary
crossings affect only a later sentence. Already synthesized, appended, or
published audio is never mutated.

The first retained sentence of a new epoch is always synthesized at displayed
`1.0x`, regardless of its estimated length. This gives a predictable live join
and prevents idle history from causing an accelerated start.

## Kokoro speed mapping

The current configured Kokoro baseline is `1.05`. To preserve the current voice
at displayed `1.0x`, the synthesis value is:

`effective_kokoro_speed = configured_baseline_speed * displayed_multiplier`

With the current baseline:

| Displayed multiplier | Kokoro `speed` |
| ---: | ---: |
| `1.0x` | `1.05` |
| `1.2x` | `1.26` |
| `1.4x` | `1.47` |
| `1.5x` | `1.575` |

The effective value must continue to satisfy Kokoro's configured validation
range. The synthesizer API must accept the effective speed per synthesis call;
the shared model/session is not recreated for each change.

## Prepared-audio cache

Prepared audio is reusable only when all synthesis inputs match. Its identity
must include at least:

- Sentence/revision/text/language identity.
- Voice identity.
- Effective Kokoro speed.
- Any pause setting that changes PCM duration.

Each prepared result records its effective speed and exact audio duration. If a
prepared result was generated at a different speed from the controller's current
selection, it must not be released as if it had the new speed. It may be evicted
and regenerated, while respecting existing cache-size limits.

Preparation remains serialized through the existing shared worker. This change
must not create concurrent raw-text `Kokoro.create()` calls on one shared
synthesizer.

## Listener-page behavior

All listener media elements use an effective HTML playback rate of `1.0`.
Custom client-side Auto catch-up and manual rate application are removed from the
audio path. hls.js may still manage loading and recovery, but its
`maxLiveSyncPlaybackRate` must remain `1`.

The existing Playback Speed control becomes a read-only global status, such as:

`Auto - 1.2x`

The Live Audio status distinguishes two concepts:

- `Speech backlog: 18s` is shared server-side unpublished speech debt.
- `Global speed: Auto - 1.2x` is the audible rate selected for shared TTS.

`Buffering live audio` is shown only for real local media starvation or startup,
not merely because server-side speech backlog exists. A slow or disconnected
client cannot change the global speed for other listeners.

## Status contract

The HLS status response exposes enough shared state for every listener to render
the same global information:

- `speech_epoch_id`: active epoch identifier, or empty when idle.
- `global_speed_mode`: `auto`.
- `global_speed_multiplier`: one of `1.0`, `1.2`, `1.4`, or `1.5`.
- `tts_effective_speed`: effective Kokoro speed used for the next/current item.
- `translated_audio_backlog_ms`: conservative shared backlog for the active
  epoch, or `0` when idle.
- `translated_audio_backlog_count`: unique unpublished items in the active
  epoch, or `0` when idle.
- `translated_audio_backlog_estimated`: whether any component is estimated.

No status field depends on which listener requested it. Clients treat status as
read-only and do not write the global rate.

## Concurrency and ownership

All mutations of epoch, backlog membership, prepared audio, and selected speed
remain under the publisher lock. The controller belongs to the publisher and
must not be stored in an `HLSListenerLease`.

The transition that creates an encoder also creates the epoch and initializes
the controller atomically. The transition that stops the final encoder clears
the controller atomically. This prevents a second listener joining during start
or shutdown from observing a partially initialized rate state.

## Failure behavior

- A failed synthesis removes that item's debt using the existing failure path
  and records `last_error`; it does not reset the epoch.
- A stale revision is evicted with its duration estimate or prepared PCM so it
  cannot continue inflating backlog.
- If a requested effective Kokoro speed is invalid, fail safely to displayed
  `1.0x`, record the error, and keep the epoch alive.
- Browser buffering never changes the server multiplier.
- Service restart starts with no active epoch and no inherited speech debt.

## Observability

For every synthesized or published item, logs should include:

- Epoch identifier.
- Source order and revision.
- Conservative backlog before selection.
- Displayed multiplier and effective Kokoro speed.
- Whether prepared audio was reused or regenerated for a speed mismatch.
- Synthesis time, exact audio duration, and encoder pending duration.

Epoch-start logs include the number of idle items skipped. Epoch-stop logs include
the final listener-removal reason.

## Acceptance criteria

### Lifecycle

- Publishing multiple translations with no listeners performs no synthesis and
  reports zero active backlog.
- The first listener causes only the newest eligible stable item to be spoken.
- A second listener creates no second encoder, worker, or synthesis call.
- Removing the original first listener while another remains does not reset the
  epoch, backlog, or speed.
- Removing or expiring the final listener clears all epoch-specific state.

### Backlog and speed

- Exact threshold tests select `1.0x` below 10 seconds, `1.2x` at 10 seconds,
  `1.4x` at 30 seconds, and `1.5x` at 40 seconds.
- Inflight, prepared, queued, and encoder-pending audio are counted once each.
- Idle history and already-published HLS media are excluded.
- Each sentence has one fixed synthesis speed.
- A prepared result synthesized at the wrong speed is not released.

### Multi-device playback

- Windows Chrome and iPhone Safari report an effective media-element playback
  rate of `1.0`.
- Both devices hear the same server-generated speed and sentence ordering.
- Adding or removing one listener does not alter another listener's stream.
- A locally buffering iPhone does not cause server speed changes.

### Regression safety

- Existing listener lease, queue-capacity, caption, HLS segment, and final-item
  stabilization tests remain passing.
- Long translated sessions do not reintroduce sentence-start stalls.
- A new listener after a completely idle period begins near the current live
  sentence rather than replaying history.

## Rollback

Keep the existing configured baseline TTS speed as the fallback. The global
controller should be independently disableable so that rollback restores fixed
server-side baseline synthesis and client playback at `1.0`, without restoring
unreliable per-listener iPhone playback-rate acceleration.
