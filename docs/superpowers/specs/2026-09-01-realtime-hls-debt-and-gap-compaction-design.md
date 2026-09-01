# Real-time HLS Debt and Waiting-gap Compaction Design

## Status

Approved design direction; implementation pending.

## Problem

The shared listener now generates one server-paced Kokoro stream and keeps every
browser media element at `1.0x`. Production evidence exposed two remaining
problems:

1. A translation that arrives after the HLS writer has flushed the preceding
   sentence can retain an extra encoded-silence gap. The listener may therefore
   hear more silence than the natural gap between continuously available
   sentences, even after the new sentence is safely available.
2. The HLS writer currently publishes active PCM at `2.0x` continuously while
   clients consume it at `1.0x`. Published PCM immediately leaves the server's
   unpublished-backlog calculation, even though clients have not played it.
   The Auto controller therefore sees too little debt and rarely selects the
   higher speed tiers.

In the observed production epoch, 89 sentences were published: 58 at displayed
`1.0x`, 30 at `1.2x`, and none at `1.4x` or `1.5x`. The largest decision backlog
was only 14.936 seconds because the continuous `2.0x` writer moved debt into HLS
twice as fast as the fixed-rate clients could consume it.

## Goals

- Keep enough HLS headroom to avoid sentence-start starvation without moving an
  unbounded amount of speech ahead of real-time playback.
- Keep unpublished speech visible to the existing global Auto thresholds.
- Preserve the speed chosen when a sentence is synthesized.
- Remove only wait-generated, disposable carrier silence once the next sentence
  is safely buffered.
- Preserve the exact natural PCM gap that would exist if both sentences had
  been continuously available.
- Never reintroduce the pause/play/retry loop that previously caused persistent
  `Buffering live audio` and sentence-start stalls.
- Keep every listener on the same HLS epoch and at HTML media rate `1.0x`.

## Non-goals

- Per-listener playback-rate control.
- Letting the slowest listener control global synthesis speed.
- Seeking over translated speech or the configured sentence pause.
- Replacing HLS, FFmpeg, or Kokoro.
- Replaying translations from before the active listener epoch.
- Making local network lag part of the shared server backlog.

## Root cause 1: unbounded accelerated HLS publication

`FFmpegHLSEncoder._writer_loop()` sleeps for `frame_duration / 2.0` for every
active PCM frame. This was introduced while clients could play as fast as
`1.5x`, so the producer had to stay ahead of the fastest consumer. Clients are
now intentionally fixed at `1.0x`, but the encoder still advances the HLS media
timeline at `2.0x` for the full speech backlog.

`pending_audio_ms` decreases when the writer submits PCM to FFmpeg. Consequently,
the global controller stops counting audio that was published at `2.0x`, even
though listeners can consume it only at `1.0x`. This is why visually obvious
speech debt does not reach the 30- or 40-second Auto tiers.

## Bounded HLS startup burst

Active PCM will use a bounded startup burst instead of a permanent `2.0x`
publication rate:

- Burst rate: `2.0x`.
- Burst media budget: 2.0 seconds of PCM.
- Sustained rate after the budget: `1.0x`.
- Reset the burst budget only when the writer has genuinely waited for a new
  PCM item after an empty queue.
- Do not reset the budget between adjacent queued sentences.
- Continue allowing the short tail-finalization carrier to be written at
  `2.0x`; it is bounded by the existing HLS bootstrap requirement.

This gives a newly arriving sentence roughly one second of HLS lead, enough to
finalize a segment quickly, while a long continuous backlog remains in
`pending_audio_ms` and drains at the same sustained rate as clients.

The burst is encoder transport behavior. It does not change the audible speed
or the Kokoro Auto multiplier.

## Root cause 2: provisional tail carrier becomes permanent gap

When the active PCM queue becomes empty, the encoder writes a bounded carrier
tail so FFmpeg can finalize and expose the last speech segment. If the next
translation arrives later, its scheduled start follows the carrier bytes that
were already submitted. The extra carrier is operational padding, not the
natural inter-sentence pause, but the existing caption contract does not tell a
listener which portion is disposable.

The earlier browser compactor inferred a fixed gap, changed `currentTime`, then
entered a custom pause/play/retry state machine. A seek that did not immediately
return to `playing` could repeatedly report buffering and retry, causing the
sentence-start stalls it was meant to prevent. That implementation must not be
restored.

## Exact disposable-gap metadata

The encoder already has the information needed to distinguish real PCM from
provisional carrier:

- `_scheduled_end_pcm_bytes` is where the next clip would begin if it followed
  the previous clip continuously.
- `_submitted_pcm_bytes` includes carrier already written while waiting.

Before scheduling a new PCM item, calculate:

`discardable_gap_bytes = max(0, submitted_pcm_bytes - scheduled_end_pcm_bytes)`

Convert that value to `discardable_gap_before_ms` and return it in the
`HLSAppendReceipt`. The value is zero for continuously queued clips.

When the publisher creates the next caption cue, it knows the previous cue end,
the next cue start, and the encoder's exact disposable duration. It exposes:

- `discardable_gap_before_ms`: the carrier duration that may be skipped.
- `resume_at_ms`: the earliest absolute program time that preserves the natural
  gap before the new cue.

Let `actual_gap_ms = next.start_at_ms - previous.end_at_ms`. Then:

`natural_gap_ms = max(0, actual_gap_ms - discardable_gap_before_ms)`

`resume_at_ms = next.start_at_ms - natural_gap_ms`

The natural gap includes model edge silence, the configured `300ms` sentence
pause, encoder frame alignment, and any other timing that also exists between
continuously queued clips. Only wait-generated carrier is marked disposable.

The first cue has no previous speech context, so it has no compaction marker.
The fields are additive and backward compatible for API consumers.

## One-shot listener compaction

Caption polling remains advisory and provides the next cue and its absolute
program-time marker. A listener may compact only when all of these conditions
hold:

1. Playback is running and currently in the gap after the previous cue.
2. The next cue has a positive disposable gap and a valid `resume_at_ms`.
3. At least 500ms of carrier is disposable; smaller differences are left alone
   to avoid a risky seek for negligible benefit.
4. The computed media target and at least 1.0 second beyond the next speech
   start are already inside one buffered media range.
5. No seek for that exact cue has previously been attempted by this listener.

The listener accounts for silence already heard. If `heard_gap_ms` is the time
since the previous cue ended, it keeps only the remaining natural gap:

`remaining_natural_ms = max(0, natural_gap_ms - heard_gap_ms)`

`target_program_ms = next.start_at_ms - remaining_natural_ms`

The absolute target is mapped to media time using the existing Safari
`getStartDate() + currentTime` mapping or hls.js `playingDate` mapping.

The listener then performs one `fastSeek(target)` when supported, otherwise one
`currentTime = target` assignment. It must not pause, call `play()`, change the
media playback rate, install a recovery timeout, or retry the same cue. Native
HLS/hls.js recovery remains responsible for ordinary media events. This makes a
failed compaction degrade to extra silence rather than a persistent audio stall.

Every listener receives the same absolute marker. Devices at different network
positions may perform the one-shot operation at different wall-clock moments,
but all converge on the same natural sentence boundary without changing the
shared synthesis speed.

## Prepared-audio speed ownership

The speed of a prepared sentence is selected when its Kokoro synthesis begins.
The prepared PCM already records `displayed_multiplier` and `effective_speed`.
Stable release will reuse those recorded values instead of recalculating the
tier and discarding valid PCM merely because encoder debt drained between
preparation and release.

For a stable item without prepared PCM, release still selects speed from the
current conservative backlog. Revisions continue to invalidate stale prepared
audio. The configured fixed-mode rollback continues to synthesize at the
baseline speed.

On prepared release, public global-speed status is updated to the prepared
sentence's recorded multiplier. A sentence is never retimed after synthesis.

## Backlog and Auto behavior

The existing Auto boundaries remain unchanged:

| Conservative unpublished speech | Displayed multiplier |
| ---: | ---: |
| `< 10s` | `1.0x` |
| `10-<30s` | `1.2x` |
| `30-<40s` | `1.4x` |
| `>=40s` | `1.5x` |

No client playhead is added to this shared value. The bounded writer change
keeps long debt in the unpublished PCM queue long enough for these thresholds to
measure it honestly. The 2-second transport burst is deliberately small and is
not added back as synthetic debt.

## Failure behavior

- Missing or invalid gap metadata means no compaction.
- Insufficient buffered media means no compaction yet; a later caption poll may
  make the single attempt after the buffer guard is satisfied.
- Once an attempt is made, that cue is never retried during the listener epoch.
- A seek exception leaves normal sequential HLS playback active.
- Browser buffering never changes the server Auto multiplier.
- Encoder or synthesis failures retain the existing `last_error` behavior.
- Final listener removal clears caption markers, prepared PCM, burst state, and
  the speech epoch as before.

## Observability

The existing synthesis logs retain epoch, backlog, multiplier, effective speed,
and cache-hit fields. Add enough information to distinguish:

- a prepared-speed reuse from a new release-time decision;
- encoder `discardable_gap_before_ms` for each published clip;
- listener compaction attempts in browser test instrumentation (not server logs,
  because the public page does not send client telemetry).

## Testing

### Encoder and controller

- A long PCM item writes only its first 2.0 seconds at `2.0x`; later frames use
  `1.0x` pacing.
- Adjacent queued items share one burst budget.
- A genuine empty-queue wait resets the burst budget.
- Tail finalization remains bounded and decodable.
- `pending_audio_ms` remains after the burst for a long backlog.
- Exact 10-, 30-, and 40-second tiers remain unchanged.
- Prepared PCM synthesized at a faster tier is reused at stable release and its
  recorded speed becomes the shared status.

### Gap metadata

- Continuous items report zero disposable gap.
- Carrier written while waiting is reported exactly once on the next receipt.
- Caption `resume_at_ms` removes only the reported carrier and preserves the
  computed natural gap.

### Browser

- A historical wait gap seeks once to the buffered safe target.
- Silence already heard reduces the remaining natural gap.
- No seek occurs without the 1.0-second post-start buffer guard.
- No seek occurs for a continuously queued sentence or a disposable gap below
  500ms.
- The compactor never calls `pause()` or `play()` and never installs a retry.
- `waiting`, `stalled`, and `playing` events keep media playback rate at `1.0`.
- iPhone-native HLS and desktop hls.js timestamp mappings target the same
  absolute marker.

### Regression and deployment

- Run the complete Kokoro, HLS, API, listener Playwright, and full repository
  test suites.
- Restart only `voxbridge-8024.service`.
- Verify two temporary listeners share one speech epoch and cleanly release it.
- During a controlled long-backlog run, verify logs enter `1.4x` and `1.5x`
  when unpublished PCM crosses 30 and 40 seconds.

## Rollback

The change is isolated to encoder pacing, prepared-speed reuse, additive caption
metadata, and the one-shot listener compactor. Reverting the implementation
commit restores the current behavior. `--disable-tts-global-auto-speed` remains
available to keep Kokoro at the fixed baseline independently of HLS pacing.
