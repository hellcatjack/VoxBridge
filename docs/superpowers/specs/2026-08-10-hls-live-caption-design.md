# HLS Live Audio Caption Design

Date: 2026-08-10
Status: approved for implementation

## Goal

Display the translated sentence that each `/listen` device is actually hearing
inside the existing `Live Audio` card. The caption must follow that device's HLS
playhead, including normal buffering, a user-selected playback rate, and the
existing non-skipping catch-up mode. It must not display the server's newest
translation merely because that text has already entered the TTS queue.

The feature must preserve the existing one-screen PCCS visual language, public
listener model, shared CPU synthesis, shared FFmpeg encoder, strict FIFO audio,
and native iPhone lock-screen playback.

## Rejected Approaches

### Show the latest backend translation in `/api/tts/live/status`

This is simple but incorrect. Devices can have different HLS lag and playback
rates, so server-side "now" does not identify what a specific phone is hearing.

### Add a live HLS WebVTT rendition

An in-band subtitle rendition is a standards-based solution, but it requires a
second segmented live playlist, master-playlist wiring, native text-track
compatibility work, and substantially more iOS testing. That complexity is not
justified for one foreground caption card.

## Selected Architecture

Use bounded backend timed cues plus the native audio element's measured distance
from the HLS live edge.

1. Stable ordered TTS release appends one sentence PCM payload to the shared
   encoder exactly as it does today.
2. The append operation yields a scheduled wall-clock span for that payload.
   The cue starts when sentence PCM is scheduled to begin and ends after the
   synthesized speech duration. The fixed 300 ms sentence pause remains in the
   audio payload but is outside the cue's speaking interval.
3. The publisher stores a bounded list of cues for the active HLS epoch. A cue
   contains an opaque cue identity, start and end epoch milliseconds, and the
   exact stable translated text that produced the audio.
4. A listener-scoped public endpoint returns the current playlist live-edge
   wall time and the bounded recent cue list.
5. The browser computes its estimated audio wall time as:

       playlist_live_edge_ms - (seekable_end - current_time) * 1000

6. The browser selects the cue containing that time. If the playhead is between
   cues or in the sentence pause, it keeps the most recently spoken cue and marks
   it inactive rather than clearing the card.

This calculation is per device. It therefore remains correct when one phone is
farther behind, is temporarily catching up, or uses a different playback rate.

## Encoder Timeline Contract

`HLSEncoder.append_pcm()` will return an immutable append receipt with scheduled
start and end epoch milliseconds. The FFmpeg encoder calculates the start from
the amount of unconsumed PCM already ahead of the new payload. Calculation and
pending-byte accounting remain within the encoder so publishers cannot race the
writer loop.

The receipt covers the full payload, including the sentence pause. The publisher
uses the synthesized speech duration to set the cue end and clamps it to the
receipt end. Preparation does not create a cue. A cue is created only when an
already stable item is actually accepted into the HLS PCM timeline.

An encoder implementation used by tests may return deterministic receipts. A
legacy or unavailable encoder receipt must not prevent audio publication; the
publisher records no cue and exposes an empty caption snapshot instead.

## Live Edge Contract

The encoder derives the current live edge from the latest complete HLS segment:
the segment's `EXT-X-PROGRAM-DATE-TIME` plus its `EXTINF` duration. The value is
therefore aligned with the same completed media range represented by the native
audio element's `seekable.end`, rather than with unsegmented PCM still inside
FFmpeg.

Malformed or not-yet-ready playlists return no live-edge value. The listener
keeps its previous caption and retries; this condition must not interrupt audio.

## Public API

Add:

    GET /api/tts/live/{listener_id}/captions

The endpoint requires the same valid listener bearer lease as playlist and
segment routes. It returns `404` for an absent, expired, or foreign lease and
uses `Cache-Control: no-store`.

Example response:

```json
{
  "live_edge_at_ms": 1786420805120,
  "cues": [
    {
      "cue_id": "opaque-hash",
      "start_at_ms": 1786420798200,
      "end_at_ms": 1786420801400,
      "text": "The stable translated sentence being spoken."
    }
  ]
}
```

Caption text is public to the same audience already receiving the public spoken
translation. Raw sentence IDs and TTS job IDs are not exposed. Application logs
record cue counts and timing only, never caption text.

The cue store is bounded to 256 entries and scoped to one HLS epoch. Removing or
expiring the final listener closes the encoder and clears every cue.

## Listener UI

The existing `Live Audio` card remains in the same grid row and keeps the current
cream, sage, forest, coral, mustard, Georgia, and Avenir visual system.

The card contains:

- `LIVE AUDIO` eyebrow.
- A multi-line primary translated caption in Georgia.
- A smaller Avenir playback state such as `Listening live`, `Buffering`, or
  `Paused`.
- The existing animated audio mark.

Text wraps naturally. JavaScript does not split, punctuate, abbreviate, or style
content based on language-specific words. The flexible card absorbs normal
sentence lengths without introducing a document scrollbar. Narrow and short
viewports retain the existing single-screen layout.

The caption changes only when the playhead enters a different cue. Replacement
uses a restrained opacity/vertical reveal and honors `prefers-reduced-motion`.
The DOM is never cleared between sentences. After a cue ends, the previous text
remains visible at reduced emphasis until the next cue starts. Stop resets the
card to `Waiting to start`.

Polling runs only while listening and the page is visible. The foreground interval
is 500 ms because HLS segments are one second. When an
iPhone unlocks, an immediate poll realigns the caption. Audio and lock-screen
playback never depend on polling success.

## Failure Handling

- Caption endpoint failure leaves the previous caption visible and does not
  change audio state.
- Missing live-edge data leaves the previous caption visible.
- No cue before the playhead displays `Waiting for translated speech`.
- A stale response for a previous listener ID is ignored.
- Stop cancels caption polling before deleting the listener lease.
- Cue overflow evicts only the oldest cue; audio is never dropped.

## Testing

Backend tests will verify:

- FIFO append receipts produce non-overlapping cue spans.
- Cue end excludes the fixed sentence pause.
- Preparation does not publish a cue.
- Cue storage is bounded and cleared with the HLS epoch.
- Caption API validates the listener lease and returns no-store responses.
- Playlist live-edge parsing uses the final complete segment.

Browser tests will verify:

- The selected caption follows `seekable.end - currentTime`, not the newest cue.
- Different simulated device lags select different captions from one shared feed.
- Sentence pauses retain and de-emphasize the previous caption.
- A new cue replaces text without an intermediate empty value.
- Stop resets caption and state.
- Desktop, narrow mobile, and short landscape layouts remain one screen without
  document scrollbars.

## Non-Goals

- Captions on the iOS lock screen or in native media controls.
- Word-level highlighting.
- Karaoke timing or token timestamps.
- Per-device audio synthesis or per-device backend queues.
- Replacing the shared HLS stream with sentence WAV playback.
