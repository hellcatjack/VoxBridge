# TTS Single-Lookahead Prefetch Design

**Date:** 2026-07-28

## Goal

Reduce the pause between queued translated-speech items by preparing the next
FIFO item's shared WAV while the current item is playing, without allowing
unbounded browser memory, concurrent playback, or duplicate backend synthesis.

## Root Cause

The listener currently fetches an item's audio only after it removes that item
from the FIFO. The next item is not removed until the current media element emits
`ended`. Because the broadcast audio endpoint performs lazy Kokoro synthesis on
the first fetch, every transition can include synthesis, HTTP transfer, and WAV
loading after the previous audio has already stopped.

## Architecture

Keep the existing FIFO and persistent `HTMLAudioElement`. Add a browser-local
preparation registry keyed by opaque job ID. A preparation owns one
`AbortController`, one settled promise, and either WAV bytes or a normalized
error. The current item may have a preparation, and only the first queued future
item may be prepared concurrently. No later queue item is fetched until it moves
into that one-item lookahead position.

When a job arrives during playback, the listener starts preparation for
`queue[0]`. When `pumpQueue()` promotes an item to current, it reuses that promise
instead of issuing another request, then immediately starts preparation for the
new `queue[0]`. The backend synthesis lock remains authoritative, so concurrent
browser requests cannot cause concurrent Kokoro inference. Existing shared WAV
caching prevents another listener from synthesizing the same job again.

## Ordering And Acknowledgement

- Playback remains strictly sequential and starts only from `queue.shift()`.
- At most one future queued item is fetched ahead of current playback.
- `tts_received` remains tied to complete WAV receipt, so a prefetched item may
  be acknowledged before it plays. This is safe because the browser then owns
  the complete bytes and Stop is already defined to discard local queued audio.
- A prepared item is consumed exactly once and removed from the registry when it
  finishes, fails, or is discarded.
- A preparation failure is stored rather than left as a rejected unobserved
  promise. When the item reaches the head, it follows the existing skip-and-
  continue behavior.

## Lifecycle And Bounds

- Normal state holds at most the current item's preparation plus one future
  preparation. Only the future preparation can retain a complete unplayed WAV.
- Stop, socket close, start reset, and page unload abort all in-flight fetches,
  clear prepared bytes, and then clear the FIFO.
- Generation checks continue preventing stale completion handlers from advancing
  a new listener session.
- Queue status continues reporting untranslated-speech jobs waiting after the
  current item; prefetch does not alter its count.

## Scope

- Modify only the standalone listener's queue/fetch orchestration.
- Do not change Kokoro, broadcast routes, WebSocket messages, authentication,
  systemd arguments, ASR, translation, revision stability, or port `8024`.
- Do not introduce a second media element or promise sample-accurate gapless
  playback. One-item lookahead removes synthesis and transfer from the normal
  transition when preparation finishes in time; browser media startup may leave
  a much smaller residual gap.

## Testing

- Static contract tests require a one-item preparation registry, reuse, bounded
  lookahead, and reset cancellation.
- Playwright injects a fake listener WebSocket and fetch implementation, sends
  three jobs, and proves the second is fetched during first playback while the
  third is withheld until the second becomes current.
- Playwright also proves a prefetched WAV is reused without a duplicate fetch and
  Stop aborts all active preparation.
- Existing FIFO, authentication, broadcast-cache, speed, mobile, and full-suite
  tests must remain green.

## Deployment

Fast-forward only after full worktree and main-tree verification. Restart only
the existing user `voxbridge-8024.service`, wait for one listener on `8024`, and
verify one backend, one EngineCore, HTTPS authentication routing, and clean
startup logs. Do not change the service command.
