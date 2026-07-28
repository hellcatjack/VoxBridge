# Per-Device TTS Listener Playback Rate Design

**Date:** 2026-07-28

## Goal

Let each `/listen` device choose its translated-speech playback rate without
changing the producer session, other listeners, Kokoro synthesis, shared WAV
cache entries, or FIFO delivery guarantees.

## Scope

- Add a compact playback-rate control to the standalone `/listen` page.
- Offer `0.8x`, `0.9x`, `1.0x`, `1.1x`, and `1.2x`; default to `1.0x`.
- Treat stored values outside that current allowlist as legacy values and fall
  back to `1.0x`.
- Persist the selected value in `localStorage` on that browser only.
- Apply a new selection immediately to the item currently playing and to every
  queued item that follows.
- Preserve the setting across Stop, reconnect, and page reload.
- Keep the existing backend `--tts-speed` option as the fixed Kokoro synthesis
  baseline. The new control is a listener-side playback preference layered on
  top of the synthesized WAV.

## Non-Goals

- No global meeting-wide speed control.
- No WebSocket or HTTP protocol changes.
- No per-rate TTS synthesis, transformed audio endpoint, or extra server cache.
- No changes to ASR, translation timing, revision stability, or TTS publication.
- No synchronization of preferences between browsers or devices.

## Architecture

The browser remains the owner of its playback queue. One persistent
`HTMLAudioElement` replaces the current one-shot `AudioBufferSourceNode` path so
the active playback rate can be changed while a clip is running. Each fetched
WAV is attached through an object URL, played to completion, then detached and
revoked before the next FIFO item starts.

The media element sets `preservesPitch` and browser-prefixed equivalents when
available. The explicit Start action unlocks playback from a user gesture before
the listener WebSocket begins receiving future jobs. If playback cannot be
unlocked, Start fails locally and does not open a listener subscription.

The server continues synthesizing each stable translation once and broadcasting
the same cached WAV to all assigned listeners. Playback-rate changes therefore
consume no additional Kokoro CPU time or server memory.

## UI And State

The action section gains a labeled select control that remains available before
and during listening. Its displayed value is always the normalized active rate.
The control is keyboard accessible and fits the existing narrow-screen layout
without forcing horizontal scrolling.

On page load:

1. Read `voxbridge.ttsPlaybackRate` from `localStorage`.
2. Accept it only when it exactly matches one of the supported rates.
3. Fall back to `1.0` for missing, malformed, or unsupported values.
4. Apply the normalized value to both `defaultPlaybackRate` and `playbackRate`.

On selection change:

1. Normalize the selected value against the fixed allowlist.
2. Persist the normalized value.
3. Apply it immediately to the persistent media element, including an active
   clip.

Stop and disconnect clear playback and the FIFO exactly as before, but do not
clear the selected speed.

## Playback Lifecycle

- Start creates or reuses the media element and performs a user-gesture unlock.
- `pumpQueue()` still removes exactly one FIFO head at a time.
- The audio fetch and `tts_received` acknowledgement behavior is unchanged.
- The fetched `ArrayBuffer` becomes a WAV `Blob` and a temporary object URL.
- Playback completion resolves from the media element's `ended` event.
- Playback errors reject only the current item and continue to the next one.
- Stop, disconnect, or generation change pauses playback, removes event
  handlers, clears the source, revokes the active object URL, and prevents stale
  completion handlers from advancing the new queue.

## Error Handling

- Storage read/write failures are non-fatal; the in-memory setting still works.
- Invalid persisted values resolve to `1.0x` and are never applied directly.
- A browser that lacks pitch-preservation properties still changes speed using
  standard `playbackRate`; this is a playback-quality degradation, not a start
  failure.
- A rejected media `play()` updates the existing local playback status and skips
  that item without affecting other listeners.

## Security And Privacy

The speed value never leaves the browser. It contains no text, sentence ID,
listener ID, credential, or meeting metadata. Existing authenticated audio fetch
and WebSocket behavior remains unchanged.

## Testing

- Template contract tests cover the allowlist, default, persistence, invalid
  storage fallback, active playback-rate updates, pitch-preservation properties,
  object URL cleanup, and unchanged FIFO semantics.
- Browser tests use Playwright against the authenticated application to confirm
  desktop and mobile layout, persistence after reload, Start/Stop retention, and
  active media-element rate changes without opening another backend.
- The full Python suite must remain green.
- Deployment verification must confirm one user service, one EngineCore, one
  `8024` listener, successful HTTPS authentication routing, and no startup errors.

## Deployment

Fast-forward the verified feature branch into the service working tree. Do not
change the systemd command, Kokoro `--tts-speed`, authentication, model, memory,
translation, VAD, `.venv`, or port. Restart only
`voxbridge-8024.service`, then verify the existing single-process topology.
