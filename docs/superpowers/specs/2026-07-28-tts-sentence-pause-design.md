# TTS Sentence Pause Design

**Date:** 2026-07-28

## Goal

Restore a short, natural sentence boundary after single-item lookahead removed
the accidental synthesis gap between translated-speech clips.

## Behavior

- Wait `300ms` after every successfully completed TTS clip before promoting the
  next FIFO item.
- Keep preparing exactly one next WAV during current playback and during the
  pause; the delay gates playback only and does not delay synthesis or transfer.
- Apply the same wall-clock pause to both translation directions and every
  playback-rate setting.
- Do not inspect text, punctuation, language-specific words, or sentence
  contents. Backend stability already defines each published item as a complete
  sentence suitable for speech.
- Start the pause even when no next item is queued yet. A sentence arriving
  during the window is prepared but cannot play until the remaining delay ends.
- If preparation is not complete after `300ms`, the existing readiness wait
  continues naturally.

## Lifecycle

One cancellable timer belongs to the listener session. Stop, disconnect, Start
reset, and page unload clear it immediately. Cancellation rejects with
`AbortError`, so the existing generation guard prevents an old queue pump from
advancing a new session.

The playing indicator turns off when audio ends, before the silent sentence
pause. Queue count and FIFO order do not change.

## Scope

- Modify only standalone listener playback timing.
- Do not change TTS synthesis, prefetch bounds, routes, WebSocket messages,
  translation, ASR, systemd settings, or port `8024`.
- Do not expose a new setting yet; `300ms` is the single documented default.

## Testing

- Static tests require the `300ms` constant and cancellation on reset/unload.
- Playwright controls `ended` explicitly and measures that the next prepared
  clip starts no earlier than the pause threshold.
- Playwright proves Stop cancels an active pause without waiting `300ms`.
- All existing prefetch ordering, speed, mobile, protocol, and full-suite tests
  remain green.
