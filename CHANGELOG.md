# Changelog

All notable public changes to VoxBridge are documented here.

## [Unreleased]

### Fixed

- Prevented Kokoro playback from speaking a translated sentence revision that
  is superseded inside the backend revision-stability window.
- Added newest-source-only TTS revision grace and backend segment sealing so
  late model corrections are protected without a global seven-second delay.

### Added

- Applied a shared, anti-hallucination ESV terminology policy to both
  Chinese-to-English translation backends.
- Added optional CPU-only Kokoro-82M speech for fully stable translated
  sentences, with backend source-order jobs and strict browser FIFO playback.
- Added authenticated in-memory TTS audio jobs, acknowledgement/cancellation,
  TTL cleanup, and default-off browser controls.
- Added an authenticated, future-only, multi-listener `/listen` broadcast page
  with per-device FIFO playback and one shared Kokoro synthesis per translation.
- Decoupled translated-speech playback from the main subtitle page; listener
  Start and Stop now affect only that device.
- Disabled raw Uvicorn access logging so opaque TTS job identifiers are not
  persisted in request paths; broadcast diagnostics use short hashes instead.
- Added a bounded vLLM multimodal processor cache option and the user-level
  `voxbridge-logrotate.timer` deployment templates for long-running sessions.

## [0.2.0] - 2026-07-23

### Added

- Backend-provided sentence stability metadata, sentence revisions, and revision-safe translation updates.
- Per-session professional-term context and bounded time-window context schedules.
- Optional single-user login with server-side sessions and secure-cookie deployment support.
- Structured subtitle/text-pool trace events for diagnosing generating and solidified text.

### Changed

- Kept segmentation, final flush, overlap handling, and sentence commitment in the backend.
- Added bounded audio backpressure and state rotation for long-running streams.
- Preserved stable short sentence-tail extensions without accepting transient model rewrites.
- Kept the previous translation visible while a newer source revision is translated.
- Made translation direction, subtitle history, manual scrolling, font sizing, and system-audio capture explicit UI behavior.

### Security

- Disabled local debug-file access in the recommended public deployment.
- Kept credentials, meeting logs, audio, and subtitle traces outside version control.

## [0.1.0]

- Initial standalone VoxBridge repository.
