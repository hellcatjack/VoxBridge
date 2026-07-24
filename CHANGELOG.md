# Changelog

All notable public changes to VoxBridge are documented here.

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
