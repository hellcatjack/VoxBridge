# Changelog

All notable public changes to VoxBridge are documented here.

## [Unreleased]

### Fixed

- Corrected the bounded HLS caption timeline to use FFmpeg PCM sample positions,
  AAC presentation delay, synthesized-speech activity bounds, and Safari's
  device-local program date. This removes the roughly two-second early caption
  display and prevents stale device playlists from selecting newer server cues.
- Retained the previous Live Audio sentence between cues without making advisory
  caption polling a dependency of lock-screen audio.
- Prepared exact translated-sentence revisions on the existing single Kokoro
  worker while the backend stability gate is still active, then reused only an
  exact cache hit at ordered HLS release. Revised text invalidates stale audio,
  preparation never publishes early, and the bounded cache is listener-scoped.
- Retained a bounded current-session TTS backlog before the first HLS listener
  joins, then drained it through the existing single Kokoro worker so visible
  stable translations no longer turn into long encoded-silence gaps.
- Counted in-flight Kokoro work in HLS queue diagnostics and logged synthesis
  latency, audio duration, and real-time factor without recording translated text.
- Exposed synthesized PCM backlog as `pending_audio_ms` and added a non-skipping
  listener live-edge guard so slow playback cannot accumulate unbounded delay,
  including while an iPhone page is locked.
- Prevented untranslated Han text from leaking into English subtitles or TTS;
  built-in translators now make one strict target-language retry and reject a
  still-mixed result without blocking later translations.
- Repaired compatible streaming sentence-tail omissions at each segment final
  before source commit, translation, and TTS publication; failed or divergent
  validation no longer causes immediate TTS sealing.
- Prevented the newest unsealed source from reaching TTS on a timer before its
  segment-level canonical ASR validation or rollback-safe successor exists.
- Prevented context hotwords from becoming subtitles after long client silence
  by requiring backend speech evidence before post-silence output is accepted.
- Prevented control-only silent sessions from invoking streaming ASR finish and
  publishing context-biased empty-audio hallucinations.
- Preserved rollback-deferred text across VAD state rotation when the next
  segment correctly begins without textual overlap, eliminating a deterministic
  whole-sentence drop before translation.
- Allowed high-similarity final tail repairs from speech-confirmed segments to
  survive the context silence guard, restoring weak sentence-final syllables.
- Preserved short pauses and sub-120 ms stop tails while suppressing only
  confirmed long-silence transport spans.
- Preserved a bounded 400 ms low-energy endpoint tail and decoded any backend
  pre-roll that had never reached ASR before segment finalization, preventing
  weak final syllables from being discarded at client-silence cuts.
- Prevented Kokoro playback from speaking a translated sentence revision that
  is superseded inside the backend revision-stability window.
- Added newest-source-only TTS revision grace and backend segment sealing so
  late model corrections are protected without a global seven-second delay.
- Prevented mid-speech hard cuts from blocking on full-segment re-decode and
  overflowing the live audio queue.
- Prevented fuzzy candidate-cursor remapping from consuming a newly completed
  terminal sentence after segment resegmentation.
- Rejected segment-final corrections that reduce complete units either inside
  the current segment or after carrying its predecessor's pending prefix,
  preserving stable streaming tails across VAD finalization.
- Rejected canonical candidate updates that move a material suffix of the
  preceding sentence into the current sentence ID, preventing overlapping
  duplicate rows after segment-final resegmentation.
- Prevented a single segment-final one-shot decode from shortening an already
  solidified sentence to its strict normalized prefix, preserving stable final
  words observed throughout the streaming decode.
- Disabled blocking full-session Stop re-decode by default so normal Stop keeps
  visible subtitle history and no longer emits `sentence_reset`.

### Added

- Added a public, English-only PCCS listener that always fits one viewport, plus
  a static QR on the authenticated main page for the fixed church listening URL.
- Added a default 128-lease cap to public HLS bearer capabilities while retaining
  one shared Kokoro worker and one FFmpeg encoder for every listener.
- Replaced per-sentence browser WAV playback on `/listen` with one public
  continuous AAC/HLS live stream, allowing native iPhone lock-screen playback
  and many bounded listeners without per-device Kokoro or FFmpeg work.
- Added listener-scoped HLS listener leases, bounded translated-speech and PCM
  queues, Media Session lock-screen controls, and a user-gesture retry path for
  iOS playback policy failures.
- Added optional `--segment-final-redecode` with structured latency/result
  probes for segment-level canonical ASR validation.
- Added a YouTube `json3` reference-coverage diagnostic for likely whole-cue
  gaps, aligned final-suffix gaps, duplicate commits, and translation ID gaps.
- Added ordered `audio_silence` and `audio_speech_start` WebSocket messages,
  400 ms client pre-roll, source-timeline accounting, and Chromium/WebKit tests.
- Applied a shared, anti-hallucination ESV terminology policy to both
  Chinese-to-English translation backends.
- Added optional CPU-only Kokoro-82M speech for fully stable translated
  sentences, with backend source-order jobs and strict browser FIFO playback.
- Added authenticated in-memory TTS audio jobs, acknowledgement/cancellation,
  TTL cleanup, and default-off browser controls.
- Added a multi-listener `/listen` broadcast page
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
