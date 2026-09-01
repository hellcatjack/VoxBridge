# VoxBridge Backend API

This document describes the current backend API exposed by
`voxbridge.cli.demo_streaming_ws`. Local development and testing use port `8024`.

## HTTP Endpoints

Authentication is disabled by default for local development. When the server is
started with `--auth-enabled`, browser access uses an `HttpOnly` session cookie.
Generate a password hash with:

```bash
../.venv/bin/python - <<'PY'
from voxbridge.cli.demo_streaming_ws import _hash_auth_password
print(_hash_auth_password("replace-with-a-strong-password"))
PY
```

Use `--auth-password-hash <hash>` or `VOXBRIDGE_AUTH_PASSWORD_HASH=<hash>`.
For HTTPS/WSS public deployments, add `--auth-cookie-secure`. For public
deployments, also prefer `--disable-debug-file`. Authentication still protects
the main subtitle UI, ASR WebSocket, and management/compatibility routes. The
explicit public listener routes documented below are the only exception.

### `GET /`

Returns the browser UI as HTML.

When authentication is enabled, unauthenticated requests redirect to `/login`.

### `GET /listen`

Returns the standalone translated-speech listener. The main subtitle page does
not synthesize or play audio. This route is public even when authentication is
enabled, so a phone can enter directly from the fixed QR code. Each browser must
explicitly select Start to activate audio.

The English-only, one-screen page is branded for Pittsburgh Christian Church South
and assigns one persistent media element a listener-scoped HLS URL. Safari uses
native HLS, preserving iPhone lock-screen playback without a foreground JavaScript
queue. Desktop Chrome, Edge, and Firefox use the locally served hls.js MSE fallback
when MSE AAC is available, even if desktop Chrome reports unreliable native HLS
support. Both paths consume the same shared stream and call
`play()` from the Start gesture; neither fetches sentence WAVs nor creates another
synthesis or encoder. The continuous stream carries a near-silent decodable
carrier at real-time `1.0x` when no translated speech is ready. This keeps the
live playlist advancing through long speech gaps so native Safari can resume
without a new play gesture.

Playback Speed is a read-only shared status. Every browser keeps the HTML media
element at `defaultPlaybackRate = 1.0` and `playbackRate = 1.0`; hls.js is also
configured with `maxLiveSyncPlaybackRate: 1`. The publisher selects the audible
Auto multiplier for each sentence and passes the resulting absolute speed to
Kokoro before synthesis. All listeners therefore receive the same already-paced
PCM instead of relying on browser-specific live-HLS rate control. A device's
buffering state, HLS playhead, visibility, and join time do not change the
global multiplier.

### `GET /listen/assets/hls.min.js`

Returns the pinned hls.js browser build used only when native HLS is unavailable.
The asset is public, locally hosted, served as JavaScript with `nosniff`, and
cacheable for one day. Its pinned version, upstream release, SHA-256 digest, and
Apache-2.0 license are stored with the vendored asset.

The listener verifies both hls.js MSE support and the `mp4a.40.2` AAC codec before
selecting this path. The shared encoder continuously emits a near-silent decodable
carrier at real-time `1.0x` while idle because exact zero PCM can produce MPEG-TS
segments with no AAC frames. New speech is checked once per encoder frame
(default `100ms`), and queued adjacent clips are selected before carrier, so this
keepalive does not add an inter-sentence pause or speech backlog.

### `GET /listen/qr.svg`

Returns a cacheable, script-free static QR code for
`https://ushome.amycat.com:18024/listen`. The SVG is generated at build time and
does not contact an external QR service. This route is public.

### `GET /api/tts/live/status`

Returns public foreground diagnostics for the shared stream:

```json
{
  "available": true,
  "listener_count": 2,
  "queue_depth": 0,
  "synthesis_active": false,
  "preparation_queue_depth": 0,
  "preparation_active": false,
  "prepared_audio_count": 1,
  "pending_audio_ms": 0,
  "translated_audio_backlog_ms": 82000,
  "translated_audio_backlog_count": 9,
  "translated_audio_backlog_estimated": true,
  "speech_epoch_id": "b295d74643d84c8d9ed0a25104266f3a",
  "global_speed_mode": "auto",
  "global_speed_multiplier": 1.5,
  "tts_effective_speed": 1.575,
  "encoder_active": true,
  "producer_active": true,
  "last_error": ""
}
```

The status request does not create a listener lease. It is advisory only; native
HLS playback continues if browser JavaScript is suspended while the device is
locked. `queue_depth` counts queued stable translations plus the one item already
removed from the queue for Kokoro inference. `synthesis_active` makes that
in-flight state explicit. `preparation_queue_depth` counts translated revisions
waiting for speculative Kokoro work, `preparation_active` identifies that kind
of in-flight work, and `prepared_audio_count` counts exact revisions cached but
not yet permitted into the live stream. `pending_audio_ms` measures synthesized PCM, including
the inter-sentence pause, that has not yet been written into the real-time HLS
timeline. `translated_audio_backlog_ms` adds every unique successful
translation known to the shared publisher but not yet submitted to that HLS
timeline. Prepared revisions contribute their exact PCM duration; revisions
still waiting for Kokoro contribute a rolling per-language duration estimate.
`translated_audio_backlog_count` is the number of those unique revisions, so a
revision present in both preparation and stable-release queues is counted only
once. `translated_audio_backlog_estimated` is true while any contribution still
uses an estimate. An unsynthesized item uses the larger of the language default
or the recently observed baseline speech duration per character with a 10%
safety margin, plus the inter-sentence pause once. These fields describe shared
server-side unpublished speech only; they exclude every device's HLS lag,
network delay, and audio already published into the live window.

`speech_epoch_id` identifies the current shared TTS/HLS lifetime.
`global_speed_mode` is `auto` by default or `fixed` when the rollback switch is
enabled. `global_speed_multiplier` is the displayed shared multiplier selected
for the next/current sentence, and `tts_effective_speed` is the absolute Kokoro
value after applying the configured `--tts-speed` baseline. With no listeners,
there is no active speech epoch: `speech_epoch_id` is empty, all translated
backlog fields are zero, `global_speed_multiplier` is `1.0`, and
`tts_effective_speed` equals the configured baseline. Polling this endpoint does
not start synthesis or create an epoch.

### `GET /api/tts/live/{listener_id}/index.m3u8`

Creates or refreshes a listener lease and returns the shared live
playlist as `application/vnd.apple.mpegurl`. The route waits up to five seconds
for the initial segment and returns `503` if FFmpeg cannot establish the stream.
The response uses `Cache-Control: no-store`. Segment entries are rewritten to
the same listener-scoped URL namespace.

Listener IDs must satisfy the same safe opaque-token format used by other TTS
routes. Because this endpoint is public, an unguessable listener ID is a public
bearer capability for that short-lived lease, not an authenticated user identity.
The server accepts at most 128 concurrent leases by default; a new listener over
that bound receives `HTTP 429`, while an existing lease may still refresh.

The publisher also keeps a bounded pre-listener pool of up to 128 stable
translations from the current producer session. It does not synthesize or count
them as active speech backlog while there are no listeners. When the first
listener creates a new live epoch, the publisher discards stale entries, retains
only the latest stable translation, and synthesizes that join sentence at the
displayed `1.0x` baseline. Future translations continue in source order, so a
late join or restart begins near the live edge instead of replaying the meeting
from the beginning. When idle, overflow retains the most recent entries;
starting a new producer session clears an older idle pool so speech cannot cross
meeting boundaries. The first listener is only the epoch trigger: adding another
listener or removing the original one leaves the shared encoder, backlog, and
speed unchanged while at least one lease remains. Removing the final lease clears
the epoch and its controller state.

While at least one listener lease is active, translation completion may start
Kokoro preparation before the source-revision stability gate releases the item.
The cache key is the exact translation revision: sentence ID, revision, target
language, and a SHA-256 text digest. A newer revision invalidates stale prepared
audio. Preparation never writes PCM to HLS; only the existing ordered stability
release can publish it. Stable release reuses an exact cache hit, including work
already in flight, so it does not create a second Kokoro synthesis. The cache is
bounded to eight entries and is cleared with the listener epoch. Prepared audio
records its displayed multiplier and effective Kokoro speed. Stable release
keeps the speed selected when that exact revision began synthesis and reuses its
PCM even if the unpublished backlog has since crossed into a lower Auto tier.

### `GET /api/tts/live/{listener_id}/captions`

Returns the recent translated caption cues for the matching live HLS lease:

```json
{
  "live_edge_at_ms": 1786400000123,
  "cues": [
    {
      "cue_id": "a3f85d7f59ee7f29",
      "start_at_ms": 1786399995200,
      "end_at_ms": 1786399997880,
      "text": "The sentence currently being spoken.",
      "discardable_gap_before_ms": 0,
      "resume_at_ms": null
    }
  ]
}
```

`live_edge_at_ms` is the wall-clock end of the newest complete HLS segment.
Each cue is derived from the PCM media timeline actually submitted to FFmpeg,
not from the wall clock when a sentence entered the queue. The mapping includes
the MPEG-TS AAC `1024-sample` encoder frame and language-independent waveform
edge silence detection. A cue therefore starts at synthesized speech activity
and ends after its final activity; `end_at_ms` excludes trailing model silence
and the fixed `300ms` inter-sentence pause. The server retains at most 256
caption cues for the current encoder epoch and clears them when that shared
stream ends.

`discardable_gap_before_ms` is the clamped part of the preceding cue gap caused
only by continuous idle carrier PCM submitted while waiting for another
translation. It never includes the normal sentence pause or model edge silence.
`resume_at_ms` is `null` when no such carrier exists; otherwise it is the absolute
program-time point after the disposable carrier from which only the still-unheard
part of the natural gap must be preserved. If the actual wait already covered the
natural gap, the browser targets the next speech start and adds no extra pause.

On Safari, the listener maps the native media timeline to an absolute playhead
using `getStartDate() + currentTime` and selects the newest cue whose start is
not later than that device-local position. This remains correct when the
device's loaded playlist is several segments behind the server. The
server-live-edge estimate is only a compatibility fallback
when the media element cannot provide a valid start date. The page therefore
displays what that device is hearing instead of the newest server translation.
Between cues it keeps the previous sentence visible without clearing, dimming,
or flashing the text. When both `resume_at_ms` and the next second of speech are
inside one media buffer range, the page performs an immediately buffered seek.
Otherwise it seeks as soon as the target and `100ms` beyond the next speech start
are inside one media `seekable` range, allowing native HLS or hls.js to load the
target fragment instead of consuming accumulated idle carrier. Each cue is
attempted at most once. The page never changes the fixed `1.0` media rate and has
no custom pause/play/retry recovery loop. The response uses `Cache-Control:
no-store`. Caption
polling is advisory and does not gate HLS audio; lock-screen playback continues
if polling is suspended or temporarily fails.

The endpoint requires an existing matching public bearer capability and returns
`404` for an unknown or expired listener. It does not create or refresh a lease.

### `GET /api/tts/live/{listener_id}/segments/{segment_name}`

Refreshes the matching lease and returns one shared MPEG-TS/AAC segment. Segment
names are restricted to `segment_#########.ts`; traversal and unknown leases
return `404`. Different listeners read the same file and do not cause additional
Kokoro or FFmpeg work. Possession of the matching listener ID is required.

### `DELETE /api/tts/live/{listener_id}`

Removes only the matching public bearer capability's listener lease. The shared
encoder remains active while any other lease exists. The last listener removal,
or expiry after 90 seconds without playlist/segment traffic, closes FFmpeg and
removes the temporary HLS epoch directory.

### `GET /login`

Returns the login page when authentication is enabled.

### `POST /login`

Accepts `application/x-www-form-urlencoded` fields:

- `username`
- `password`

On success, sets the `voxbridge_session` cookie and redirects to `/`.

### `POST /logout`

Deletes the current server-side session and clears the browser cookie.

### `GET /__debug/file?path=<path>`

Returns a local debug file from the configured debug roots.

Use this endpoint only for local diagnostics. Do not expose it on an untrusted
network. When authentication is enabled, this endpoint requires a valid session.
When `--disable-debug-file` is set, this endpoint always returns `404`.

### Legacy `POST /api/tts/broadcast/jobs/{job_id}/audio`

Returns the shared `audio/wav` for one job assigned to the authenticated
listener in `X-TTS-Listener-ID`. Missing, expired, unassigned, and foreign jobs
all return `404`; unavailable or failed synthesis returns `503`. The response is
`Cache-Control: no-store` and includes `X-TTS-Sample-Rate` and
`X-TTS-Duration-Ms`.

Kokoro synthesis is globally serialized on CPU and cached once per stable
translation. Multiple assigned listeners fetch the same cached WAV. A job is
released only after all intended listeners acknowledge receipt or disconnect,
and no audio request still holds a lease.

### Deprecated private TTS endpoints

The following owner/client-scoped routes remain for one compatibility cycle.
New clients must use `/listen`, `WS /ws/tts`, and the broadcast audio route.

#### `POST /api/tts/jobs/{job_id}/audio`

Returns one backend-issued TTS job as `audio/wav`. With `--auth-enabled`, the
route requires the same authenticated cookie session that owns the job, uses
`Cache-Control: no-store`, and returns `404` for absent, expired, or foreign jobs.
Authentication-disabled mode is intended only for trusted local development and
uses anonymous ownership; public TTS deployments must enable global
authentication. Synthesis runs one job at a time on the CPU; a generated WAV is
cached in memory only until acknowledgement.

#### `DELETE /api/tts/jobs/{job_id}`

Acknowledges playback preparation and removes the job plus cached WAV from
memory. The translated text never appears in the URL.

#### `DELETE /api/tts/clients/{client_id}/jobs`

Cancels every unread job owned by the current session and page client. The
session is authenticated when `--auth-enabled` is in use and anonymous only in
trusted local mode. The browser calls this when the user disables translated
speech.

## WebSocket Endpoints

### Legacy `WS /ws/tts`

Compatibility translated-speech listener protocol. The current `/listen` page
uses shared HLS instead. Listener sockets
do not count against the ASR `--max-connections` limit. After connection the
server returns:

```json
{
  "type": "tts_listener_ready",
  "listener_id": "opaque-random-token",
  "tts_available": true,
  "producer_active": true
}
```

The listener is future-only: it receives no job history and only participates in
stable translation jobs published after registration. Each job event contains
metadata only, never translated text:

```json
{
  "type": "tts_job",
  "job_id": "opaque-random-token",
  "sentence_id": "1781901841676-386666-1",
  "revision": 2,
  "source_order": 7,
  "target_language": "Chinese",
  "is_stable": true
}
```

After the browser has fetched the complete WAV, it acknowledges receipt:

```json
{
  "type": "tts_received",
  "job_id": "opaque-random-token"
}
```

`producer_status` reports whether the main ASR session is active. An inactive
event is sent after the last stable job at graceful stop and does not clear the
device FIFO. `ping` receives `pong`. Start and Stop are local UI actions: Stop
closes only that listener socket, aborts its current fetch/playback, and clears
only its local queue. Queue overflow disconnects only the slow listener.

### `WS /ws`

The main streaming ASR and subtitle protocol.

When authentication is enabled, the WebSocket handshake must include the
`voxbridge_session` cookie. Unauthenticated connections receive:

```json
{
  "type": "error",
  "message": "unauthorized"
}
```

The backend then closes the socket with policy violation code `1008`.

Audio frames are sent as binary WebSocket messages:

- Format: raw PCM signed 16-bit little-endian.
- Sample rate: `16000`.
- Channels: mono.
- Maximum frame size: controlled by `--max-frame-samples`.

Control messages and backend events are UTF-8 JSON text messages.

## Client Messages

### `start`

Starts or restarts one streaming session on the current WebSocket connection.
The backend resets ASR state, subtitle state, text pool state, pending
translations, alignment counters, and audio queues.

```json
{
  "type": "start",
  "language": "English",
  "translation_direction": "en2zh",
  "asr_context_terms": ["Elisha", "Jordan"]
}
```

Fields:

- `language`: optional ASR force language. Supported values are normalized by
  the backend, typically `Chinese` or `English`.
- `translation_direction`: optional translation direction. Supported values:
  `zh2en` and `en2zh`. Unknown values fall back to `zh2en`.
- `asr_context_terms`: optional array of short glossary terms. Field presence is
  significant: a non-empty array overrides the configured context schedule, an
  empty array explicitly disables context, and an omitted field preserves the
  configured schedule for legacy clients.
- `tts_enabled` and `tts_client_id`: deprecated private-TTS compatibility fields.
  The main VoxBridge page no longer sends them. New listener devices use
  `/listen`; stable translation broadcast does not depend on these fields.

`start` validates and constructs the replacement ASR state before resetting the
active session. Invalid context therefore returns `error` without replacing the
current state or beginning a new audio stream.

### `audio_silence`

Compresses confirmed client-side transport silence without sending silent PCM:

```json
{
  "type": "audio_silence",
  "duration_ms": 1000,
  "capture_sample_index": 1920000
}
```

- `duration_ms` is the confirmed quiet interval represented for endpoint policy
  and must be an integer from `1` through `5000`. The bundled client can also
  retain a bounded prefix of that interval as endpoint-tail PCM for ASR
  accuracy; `capture_sample_index` keeps source time monotonic.
- `capture_sample_index` is the client's monotonic 16 kHz source-sample cursor
  and must be a non-negative safe JavaScript integer.
- The event is queued in order with preceding PCM. It advances backend silence
  state, endpoint policy, idle-tail handling, and the source timeline, but it is
  never converted to PCM or passed to ASR inference.
- If a streaming state receives only `audio_silence` controls and no audio was
  decoded, Stop returns an empty stable `final` without calling
  `finish_streaming_transcribe`. This prevents context-biased empty-audio output
  from entering subtitles, translation, or TTS.
- The bundled browser delays the first event until 700 ms of certain silence,
  retains a bounded 400 ms prefix as low-energy endpoint-tail PCM, then emits a
  heartbeat for each additional 1000 ms. Shorter pauses are held briefly and
  replayed unchanged if speech resumes. The source cursor prevents the retained
  tail from advancing context-schedule time twice.

The backend remains authoritative for VAD cuts. This event is a transport hint,
not permission for the frontend to finalize text or choose a sentence boundary.

### `audio_speech_start`

Precedes PCM when speech resumes after suppressed silence:

```json
{
  "type": "audio_speech_start",
  "capture_sample_index": 1939200,
  "preroll_samples": 6400
}
```

- `capture_sample_index` uses the same monotonic 16 kHz source cursor.
- `preroll_samples` is the number of following PCM samples replayed from the
  client pre-roll and must be between `0` and `32000`.
- Replayed samples remain available to ASR for onset protection but are not
  counted twice in the source timeline used by context schedules.
- The event does not itself prove speech. After a long-silence transition, a
  context-bearing state remains output-quarantined until backend Silero or the
  backend energy fallback confirms speech.

### `set_translation_direction`

Changes the translation direction for the current connection and clears pending
translation work.

```json
{
  "type": "set_translation_direction",
  "translation_direction": "zh2en"
}
```

Recommended UI behavior is to send this before `start` and lock the selector
during an active session, because the ASR force language and translation
direction should not drift apart mid-session.

### `set_tts_enabled` (deprecated)

Changes translated speech without changing ASR or translation state:

```json
{
  "type": "set_tts_enabled",
  "enabled": false,
  "tts_client_id": "page-9054a7d8-7f3e-4fd4-a147"
}
```

This message controls only the legacy owner/client-scoped TTS stream. It does
not enable or disable `/ws/tts` listeners. It does not rewrite subtitle text.

### `finish`

Requests graceful stop. The backend drains queued audio for tail accuracy,
flushes the current ASR state, commits the final safe tail, waits for pending
stable translation work needed by active broadcast or legacy output, and sends
one `final` message.

```json
{
  "type": "finish"
}
```

`mode: "slice"` is accepted for compatibility but ignored. The backend applies
normal stop semantics.

### `ping`

Health check.

```json
{
  "type": "ping"
}
```

The backend replies with `pong`.

## Server Messages

### `ready`

First message after WebSocket accept.

```json
{
  "type": "ready",
  "sample_rate": 16000,
  "translation_direction": "zh2en",
  "translation_source_language": "中文",
  "translation_target_language": "English",
  "tts_available": true,
  "tts_enabled": false
}
```

### `started`

Sent after `start` is applied.

```json
{
  "type": "started",
  "language": "English",
  "translation_direction": "en2zh",
  "translation_source_language": "English",
  "translation_target_language": "中文",
  "tts_available": true,
  "tts_enabled": true,
  "asr_context_active": true,
  "asr_context_term_count": 2,
  "asr_context_chars": 13
}
```

Context strings are never returned. Only accepted status, term count, and total
context characters are exposed.

### `partial`

Streaming ASR state update. This is generating text and can still change.

```json
{
  "type": "partial",
  "language": "English",
  "text": "Current ASR state text",
  "state_text": "Current ASR state text",
  "delta_text": "new suffix",
  "text_reset": false,
  "tentative_text": "newest uncommitted tail",
  "committed_text": "up to 100 recent solidified source subtitles",
  "translation": "up to 100 recent committed translations",
  "seq": 42,
  "is_stable": false,
  "stability": {
    "is_stable": false,
    "phase": "generating",
    "reason": "tentative_tail",
    "sentence_id": "",
    "segment_id": 1,
    "seq": 42,
    "committed_count": 3,
    "tentative_chars": 24,
    "unstable_chars": 24
  }
}
```

Frontend code should use `stability` from the backend instead of guessing
whether a sentence is stable from local word lists or punctuation heuristics.
`committed_text` and `translation` are bounded compatibility snapshots. Their
row limit is configured by `--subtitle-snapshot-history-size` (default `100`).
Use sentence events keyed by `sentence_id` for the canonical incremental stream;
the backend does not trim its internal meeting state when it trims a snapshot.

### `sentence_committed`

Sent when a source subtitle sentence becomes solidified. The frontend should add
one stable subtitle row keyed by `sentence_id`.

```json
{
  "type": "sentence_committed",
  "sentence_id": "1781901841676-386666-1",
  "revision": 1,
  "text": "The sentence is complete.",
  "language": "English",
  "seq": 43,
  "ts_ms": 1781901842000,
  "slice_commit": false,
  "boundary_kind": "sentence",
  "is_stable": true,
  "stability": {
    "is_stable": true,
    "phase": "solidified",
    "reason": "sentence_committed",
    "sentence_id": "1781901841676-386666-1",
    "segment_id": 1,
    "seq": 43,
    "committed_count": 4,
    "tentative_chars": 0,
    "unstable_chars": 0
  }
}
```

`revision` starts at `1`. Candidate position, rather than source-text equality,
is authoritative, so intentionally repeated speech remains as separate sentence
events with separate `sentence_id` values.

`boundary_kind` is `sentence` for a strong terminal boundary or
`stable_clause` for a long clause committed at a comma, semicolon, or colon.
Both are backend-authoritative stable units. A stable clause does not rotate or
finish the ASR streaming state.

### `sentence_updated`

Sent when a previously committed sentence is upgraded with a longer or more
complete version. The frontend must replace the existing source row with the
same `sentence_id`, not append a new row.

```json
{
  "type": "sentence_updated",
  "sentence_id": "1781901841676-386666-1",
  "revision": 2,
  "text": "The sentence is complete and now includes the corrected tail.",
  "language": "English",
  "seq": 44,
  "ts_ms": 1781901843000,
  "slice_commit": false,
  "is_stable": true,
  "stability": {
    "is_stable": true,
    "phase": "solidified",
    "reason": "sentence_updated",
    "sentence_id": "1781901841676-386666-1",
    "segment_id": 1,
    "seq": 44,
    "committed_count": 4,
    "tentative_chars": 0,
    "unstable_chars": 0
  }
}
```

Keep the old translation visible until a replacement `sentence_translation` for
the same `sentence_id` and `revision` arrives. Each accepted source upgrade
increments `revision` exactly once.

### `sentence_translation`

Sent when a committed or updated sentence has a translation.

```json
{
  "type": "sentence_translation",
  "sentence_id": "1781901841676-386666-1",
  "revision": 2,
  "translation": "译文",
  "seq": 44,
  "is_stable": true
}
```

The frontend should update the translation row by `sentence_id` and retain the
highest received `revision`. The backend checks sentence ID, revision, source
text, stream generation, and translation direction both before inference and
before publication. A superseded result is discarded and is never sent as a
`sentence_translation` event.

Built-in translation backends also validate the target script. For an English
target, a result that still contains Han characters is retried once with a
strict target-language prompt. If the retry is still mixed-language, no
`sentence_translation` or TTS job is published for that result; later sentence
jobs continue normally.

### `tts_job` (deprecated on `WS /ws`)

Sent only after the current stable `sentence_id` and `revision` translation has
passed the backend pre- and post-inference guards. It contains no text or audio:

```json
{
  "type": "tts_job",
  "job_id": "opaque-random-token",
  "sentence_id": "1781901841676-386666-1",
  "revision": 2,
  "source_order": 7,
  "target_language": "Chinese",
  "is_stable": true
}
```

The main page no longer requests or consumes this legacy event. Parallel
translations are reordered by `source_order`. A failed current
translation is explicitly skipped so it cannot block later jobs; a stale
revision cannot advance the queue or produce spoken output.

### `tts_status`

Reports `enabled`, `disabled`, `unavailable`, `queue_full`, or
`translation_drain_timeout`, together with `tts_available` and `tts_enabled`.
ASR and subtitle translation continue if TTS is unavailable or full.
`translation_drain_timeout` is a slow-drain warning threshold, not a discard:
the backend continues waiting for pending stable translations and emits their
ordered `tts_job` events before `final`. It never waits for audio synthesis or
browser playback.

### `sentence_reset`

Tells the frontend to clear committed subtitle rows and rebuild from subsequent
sentence events. Normal Stop does not emit this event. It is reserved for the
explicit, blocking `--final-redecode-on-stop` compatibility mode.

```json
{
  "type": "sentence_reset",
  "reason": "final_redecode"
}
```

Current reasons:

- `final_redecode`: a canonical full-session final re-decode was applied.
- `final_commit_reconcile`: only allowed when the final text is canonical. If
  full final re-decode is skipped because the audio is too long, the backend no
  longer emits this reset for a non-canonical tail.

### `final`

Sent once after `finish`.

```json
{
  "type": "final",
  "language": "English",
  "text": "Final backend state text",
  "state_text": "Final backend state text",
  "delta_text": "",
  "text_reset": false,
  "tentative_text": "",
  "committed_text": "up to 100 recent committed source subtitles",
  "translation": "up to 100 recent committed translations",
  "seq": 100,
  "is_stable": true,
  "stability": {
    "is_stable": true,
    "phase": "final",
    "reason": "stop",
    "sentence_id": "",
    "segment_id": 1,
    "seq": 100,
    "committed_count": 18,
    "tentative_chars": 0,
    "unstable_chars": 0
  }
}
```

`text` can be only the final ASR state, especially when full final re-decode is
skipped by `--final-redecode-max-sec`. Use `committed_text` and sentence events
to restore recent display state; sentence events remain the canonical complete
subtitle stream.

When translated speech is available, the backend waits for pending stable
translations and publishes all pending spoken items before `final`.
`--tts-final-translation-drain-sec` is the threshold for a slow-drain status,
not a hard timeout. The backend does not wait for CPU synthesis or browser
playback. The shared HLS timeline therefore continues independently after the
producer becomes inactive while at least one listener lease remains.

### Spoken Translation Revision Stability

Visible `sentence_committed`, `sentence_updated`, and `sentence_translation`
events remain immediate. Spoken translation has a separate backend gate:
`--tts-revision-stable-sec 3.0` requires the current source revision to remain
unchanged for 3.0 seconds before a TTS job is published. The timer starts at the
latest source revision, not at translation completion. A translation that
finishes after the source revision deadline can publish immediately.

`--tts-latest-revision-grace-sec 4.0` adds revision protection only to the
highest unsealed source order. It is not a global seven-second delay. Registering
a successor removes the extra grace from its predecessor, which then follows
the ordinary three-second rule. When `--segment-final-redecode` is enabled,
natural VAD finalization flushes pending endpoint audio and runs one bounded
one-shot decode over the current segment before sentence reconciliation. A
mid-speech hard cut only flushes the streaming state and rotates immediately so
incoming speech is not dropped while an old segment is re-decoded. A hard cut
is classified as mid-speech whenever its trailing silence has not reached the
configured `--vad-silence-sec` endpoint; there is no separate fixed silence
constant for this decision. An unchanged result validates the streaming text;
a safely compatible result may
repair its tail while retaining sentence revision semantics. Empty, failed,
context-echo, substantially divergent, or completed-unit-regressing results
are rejected. A `completed_unit_regression` keeps the streaming text when the
one-shot result would reduce the number of complete units inside the current
segment. An `effective_completed_unit_regression` applies the same protection
after carrying the previous segment's pending boundary prefix, before any
sentence cursor, translation, or TTS state is updated. Canonical candidate
updates are also rejected when they replay a material suffix of the preceding
candidate into the current sentence ID; already solidified neighboring rows
therefore cannot become overlapping duplicates after resegmentation. A
one-shot candidate that is only a strict normalized prefix of its already
solidified row is rejected as a terminal contraction; segment-final validation
may extend a tail or apply a compatible lexical repair, but cannot delete a
stable sentence ending.
Backend segment sealing occurs only for a validated segment, before the
streaming state and candidate cursor rotate. Ready sealed sources publish
immediately; an unvalidated segment remains protected by the normal revision
timers.

With segment-final re-decode enabled, the highest unsealed and unconfirmed
source has no timer deadline. `wait_state` reports
`waiting_for_segment_seal: true` with `required_quiet_ms: -1` and
`remaining_ms: -1`. Registering a successor removes this hold from its
predecessor; rollback confirmation, segment sealing, and orderly Stop continue
to release through their existing paths.

`--final-redecode-on-stop` is disabled by default. When explicitly enabled, it
can reconcile buffered full-session text only when Stop is requested, emits a
`sentence_reset`, and is too late to protect translations or TTS already
published during a live session.
Segment diagnostics emit `segment_final_redecode_done`,
`segment_final_redecode_skipped`, `segment_final_redecode_failed`, and
`tts_source_seal_deferred`. They include duration, latency, lengths, hashes, and
cut reason, but not source text.

A higher revision inside the quiet window invalidates the older ready
translation and restarts the source timer. Source order remains strict, so a
later sentence cannot overtake an unresolved earlier sentence. Set the option
to `0` only to disable the delay for controlled compatibility testing; revision
and source-order validation remain active.

Normal `finish` first reconciles final ASR text and drains current translation
work, then bypasses only the remaining quiet-window duration for the latest
ready revisions. An abrupt WebSocket disconnect does not force speech and
discards pending TTS state.

Structured diagnostics are `tts_stability_wait`, `tts_stability_reset`,
`tts_source_sealed`, `tts_stability_release`, and
`tts_late_revision_after_release`. Release reasons are `quiet_window`,
`latest_revision_grace`, `source_sealed`, or `final_force`. These events
contain short fingerprints, revisions, timing, and queue counts, but no source
text, translated text, raw sentence IDs, or raw TTS job IDs.

### Other Messages

- `processing`: backend is running a longer blocking decode.
- `pong`: reply to `ping`.
- `error`: recoverable or terminal error message.

Example:

```json
{
  "type": "error",
  "message": "too many active connections"
}
```

## Backend Behavior Updates

The current backend owns the streaming state and all complex segmentation logic.
The frontend sends audio chunks, compresses only confidently silent transport
spans, and renders backend sentence events; it does not run sentence stability
or slicing heuristics.

Important behavior updates:

- Explicit stability contract: `partial`, `final`, `sentence_committed`, and
  `sentence_updated` include `is_stable` plus a `stability` object.
- Stable sentence updates: committed sentences can be upgraded through
  `sentence_updated`; clients must update by `sentence_id` and `revision`.
- Candidate cursor: every completed ASR candidate position is consumed once.
  Normal speech is not removed merely because its text equals an earlier
  sentence. Duplicate suppression is limited to segment-boundary replay with
  explicit overlap evidence.
- Revision-safe translation: queued work carries the source sentence revision.
  Superseded requests are rejected before translation when possible and again
  before publication, preventing an older response from overwriting a newer
  source sentence.
- Translation direction: `zh2en` and `en2zh` are first-class protocol values.
  `start` applies the requested direction and `set_translation_direction`
  returns the effective source and target labels.
- Translation queue cleanup: changing direction or starting a new session
  clears pending translation tasks to avoid stale-language output.
- Target-language output guard: built-in `zh2en` translation retries an English
  result containing untranslated Han characters once and rejects a still-mixed
  retry before subtitle or TTS publication.
- Final reconcile guard: `final_commit_reconcile` no longer resets subtitles
  when final text is non-canonical, such as when full final re-decode is skipped
  due to `--final-redecode-max-sec`.
- Punctuation timeout cutting is disabled in streaming mode because it caused
  sentence loss/regression in long speech.
- Backend trace logging can write structured subtitle and text-pool rows to a
  JSONL file through `--subtitle-trace-log-file`.

### ESV terminology policy for Chinese-to-English

Chinese-to-English requests instruct both translation backends to use standard
English Standard Version (ESV) conventions for Christian, biblical, and
theological terminology. The policy may prefer ESV wording for a clearly
identifiable quotation, but it must not complete fragments, add omitted text, or
replace the supplied source with a memorized verse. Other translation directions
retain the general fidelity-only prompt.

### Bounded ASR Context Schedule

`--asr-context-schedule <path>` optionally supplies a short, time-windowed
glossary for vLLM ASR states. It is disabled by default. The schedule is data,
not sentence-splitting logic, and must not contain transcript sentences or
punctuation-delimited phrases.

```json
{
  "version": 1,
  "language": "Chinese",
  "global_terms": ["出埃及记"],
  "segments": [
    {
      "start_sec": 0,
      "end_sec": 120,
      "terms": ["暗兰", "约基别"],
      "anchors": ["出埃及记六章"]
    }
  ]
}
```

The backend selects terms using consumed audio time when a streaming state is
created. `global_terms` are considered first and intentionally remain active for
the whole session; leave them empty when every term must expire with a time
window. Matching segment terms follow. `anchors` document how a window was
located but are not sent to the ASR model. Segment ranges must be ordered and
non-overlapping. A schedule is ignored when the session language is automatic or
when its `language` does not match the session's forced ASR language.

Term validation rejects sentence punctuation, including sentence-final ASCII
periods and colons. Dotted uppercase initialisms such as `U.S.` remain valid.

Context is applied in `streaming` mode by default. The selected terms are sent
to every streaming decode, so terminology can affect the generating subtitle
before VAD or hard-cut finalization. The exact selected term tuple is also kept
on the streaming state; the backend does not reconstruct the list by parsing
the prompt.

Streaming context can occasionally make uncertain audio copy several glossary
terms. Before a completed sentence receives a `sentence_id`, the backend scans
it using longest-term-first matching. If one run contains at least three context
terms and every separator in that run is either whitespace or zero length, the
backend removes only the matched run. Any real text before or after the run is
preserved for solidification, translation, and TTS. A pure context-only
candidate is discarded. Punctuation or any non-context text between terms
breaks the run. The model output may appear temporarily in the generating
partial subtitle because partial output intentionally reflects the live model
state. The same gate rejects a context-shaped revision of an already solidified
sentence.

This rule is language-independent and uses only the active session terms. It
does not contain a fixed vocabulary, language-specific tokenization, or known
phrase replacements. The existing silence-resume quarantine and whole-glossary
echo checks remain as earlier defenses.

`--asr-context-apply-mode segment_final` remains available as an explicit
compatibility mode. It keeps context out of live partial recognition and runs a
context-assisted decode only after VAD or hard cut flushes a complete segment.
Accepted lexical corrections can revise existing sentence IDs and replacement
translations, so this mode is not suitable when terminology must affect the
first translation and TTS publication.

Safety and tuning options:

- `--asr-context-max-terms 24`: maximum terms selected for one state.
- `--asr-context-max-chars 160`: maximum context length; only whole terms are
  retained.
- `--asr-context-lookaround-sec 30`: includes nearby windows to tolerate timing
  drift at state boundaries.
- `--asr-context-apply-mode streaming`: default; accepted values are
  `streaming` and `segment_final`.

Invalid schedules fail startup before model loading. The `asr_context_selected`
trace event contains elapsed audio time, language, apply mode, term/character
counts, and a SHA-256 fingerprint. `context_run_commit_trimmed` records a
matched run removed while retaining the candidate's real text;
`context_run_commit_dropped` records a pure context-only candidate rejected by
the streaming commit gate; `context_run_upgrade_rejected` records a rejected
revision. These events contain only candidate hashes, lengths, matched-term
counts, and segment metadata, never the context terms themselves.
Segment-final compatibility correction emits
`asr_context_final_redecode_done`, `asr_context_final_redecode_skipped`, or
`asr_context_final_redecode_failed`; these events also omit context text and
store only exception type/fingerprint on failures. `sentence_upgrade_commit`
sets `context_correction: true` when a segment-final lexical correction updates
an already solidified row.
Because strong context can bias or repeat rare terms, validate every schedule
against both its target recording and unrelated speech before enabling it in a
service.

### Per-session context override

Authenticated web clients may add `asr_context_terms` to a `start` message:

```json
{
  "type": "start",
  "language": "English",
  "translation_direction": "en2zh",
  "asr_context_terms": ["Elisha", "Jordan"]
}
```

Field presence is significant. A non-empty array overrides the configured
schedule for that WebSocket session. An empty array explicitly disables context.
Omitting the field retains schedule behavior for legacy clients. The accepted
override is immutable for the active session and is inherited by every VAD and
hard-cut streaming-state rotation.

The backend trims entries, removes empty entries and case-insensitive
duplicates, then enforces `--asr-context-max-terms` and
`--asr-context-max-chars`. Each entry must be one term without internal
whitespace or sentence punctuation. Dotted uppercase initialisms such as `U.S.`
remain valid. Invalid input rejects `start` before the current state is replaced.

`started` reports only `asr_context_active`, `asr_context_term_count`, and
`asr_context_chars`. Traces add context source, active status, counts, and a
SHA-256 fingerprint, but never contain submitted strings. Validation and state
creation failures are traced by exception type and fingerprint rather than
exception text.

### Stable Terminal Sentence Promotion

The backend can solidify the newest completed sentence before another sentence
arrives, but only after the ASR output remains unchanged for both a duration and
an observation-count threshold:

- `--early-translation-stable-sec 0.8`
- `--early-translation-stable-hits 3`
- `--early-translation-short-stable-sec 1.2`
- `--early-translation-short-stable-hits 4`
- `--stable-clause-target-cjk-chars 32`
- `--stable-clause-target-latin-words 24`

The stricter pair applies structurally to short English terminal sentences.
A terminal sentence or long clause is not promoted until unchanged-duration and
observation gates pass and following tokenizer output reaches at least
`--unfixed-token-num`. This keeps the committed unit outside the model rollback
window.

Long text can be exposed as smaller stable translation units at generic comma,
semicolon, or colon boundaries. The target is selected by script, not by words
or phrases. The splitter chooses the first eligible boundary at or after the
target, so later text cannot move an earlier boundary. It leaves the newest unit
tentative, requires the full tokenizer rollback lookahead, and never cuts the
ASR audio/state at a soft boundary. Setting the corresponding target to `0`
disables this behavior.
Short text without strong terminal punctuation remains the generating tail.
These decisions are backend-authoritative; clients must not maintain word
lists, punctuation rules, or fixed timers to infer stability.

### Stable Small Sentence-tail Revisions

A solidified sentence can still receive a short monotonic suffix from a later
model state. The backend does not publish the first hypothesis: it requires
three identical observations spanning at least 600 ms. Once stable, it emits a
single `sentence_updated` message with the same `sentence_id` and the next
`revision`.

If the candidate changes, retracts, or disappears before acceptance, the
backend discards it. A successful `finish_streaming_transcribe` during segment
rotation or final WebSocket shutdown can reconcile a finalized monotonic
extension immediately; ordinary partial output cannot bypass the stability
gate.

Translation work is revision-aware. A new source revision supersedes queued or
in-flight work for older revisions, while the previous translation remains
visible until its replacement is ready. This avoids a blank subtitle row during
correction.

The transition-oriented trace events are
`sentence_upgrade_deferred`, `sentence_upgrade_candidate_reset`,
`sentence_upgrade_rejected`, and `sentence_upgrade_small_commit`.

## Trace Events

When `--subtitle-trace-log` is enabled, logs contain two structured topics:

- `subtitle_state`: WebSocket lifecycle, ASR output, segmentation, VAD, final
  reconcile, alignment summary, and message-send probes.
- `text_pool`: generating and solidified text pool transitions, including
  pending prefix, committed text, resets, and regression snapshots.

Relevant recent events:

- `silero_shadow_ready`: the per-WebSocket ONNX observer loaded successfully.
  `control_mode` is `observe_only` unless `--silero-vad-rescue` is enabled; the
  rescue mode may prevent strongly speech-positive audio from being skipped,
  but cannot cut a segment or commit text. Silero speech evidence can also
  release the context-only output guard described below.
- `silero_decode_rescue`: a low-energy batch was retained for ASR because its
  Silero peak and accumulated speech evidence passed the conservative rescue
  threshold. This is intentionally based on the complete batch rather than its
  mean or final Silero frame, because inference backpressure may coalesce speech
  followed by silence into one batch. A single isolated probability spike does
  not satisfy the accumulated-evidence floor. The event records probabilities,
  duration, SNR, and the preserved energy-VAD endpoint state but no text.
- `audio_backpressure_wait_start` / `audio_backpressure_wait_end`: queued PCM
  reached the hard duration cap, so the backend paused WebSocket ingress until
  the independent consumer recovered. The current frame remains in bounded
  spill storage; this path does not delete oldest audio.
- `silero_shadow_observation`: sampled speech probability, SNR state, current
  decode-skip decision, and whether the two detectors disagree.
- `silero_shadow_transition`: immediate Silero speech/non-speech state change.
- `silero_shadow_disagreement`: transition into or out of disagreement between
  Silero and the existing SNR decode gate.
- `silero_shadow_unavailable`: model loading or inference failed. ASR continues
  with the existing VAD because the shadow observer is fail-open.
- `audio_preroll_replayed`: the number and duration of previously skipped
  samples prepended once when ASR decoding resumes.
- `endpoint_tail_decoded`: a bounded skipped-audio tail was decoded once
  immediately before segment or stop finalization; includes duration and RMS
  but no recognized text.
- `endpoint_tail_decode_failed`: endpoint-tail decoding failed and the buffered
  samples were restored for retry.
- `idle_preroll_discarded`: skipped low-energy PCM was discarded because a
  client-silence event arrived without an active segment or pending text.
- `client_silence_queued`: an authenticated client silence span entered the
  ordered consumer queue behind all preceding PCM.
- `client_silence_applied`: the span advanced backend silence and source clocks
  without an ASR decode; includes duration, VAD readiness, and context-guard
  state, but no audio or text.
- `client_speech_start`: records the monotonic source cursor and pre-roll size;
  it does not release the backend speech-evidence guard by itself.
- `asr_context_resume_guard_armed`: at least three consecutive silent decode
  skips, a client long-silence event, or a new post-cut state activated the
  context-output guard. Long-silence and post-cut states set
  `require_speech_evidence: true`; legacy energy-only resumes retain the earlier
  context-fragment behavior.
- `asr_context_resume_partial_quarantined`: under strict long-silence protection,
  every partial is withheld until 200 ms of backend Silero speech is confirmed,
  regardless of how many context terms it contains. Without Silero, sustained
  backend energy provides the fallback. The client `audio_speech_start` event
  alone cannot release this guard.
- `asr_context_resume_guard_released`: the guard was released by confirmed
  speech, ordinary non-context output, or expiry of the fallback window.
- `asr_context_silent_segment_discarded`: segment or stop finalization produced
  a context-dominated candidate while no resumed speech had been confirmed.
  The candidate was not committed or translated. A prior non-context subtitle
  snapshot is retained when available. These events contain hashes and lengths,
  not the context or recognized text itself.
- `asr_context_final_tail_repair_trusted`: an active context silence guard
  accepted a final candidate because the segment had sufficient backend speech
  activity, the candidate was an aligned correction of the prior non-context
  snapshot, and context-echo checks passed. Only lengths, hashes, and activity
  duration are logged.
- `stream_finish_skipped_no_decode`: Stop did not flush or finish a state that
  had never performed streaming ASR decode. Any buffered low-energy pre-roll is
  discarded and cannot create context hotword output.
- `pending_prefix_vad_boundary_preserved`: a VAD-finalized text unit remained
  deferred by rollback protection and was retained as an independent prefix of
  the next candidate timeline despite having no textual overlap with the next
  segment. It is cleared only through normal commit/alignment handling.

- `final_commit_reconcile_skipped_noncanonical`: final text differs from
  committed text but is not a canonical full-session decode, so the backend kept
  existing subtitles.
- `sentence_new_commit`: a new solidified source unit was emitted; includes
  `boundary_kind`, tokenizer lookahead, required lookahead, and rollback safety.
- `sentence_updated`: an existing source sentence was upgraded.
- `candidate_action`: records whether a completed candidate was committed,
  upgraded, or skipped as structural segment overlap, including its candidate
  index, sentence ID, revision, and text hash.
- `translation_stale_drop`: records a superseded translation request rejected
  during `pre_inference` or `post_inference`, with queued/current revisions and
  source hashes.
- `translation_target_language_mismatch`: the first built-in translation result
  did not match the English target script and a strict retry was requested.
- `translation_target_language_rejected`: the strict retry remained
  mixed-language and was withheld from subtitles and TTS.
- `early_translation_stability_wait`: records the first-seen timestamp, stable
  age, hit count, required thresholds, and short-English classification while a
  terminal sentence remains tentative.
- `early_translation_stable_commit`: records the same readiness evidence when
  the terminal sentence becomes solidified.
- `stable_clause_rollback_wait`: records a deterministic clause held because
  its following tokenizer output has not yet crossed `--unfixed-token-num`.
- `pending_prefix_terminal_boundary_preserved`: records when a short terminal
  sentence immediately before a hard cut is kept as its own candidate because
  the next segment starts with a distinct completed sentence. The associated
  prefix remains virtual until candidate positions are stable, preventing a
  later source-sentence expansion from being skipped after index realignment.
- `alignment_summary`: model-observed versus committed sentence coverage at
  WebSocket close.

Trace logs can contain meeting content. Treat them as sensitive data and do not
commit them.
