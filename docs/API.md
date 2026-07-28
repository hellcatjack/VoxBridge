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
deployments, also prefer `--disable-debug-file`.

### `GET /`

Returns the browser UI as HTML.

When authentication is enabled, unauthenticated requests redirect to `/login`.

### `GET /listen`

Returns the standalone translated-speech listener. The main subtitle page does
not synthesize or play audio. When authentication is enabled, an unauthenticated
request redirects to `/login?next=%2Flisten`, and a successful login returns to
the listener page. Each browser must explicitly select Start to activate audio.

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

### `POST /api/tts/broadcast/jobs/{job_id}/audio`

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

### `WS /ws/tts`

Authenticated standalone translated-speech listener protocol. Listener sockets
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
  "committed_text": "all solidified source subtitles",
  "translation": "all committed translations",
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
sentence events.

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
  "committed_text": "all committed source subtitles",
  "translation": "all committed translations",
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
as the canonical subtitle stream.

When broadcast TTS is available, the backend waits for pending stable
translations and publishes all pending listener jobs before `final`.
`--tts-final-translation-drain-sec` is the threshold for a slow-drain status,
not a hard timeout. The backend does not wait for CPU synthesis or browser
playback. Listener FIFOs therefore continue independently after the producer
becomes inactive.

### Spoken Translation Revision Stability

Visible `sentence_committed`, `sentence_updated`, and `sentence_translation`
events remain immediate. Spoken translation has a separate backend gate:
`--tts-revision-stable-sec 3.0` requires the current source revision to remain
unchanged for 3.0 seconds before a TTS job is published. The timer starts at the
latest source revision, not at translation completion. A translation that
finishes after the source revision deadline can publish immediately.

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
`tts_stability_release`, and `tts_late_revision_after_release`. These events
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
The frontend sends audio chunks and renders backend sentence events; it should
not run its own sentence stability or slicing heuristics.

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

Context is applied in `segment_final` mode by default. Live partial recognition
uses an empty context, preserving natural low-latency output. After VAD or hard
cut flushes the complete state, the backend runs one context-assisted decode on
that complete segment and sends the corrected result through the existing text
pool and translation pipeline. This avoids exposing a glossary to every 0.6
second decode, which can make uncertain audio copy the glossary as speech.
If a multi-term correction still closely matches the glossary itself and the
context-free streaming output provides no comparable acoustic evidence, the
backend rejects it as `context_echo` and keeps the streaming text. Accepted
same-length lexical corrections update the existing `sentence_id`, increment
its revision, and enqueue a replacement translation.

`--asr-context-apply-mode streaming` restores the original compatibility
behavior and sends the selected context to every streaming decode. Do not use
that mode for an unverified glossary. A glossary-shaped output is filtered only
when it appears without similar incremental text from the preceding decode, so
a speaker who audibly enumerates the same terms is retained. After at least one `segment_final`
correction is accepted, the backend skips the context-free whole-session stop
re-decode so it cannot overwrite already corrected segments. If no schedule
window matched, normal stop-time re-decode remains enabled.

Safety and tuning options:

- `--asr-context-max-terms 24`: maximum terms selected for one state.
- `--asr-context-max-chars 160`: maximum context length; only whole terms are
  retained.
- `--asr-context-lookaround-sec 30`: includes nearby windows to tolerate timing
  drift at state boundaries.
- `--asr-context-apply-mode segment_final`: safe default; accepted values are
  `segment_final` and `streaming`.

Invalid schedules fail startup before model loading. The `asr_context_selected`
trace event contains elapsed audio time, language, apply mode, term/character
counts, and a SHA-256 fingerprint. Segment-final correction emits
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

The stricter pair applies structurally to short English terminal sentences.
Text without terminal punctuation remains the generating tail and is not
promoted by this gate. These decisions are backend-authoritative; clients must
not maintain word lists, punctuation rules, or fixed timers to infer stability.

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

- `final_commit_reconcile_skipped_noncanonical`: final text differs from
  committed text but is not a canonical full-session decode, so the backend kept
  existing subtitles.
- `sentence_new_commit`: a new solidified source sentence was emitted.
- `sentence_updated`: an existing source sentence was upgraded.
- `candidate_action`: records whether a completed candidate was committed,
  upgraded, or skipped as structural segment overlap, including its candidate
  index, sentence ID, revision, and text hash.
- `translation_stale_drop`: records a superseded translation request rejected
  during `pre_inference` or `post_inference`, with queued/current revisions and
  source hashes.
- `early_translation_stability_wait`: records the first-seen timestamp, stable
  age, hit count, required thresholds, and short-English classification while a
  terminal sentence remains tentative.
- `early_translation_stable_commit`: records the same readiness evidence when
  the terminal sentence becomes solidified.
- `pending_prefix_terminal_boundary_preserved`: records when a short terminal
  sentence immediately before a hard cut is kept as its own candidate because
  the next segment starts with a distinct completed sentence. The associated
  prefix remains virtual until candidate positions are stable, preventing a
  later source-sentence expansion from being skipped after index realignment.
- `alignment_summary`: model-observed versus committed sentence coverage at
  WebSocket close.

Trace logs can contain meeting content. Treat them as sensitive data and do not
commit them.
