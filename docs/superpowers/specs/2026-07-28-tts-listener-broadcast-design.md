# Independent Multi-Listener TTS Broadcast Design

## Objective

Move translated-speech playback out of the main subtitle page and into a dedicated,
authenticated listener page. One active ASR/translation session publishes only stable
translations. Any number of devices may join the live listener page and play future
translations in strict FIFO order without affecting the producer or one another.

Late listeners do not receive history. Each listener starts with the first stable
translation published after that listener has joined.

## Scope

This change provides one process-local live TTS channel because VoxBridge currently
permits one active ASR WebSocket session. It does not add meeting rooms, historical
replay, persistent audio storage, synchronized playback positions, or a distributed
message broker.

## Selected Approach

Use an application-level TTS broadcast hub with one bounded queue per listener and one
shared job/audio record per translated sentence.

Alternatives were rejected for the following reasons:

- Per-listener job copies multiply text, audio, cleanup state, and synthesis work as
  listener count grows.
- Polling or SSE can carry one-way notifications, but the WebSocket also needs listener
  acknowledgements and explicit lifecycle messages. A dedicated WebSocket keeps that
  contract in one place.
- Redis or another external broker is unnecessary for the current single-process,
  single-channel deployment.

## Architecture

### Producer

The existing ASR WebSocket remains the only producer. It continues to use
`OrderedTTSBuffer` so translations are eligible for speech only when all of these are
true:

1. The source sentence is committed and stable.
2. The translation revision is still current.
3. Earlier source orders are ready or explicitly failed.

TTS ordering is enabled automatically whenever the configured translator and Kokoro
synthesizer are available. It no longer depends on a checkbox, browser-generated
client ID, or audio playback in the subtitle page.

When an ordered translation becomes ready, the producer calls the application-level
broadcast hub. If no listener is connected, no broadcast job is retained. This matches
the future-only join contract and avoids idle memory growth. If listeners are present,
one immutable job is created and announced to all listeners that were connected at
publication time.

The producer still drains pending stable translation work before its final WebSocket
message. Listener playback is independent and may continue after the producer has
stopped.

### Broadcast Hub

Create a focused module at `voxbridge/tts/broadcast.py` with these responsibilities:

- Register and unregister authenticated listeners.
- Snapshot current listeners when a stable translation is published.
- Assign an opaque random job ID and listener ID.
- Fan out job metadata through bounded per-listener `asyncio.Queue` instances.
- Validate that an audio request belongs to the authenticated listener that received
  the job.
- Track delivery acknowledgements independently for every listener.
- Remove a job when all intended listeners have acknowledged it or disconnected.
- Expire abandoned jobs after the configured TTS job TTL.
- Isolate a slow listener instead of blocking the producer or other listeners.

The hub stores translated text only in memory. It does not put text in URLs, WebSocket
job notifications, access logs, or exception messages.

### Job Lifecycle

For each stable translated sentence:

1. The hub snapshots the connected listener IDs.
2. With no listeners, publication returns without creating a job.
3. With listeners, the hub stores one job containing translated text, language,
   source identity, expiration, and the pending-listener set.
4. Each listener receives the same job ID and sentence metadata.
5. A listener requests the WAV using its opaque listener ID and authenticated cookie.
6. The first request synthesizes and caches the WAV under the existing global synthesis
   lock. Concurrent and later listeners reuse the cached bytes.
7. After a browser has fully loaded the WAV into memory, it sends `tts_received` over
   its listener WebSocket.
8. The hub removes only that listener from the pending set.
9. When no pending listener or in-flight audio request remains, the hub deletes the
   text and cached WAV immediately.

An audio request temporarily increments an in-flight lease. Disconnect cleanup may
remove the listener from pending delivery, but cannot delete a job until all in-flight
requests have released their leases. A failed fetch remains retryable while the
listener is connected and the job has not expired.

### Slow Listener Isolation

Each listener has a bounded server notification queue. Normal WebSocket delivery moves
notifications quickly into the browser's FIFO playback queue. If the server queue
fills, that listener is closed with an explicit overload status and removed from all
pending job sets. The producer and other listeners continue unchanged.

No unread job is silently overwritten. A disconnected listener rejoins as a new
future-only listener and does not recover its old queue.

The browser fetches audio only for the next item to play. This keeps CPU synthesis and
browser memory bounded instead of downloading the entire pending queue in advance.

## HTTP and WebSocket Contract

### `GET /listen`

Returns the standalone translated-speech listener page. It requires the same login as
the main page. When authentication redirects a device to `/login`, a validated relative
`next=/listen` value returns the device to the listener page after login. Absolute or
cross-origin redirect targets are rejected.

### `WebSocket /ws/tts`

The WebSocket requires a valid authentication cookie. On acceptance the server sends:

```json
{
  "type": "tts_listener_ready",
  "listener_id": "opaque-random-token",
  "tts_available": true,
  "producer_active": true
}
```

New stable translations are announced without translated text:

```json
{
  "type": "tts_job",
  "job_id": "opaque-random-token",
  "sentence_id": "opaque-source-id",
  "revision": 2,
  "source_order": 7,
  "target_language": "English",
  "is_stable": true
}
```

After loading the complete audio response, the listener sends:

```json
{
  "type": "tts_received",
  "job_id": "opaque-random-token"
}
```

The server also publishes `producer_status` messages with `active: true` on ASR start
and `active: false` after final publication. Stopping a producer does not clear browser
queues.

Ping/pong messages keep idle listener connections detectable. Unknown message types or
invalid job acknowledgements return a non-sensitive error and do not affect other
listeners.

### `POST /api/tts/broadcast/jobs/{job_id}/audio`

Returns `audio/wav`. The listener supplies its opaque listener ID in the
`X-TTS-Listener-ID` header. The route requires both a valid login session and a hub
registration owned by that same session. Missing, expired, foreign, or unassigned jobs
return `404` without revealing which condition failed.

Responses retain `Cache-Control: no-store` and the existing sample-rate and duration
headers. There is no shared DELETE endpoint; delivery is acknowledged per listener over
the listener WebSocket.

The existing owner-scoped TTS job endpoints remain temporarily available for API
compatibility, but the new frontend does not create or consume those jobs. They are
documented as deprecated.

## Frontend Behavior

### Main Subtitle Page

Remove all direct audio responsibilities:

- Remove the translated-speech checkbox and local playback status badge.
- Remove Web Audio state, fetch/ack logic, and FIFO playback code.
- Stop sending `tts_enabled` and `tts_client_id` from the main page.
- Ignore legacy producer-WebSocket `tts_job` events.
- Add an authenticated link labeled `打开译文朗读页` that opens `/listen` in a new tab.

ASR input selection, subtitles, translation direction, context, scrolling, and font
controls remain unchanged.

### Listener Page

The listener page is a small mobile-first light interface containing:

- `开始收听` and `停止收听` controls.
- Connection, producer, playback, and backlog status.
- The currently playing target language and queue position, without exposing hidden
  translation text in URLs.
- A concise explanation that only translations produced after joining are played.

The explicit Start click satisfies browser autoplay policy. Start opens `/ws/tts` and
initializes/resumes one `AudioContext`. Incoming jobs append to a FIFO. The playback
pump fetches only the head item, decodes it, acknowledges receipt, waits for playback
completion, and then advances. Stop aborts the current fetch, stops the source node,
clears only that device's browser queue, and closes its listener WebSocket.

Reconnect never reuses a previous listener ID or queue. A network failure changes the
status to disconnected and requires an explicit Start click so the page cannot produce
unexpected sound.

## Authentication and Privacy

- `/listen`, `/ws/tts`, and broadcast audio require the existing authenticated session.
- Listener IDs and job IDs use cryptographically random URL-safe tokens.
- Listener ownership is derived from a hash of the authenticated cookie, never the raw
  cookie value.
- Cross-session audio access returns the same `404` as an unknown job.
- Translation text remains memory-only and is excluded from listener event payloads,
  URLs, normal logs, and trace previews.
- Authentication-disabled mode remains limited to trusted local development and uses
  an anonymous owner key.

## Resource Bounds and Cleanup

- Keep the existing TTS job TTL and maximum-job CLI controls as hard process bounds.
- Add a bounded listener notification queue with a conservative default of 128 jobs.
- Synthesize under the existing single global CPU lock.
- Share one cached WAV across all intended listeners.
- Remove pending ownership on WebSocket disconnect.
- Delete completed jobs immediately after every intended listener has acknowledged and
  all audio leases have ended.
- Prune expired jobs during publish, register, audio lookup, acknowledgement, and
  disconnect operations.
- Keep translated speech entirely process-local; service restart intentionally clears
  all listeners and pending jobs.

## Error Handling

- TTS unavailable: listener page connects, reports unavailable, and receives no jobs;
  ASR and text translation continue.
- Synthesis failure: return `503`, retain the listener's pending delivery until expiry,
  and let the browser retry once before skipping locally with a visible error.
- Listener backlog overflow: disconnect only the slow listener and release its pending
  ownership.
- Producer stop: publish inactive status after all stable TTS jobs have been announced;
  listeners finish their local queues.
- Listener disconnect: release that listener from all jobs without cancelling synthesis
  or playback on other devices.
- Expired job: return `404`; the listener skips it and continues with the next FIFO item.

## Compatibility and Migration

The ASR and sentence translation WebSocket messages do not change. The existing
`tts_enabled`, `tts_client_id`, `set_tts_enabled`, owner-scoped audio, acknowledgement,
and client-cancellation contracts remain accepted for one compatibility cycle but are
not emitted or used by the new pages. API documentation marks them deprecated.

The service command and port remain unchanged. Kokoro models, CPU thread settings,
voice settings, synthesis limits, and final translation drain behavior remain active.

## Verification

### Unit Tests

- Two listeners receive one published job and share the same job ID.
- A listener joining after publication does not receive that job.
- One listener acknowledgement preserves the job for another listener.
- All acknowledgements remove text and cached audio.
- Disconnect releases only the disconnected listener's pending deliveries.
- An in-flight lease prevents premature deletion during disconnect.
- Expiration and capacity limits remove or reject jobs deterministically.
- Slow-listener overflow does not affect another listener.
- Foreign authenticated ownership cannot claim audio or acknowledge delivery.

### Protocol Tests

- `/listen` and `/ws/tts` require authentication.
- Login safely returns an authenticated device to `/listen`.
- Two authenticated devices receive the same future stable translation.
- The WAV is synthesized once and served to both devices.
- Each listener's acknowledgement is independent.
- Producer stop announces inactive status after the last stable job.
- No listener means no retained broadcast job or CPU synthesis.
- The main page contains no audio playback code and links to `/listen`.
- The listener page appends jobs to FIFO and never replaces an active source.

### End-to-End Checks

- Start ASR from the main page and open `/listen` on desktop and mobile devices.
- Confirm both devices hear the same subsequent translations in order.
- Stop one device and confirm the other continues uninterrupted.
- Join a third device and confirm it does not replay earlier translations.
- Stop the ASR session and confirm all already queued audio finishes.
- Run a prolonged session while monitoring process RSS, listener count, retained jobs,
  synthesis count, and `NRestarts`.

