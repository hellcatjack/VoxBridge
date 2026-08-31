# Shared Speech-Only HLS Timeline Design

Date: 2026-08-31

## Context

VoxBridge currently fans one shared HLS encoder out to all active listener leases. The encoder continuously writes a very-low-level idle carrier whenever no translated PCM is available. This keeps the HLS playlist advancing in wall-clock time, but it also permanently encodes translation and stability latency as silent media.

The Listen page later tries to remove those historical silent spans by seeking to the next caption cue. Because each browser has a different HLS implementation, buffer window, caption-poll time, and playhead, iPhone native HLS and Windows Chrome/HLS.js can make that seek at different times. The current safety guard also waits for additional audio after the next cue before seeking. The result is device-dependent waiting even though every listener reads the same underlying stream.

## Goals

- Give every listener the same ordered media timeline and the same encoded sentence spacing.
- Publish newly released translated speech as soon as one decodable HLS slice can be produced.
- Permit a short, honest buffer wait at the current live edge instead of encoding several seconds of artificial silence.
- Preserve the existing shared encoder: listener count must not multiply TTS synthesis or FFmpeg work.
- Preserve the existing Auto playback-speed thresholds and rollback-safe translation release rules.
- Continue to support iPhone native HLS, Windows Chrome/HLS.js, background playback, and listener leases.

## Non-goals

- Millisecond synchronization between devices. Network delivery, decoder startup, and device scheduling can differ by roughly one HLS segment.
- Reducing the translation model's stability window or allowing provisional translations to be spoken.
- Replacing HLS with WebSocket audio, WebRTC, or per-listener audio generation.
- Forcing every listener to use the same manually selected playback rate.

## Chosen Architecture

The server will own silence compaction by maintaining a speech-only HLS media timeline. The browser will no longer rewrite its local playhead to remove server-generated idle spans.

### Encoder startup

When the first listener creates an epoch, the encoder will write only the minimum bootstrap carrier required for FFmpeg to publish a valid, decodable HLS playlist. Bootstrap duration will be derived from the configured segment and writer-frame sizes rather than a duplicated magic constant.

The bootstrap bytes are reserved in the encoder's media accounting before translated PCM can be scheduled. This prevents a sentence submitted during startup from receiving timestamps that overlap bootstrap audio.

Bootstrap audio is an epoch-start implementation detail. It is not repeated between sentences and is not counted as translated-audio backlog.

### Idle behavior

After bootstrap completes, the writer waits on the PCM queue when no translated audio is available. It does not continuously write idle carrier frames. Consequently:

- the playlist and media sequence pause while there is nothing to say;
- FFmpeg remains alive with its stdin open;
- HLS clients continue polling the open live playlist;
- no wall-clock translation latency becomes playable silence.

When PCM arrives, the writer resumes immediately. A new complete segment may require up to one configured HLS segment of media before it becomes available, which is the accepted short buffering pause.

### Sentence timeline and captions

Released TTS PCM remains ordered by the existing shared publisher. Each sentence retains the configured 300 ms sentence pause already appended by the publisher. Sentences are scheduled directly after the preceding submitted PCM; wall-clock time spent waiting for translation is excluded.

`HLSAppendReceipt`, caption cue timestamps, `playingDate`, playlist program date-time, and `live_edge_at_ms` will all continue to use the same compressed media timeline. Caption selection therefore remains tied to the media actually being decoded rather than wall-clock translation time.

The first implementation will not trim synthesizer leading or trailing audio beyond the existing activity-bound cue calculation. Such audio, if present, is shared media and therefore identical for all listeners.

### Listener behavior

The Listen page will stop performing historical-silence seek compaction. The related per-browser pending-seek state, buffer guards, recovery timer, and seek-specific event branches will be removed.

At the end of available media, `waiting` or `stalled` will be treated as an ordinary live-edge buffer condition. When the next shared segment appears, the same media element resumes through its normal `playing` event. No device chooses a different target time.

Auto playback rate remains unchanged. It may temporarily fall back to 1.0x on a device whose forward buffer is too small; that is a local anti-stutter safeguard, not a change to shared content or sentence spacing.

## Multiple-listener Semantics

All active listener IDs receive listener-scoped playlist and segment URLs that map to the same epoch files. Adding listeners only adds and renews leases; it does not create another encoder, queue, or synthesized copy.

Two listeners that choose the same playback mode will hear the same sentence order and encoded sentence pauses. They can differ by network and decoder latency, normally bounded near one segment, but neither can encounter an idle span that another client locally skipped.

A listener joining an existing epoch follows normal HLS live-join behavior and does not replay the entire retained playlist. Existing listener capacity and 90-second lease expiry remain unchanged.

## Failure and Recovery Behavior

- A stalled playlist is not an encoder failure while the PCM queue is empty.
- FFmpeg exit, broken stdin, or an unavailable playlist remains a stream error and follows existing error reporting.
- If a browser reports `waiting` at the live edge, the UI shows buffering but keeps the stream attached and allows native HLS/HLS.js polling to resume it.
- Genuine playback errors retain the existing Resume Audio path.
- When the last listener lease disappears, the epoch and encoder are stopped exactly as today.

No periodic idle heartbeat segment will be added in the first implementation because that would reintroduce artificial audio. If a verified platform abandons an unchanged live playlist, recovery must be implemented as a control-plane playlist reload, not as encoded silence.

## Test Strategy

Implementation follows test-driven development.

### Encoder tests

- A fresh encoder produces a valid playlist from only the bounded bootstrap carrier.
- Segment count and submitted media time stop advancing during an idle interval.
- PCM appended after an idle interval resumes segment publication without restarting FFmpeg.
- The second append receipt follows the first append receipt/media reservation without including elapsed wall-clock idle time.
- PCM queued during bootstrap starts after the reserved bootstrap region.
- Pending translated-audio duration excludes bootstrap and drains correctly.

### Shared publisher and protocol tests

- Multiple listener leases receive the same underlying playlist media sequence and segment bytes.
- Adding a second or third listener does not create another encoder or TTS synthesis.
- Captions for sentences separated by wall-clock translation delay remain contiguous on the compressed media timeline apart from synthesized edges and the configured sentence pause.
- The status endpoint remains healthy while the playlist is intentionally idle.

### Browser tests

- The Listen page contains no historical-silence seek or device-specific next-speech buffer guard.
- `waiting` at the live edge keeps the listener session running and resumes on `playing` without resetting the source.
- Existing Auto rate thresholds, manual rates, captions, Start/Stop, and lease cleanup continue to pass.
- Browser regression tests cover repeated idle/resume cycles without sentence-start interruption.

### Production verification

- Run the full automated test suite before deployment.
- Restart only `voxbridge-8024.service` after code deployment and verify port 8024, service state, encoder state, and empty error status.
- Run iPhone native HLS and Windows Chrome/HLS.js concurrently through several translation gaps.
- Verify one FFmpeg process and the expected listener count.
- Inspect the shared HLS audio for absence of multi-second idle carrier spans between released sentences.
- Confirm each new released sentence begins after at most normal segment/network buffering and does not need a client seek.

## Rollout and Rollback

The change creates a new HLS epoch when the service restarts, so existing carrier-filled segments cannot leak into the new behavior. No persistent data migration or API change is required.

Rollback is a normal code rollback plus service restart. The previous encoder will again generate the continuous idle carrier, and the previous client compaction logic must be restored in the same rollback commit; server and listener behavior must not be mixed across revisions.

## Acceptance Criteria

- With iPhone and Windows Chrome listening concurrently, both consume one encoder epoch and neither listener increases synthesis count.
- Translation-free wall-clock time does not increase the encoded HLS media duration after bootstrap.
- Once a stable translated sentence is published, audio becomes available as soon as HLS can complete the next slice; there is no additional two- or four-second next-speech guard.
- No client performs a historical-silence seek.
- Both devices hear the same sentence sequence and shared encoded pauses, with only normal network/segment timing variation.
- Existing Auto rate tiers remain `<10s: 1.0x`, `<30s: 1.2x`, `<40s: 1.4x`, and `>=40s: 1.5x`.
- No sentence-start interruption regression appears during repeated idle/resume cycles.
