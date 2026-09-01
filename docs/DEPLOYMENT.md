# Deployment

This guide describes a public-facing VoxBridge deployment while keeping the application itself on the fixed local port `8024`.

## 1. Prerequisites

- Linux, Python `>=3.10`, and `uv`.
- A GPU runtime supported by Qwen3-ASR and vLLM.
- Torch plus ROCm or CUDA versions selected for the target accelerator.
- Optional OpenAI-compatible translation API.
- HTTPS reverse proxy for untrusted networks.

Install accelerator-specific packages according to the hardware vendor and upstream Qwen3-ASR/vLLM documentation. Do not copy a Torch or Triton index intended for different hardware.

## 2. Install in `.venv`

```bash
git clone https://github.com/hellcatjack/VoxBridge.git
cd VoxBridge
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e .
```

For optional Kokoro translated speech, install the isolated CPU TTS extra after
the accelerator stack is already correct:

```bash
uv pip install --dry-run --python .venv/bin/python -e '.[tts]'
uv pip install --python .venv/bin/python -e '.[tts]'
```

The dry-run must not uninstall or replace Torch, Triton, ROCm/CUDA, or install a
CUDA-flavored Torch wheel. The TTS extra uses ONNX Runtime
`CPUExecutionProvider`; it does not share vLLM GPU memory.

For observation-only Silero VAD on an already validated ROCm/CUDA environment,
install the package without dependency resolution:

```bash
uv pip install --python .venv/bin/python --no-deps 'silero-vad==6.2.1'
```

Do not install `.[vad-shadow]` on a production accelerator environment without
first inspecting a dry run. A generic resolver can replace the locally matched
Torch and Triton builds. Verify those versions remain unchanged after the
no-dependency install.

Create `models/kokoro/` and obtain these assets from the official
`thewh1teagle/kokoro-onnx` releases and `hexgrad/Kokoro-82M-v1.1-zh` model page:

```text
models/kokoro/kokoro-v1.0.onnx
models/kokoro/voices-v1.0.bin
models/kokoro/kokoro-v1.1-zh.onnx
models/kokoro/voices-v1.1-zh.bin
models/kokoro/config-v1.1-zh.json
```

Download to a `.part` file, verify the published size/hash, and atomically
rename it. Model binaries remain outside Git.

Confirm that the selected environment imports the installed package:

```bash
.venv/bin/python -c 'import voxbridge; print(voxbridge.__file__)'
```

When maintaining VoxBridge inside an existing Qwen3-ASR checkout, use that checkout's `../.venv/bin/python` instead of creating a second environment.

## 3. Generate the login password hash

Generate the hash interactively so the password is not embedded in a command:

```bash
.venv/bin/python - <<'PY'
from getpass import getpass
from voxbridge.cli.demo_streaming_ws import _hash_auth_password

print(_hash_auth_password(getpass("VoxBridge password: ")))
PY
```

Create the runtime environment file and restrict it to the current user:

```bash
mkdir -p ~/.config/voxbridge
chmod 700 ~/.config/voxbridge
printf '%s\n' 'VOXBRIDGE_AUTH_PASSWORD_HASH=<generated-pbkdf2-hash>' \
  > ~/.config/voxbridge/voxbridge.env
chmod 600 ~/.config/voxbridge/voxbridge.env
```

Replace the placeholder locally. Never commit this file or its value.

## 4. Configure a user-level service

Install the repository under `~/src/VoxBridge`, then create `~/.config/systemd/user/voxbridge-8024.service`:

```ini
[Unit]
Description=VoxBridge ASR and translation service on port 8024
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/src/VoxBridge
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-%h/.config/voxbridge/voxbridge.env
ExecStart=%h/src/VoxBridge/.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-0.6B \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8024 \
  --mm-processor-cache-gb 0 \
  --segment-final-redecode \
  --auth-enabled \
  --auth-username admin \
  --auth-cookie-secure \
  --disable-debug-file
Restart=on-failure
RestartSec=3
TimeoutStopSec=45
KillSignal=SIGINT

[Install]
WantedBy=default.target
```

Add translation, VAD, context schedule, queue, or model-memory flags to `ExecStart` only after validating them with:

```bash
.venv/bin/python -m voxbridge.cli.demo_streaming_ws --help
```

### Couple the translation API lifecycle

The production host can make the existing Q8_0 llama.cpp translation server a
managed dependency of VoxBridge. Install the tracked translation unit,
readiness probe, and main-service drop-in:

```bash
mkdir -p ~/.local/libexec ~/.config/systemd/user/voxbridge-8024.service.d
install -m 755 deploy/systemd/voxbridge-wait-translation.sh \
  ~/.local/libexec/voxbridge-wait-translation
install -m 644 deploy/systemd/voxbridge-translation.service \
  ~/.config/systemd/user/voxbridge-translation.service
install -m 644 deploy/systemd/voxbridge-8024-translation.conf \
  ~/.config/systemd/user/voxbridge-8024.service.d/translation.conf
systemctl --user daemon-reload
```

`voxbridge-translation.service` invokes
`/app/llama.cpp/llama-server-normal.sh start`, tracks its PID, and waits until
`GET http://127.0.0.1:8001/v1/models` exposes
`tencent/HY-MT1.5-1.8B-GGUF:Q8_0`. The public listener remains
`0.0.0.0:8001`; the loopback URL is only the local readiness probe.
VoxBridge starts only after that check succeeds. The reverse stop ordering
releases WebSocket, ASR, translation, and TTS work before invoking
`/app/llama.cpp/llama-server-normal.sh stop`.

Do not enable or invoke the translation service separately after installing
this coupling. Manage the stack through the existing VoxBridge unit:

```bash
systemctl --user start voxbridge-8024.service
systemctl --user stop voxbridge-8024.service
systemctl --user restart voxbridge-8024.service
```

The translation unit uses `PartOf=voxbridge-8024.service`, while the main
service drop-in uses `Requires=` and `After=`. Therefore start, stop, and
restart operations propagate without putting a detached llama-server in
`ExecStartPre`.

Install the tracked cgroup guard as a drop-in after measuring the normal cold
start and active-session peak on the target host:

```bash
mkdir -p ~/.config/systemd/user/voxbridge-8024.service.d
cp deploy/systemd/voxbridge-8024-memory.conf \
  ~/.config/systemd/user/voxbridge-8024.service.d/memory.conf
systemctl --user daemon-reload
```

The production single-engine profile uses these deliberately loose guards:

```ini
MemoryAccounting=yes
MemoryHigh=16G
MemoryMax=20G
TasksMax=512
OOMPolicy=stop
```

`MemoryHigh` applies reclaim pressure before the hard `MemoryMax` boundary.
`TasksMax` leaves headroom above the measured single-engine thread count while
preventing a second full EngineCore from silently joining the same service.
These values cover only the `voxbridge-8024.service` cgroup; an external OpenAI-compatible translation service has its own process and memory budget.
Do not lower either memory boundary without a full real-time soak.

Inspect the effective cgroup after restart:

```bash
cg="$(systemctl --user show voxbridge-8024.service -p ControlGroup --value)"
systemctl --user show voxbridge-8024.service \
  -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p TasksCurrent -p TasksMax
cat "/sys/fs/cgroup${cg}/memory.current"
cat "/sys/fs/cgroup${cg}/memory.peak"
cat "/sys/fs/cgroup${cg}/memory.events"
cat "/sys/fs/cgroup${cg}/pids.current"
```

To roll back only the guard without touching the primary unit, move the drop-in
aside and reload systemd:

```bash
mkdir -p ~/.config/voxbridge/disabled-systemd
mv ~/.config/systemd/user/voxbridge-8024.service.d/memory.conf \
  ~/.config/voxbridge/disabled-systemd/voxbridge-8024-memory.conf
systemctl --user daemon-reload
systemctl --user restart voxbridge-8024.service
```

The recommended quality-preserving decode gate instrumentation is:

```text
--silent-decode-pre-roll-sec 0.4
--silero-vad-shadow
--silero-vad-rescue
--silero-vad-shadow-threshold 0.5
--silero-vad-shadow-log-sec 1.0
```

The pre-roll contains only audio skipped by the current energy gate. It is
replayed once when decoding resumes, or decoded once immediately before an ASR
segment is finalized so a weak final syllable cannot be discarded. With
`--silero-vad-rescue`, accumulated Silero speech evidence may prevent such a
batch from being skipped, including a coalesced batch that starts with speech
but ends in silence. The rescue permits decoding only: it preserves any
energy-VAD silence endpoint and cannot itself trigger a cut or commit text. Loading and
inference failures disable only the Silero observer for the current WebSocket
session.

`--segment-final-redecode` runs one bounded one-shot decode at natural VAD
endpoints before source sentences are committed, translated, or released to
TTS. A mid-speech hard cut uses the streaming flush result and rotates
immediately; re-decoding a long active segment would block inference and can
overflow the live audio queue. A hard cut remains mid-speech until trailing
silence reaches `--vad-silence-sec`; this uses the deployed VAD endpoint rather
than an independent fixed threshold. Empty, failed, or substantially divergent VAD
results, including corrections that reduce complete units after the previous
segment's pending prefix is carried forward, leave the existing revision
stability window active instead of sealing the source immediately. While this
option is enabled, the newest unsealed
source has no timer-only TTS release: a validated segment seal or a
rollback-safe successor is required. Stop flushes the current streaming state
and force-drains the final ready revision without rebuilding visible history.
Full-session Stop re-decode is an explicit opt-in via
`--final-redecode-on-stop`.

For one continuous microphone or system-audio stream, use the measured
real-time rotation and backpressure budget:

```text
--segment-hard-cut-sec 45
--backpressure-target-queue-sec 3.0
--backpressure-max-queue-sec 15.0
--backpressure-hard-relief-sec 6.0
--subtitle-snapshot-history-size 100
```

The soft threshold increases consumer batch size without discarding PCM. At the
hard threshold, the backend retains the current frame in bounded spill storage
and pauses WebSocket ingress until the independent consumer recovers; TCP/WSS
transport backpressure replaces oldest-frame deletion and requires no browser
decision logic. `--backpressure-hard-relief-sec` is a deprecated no-op retained
for existing service commands. Compatibility
snapshots sent with each `partial`/`final` are bounded to the latest 100
solidified rows; sentence events and canonical backend state remain complete.
A hard cut always flushes and rotates the streaming state and never runs the
blocking full-segment re-decode; natural VAD endpoints retain final re-decode
for tail accuracy. Keep the bounded queue and sustained-overload fallback: if
average inference remains slower than real time even with 45-second rotation,
the deployment needs more compute rather than an unbounded audio buffer.

With streaming context enabled, long silence arms a backend output quarantine.
The quarantine still accepts a compatible sentence-tail repair when the segment
already contains sufficient backend speech activity and context-echo checks
pass. A state that has decoded no audio skips streaming finish entirely, so
context hotwords cannot be generated from a control-only silent session.

Browser clients additionally hold possible silence for 700 ms. A shorter pause
is replayed as ordinary PCM; a confirmed longer pause becomes an ordered
`audio_silence` control event with one-second heartbeats. The first 400 ms of a
confirmed endpoint is retained as bounded low-energy tail PCM; the remaining
silence is suppressed. The backend advances VAD and context-schedule time from
control events without converting those events to PCM. Speech recovery sends up
to 400 ms pre-roll after `audio_speech_start`.

On macOS, use HTTPS/WSS. Safari 14.1+ supports the AudioWorklet microphone path,
and the page retains a ScriptProcessor fallback for older or restricted WebKit
environments. System-audio capture is not portable across Safari/macOS versions;
Chrome or Edge is recommended when a browser tab or another application must be
captured. The page requires a real audio track from `getDisplayMedia` and fails
the Start operation when the browser returns video-only sharing.

Long comma-delimited speech is translated in rollback-safe clause units without
rotating the ASR state. The defaults are:

```text
--stable-clause-target-cjk-chars 32
--stable-clause-target-latin-words 24
```

These are target sizes rather than hard character wrapping limits. Only the
first comma, semicolon, or colon at or after the target is eligible, making the
boundary prefix-deterministic. The newest unit remains tentative, every clause
must cross the tokenizer rollback window, and `0` disables the corresponding
script-aware splitter.

After a translation backend is configured and verified, append these TTS flags
to the same `ExecStart` command:

```text
--enable-tts
--tts-en-model-path models/kokoro/kokoro-v1.0.onnx
--tts-en-voices-path models/kokoro/voices-v1.0.bin
--tts-zh-model-path models/kokoro/kokoro-v1.1-zh.onnx
--tts-zh-voices-path models/kokoro/voices-v1.1-zh.bin
--tts-zh-vocab-path models/kokoro/config-v1.1-zh.json
--tts-speed 1.05
--tts-cpu-threads 4
--tts-listener-queue-size 128
--tts-hls-max-listeners 128
--tts-revision-stable-sec 3.0
--tts-latest-revision-grace-sec 4.0
```

Translation endpoints and model names are deployment-specific. Keep API keys outside the unit file.
The shared listener also requires `ffmpeg` on `PATH`; startup of the main service
does not spawn FFmpeg until the first public HLS listener joins.

The subtitle page never plays TTS. Each listener opens the public `/listen`
page on a phone, tablet, or other browser and explicitly selects Start on that
device. The production main page displays a static local QR for the fixed URL
`https://ushome.amycat.com:18024/listen`. The PCCS listener is English-only and
uses a fixed one-screen layout without document scrollbars.
Safari uses native HLS so iPhone lock-screen playback remains independent of
foreground JavaScript. Desktop Chrome, Edge, and Firefox use the vendored hls.js
MSE fallback from `/listen/assets/hls.min.js` when AAC MSE is available. The
server's idle carrier remains acoustically silent but forces FFmpeg to include
decodable AAC frames instead of producing table-only MPEG-TS segments.
The pinned build and license live under `voxbridge/tts/vendor/`; package builds
must include those files. Do not replace it with a runtime CDN dependency.
The backend keeps a bounded pre-listener pool of up to 128 stable translations
from the current producer session. With no listener, it performs no TTS work and
reports zero active speech backlog. When the first listener creates a new live
epoch, the backend discards stale entries, retains only the latest stable
translation, and synthesizes that join sentence at displayed `1.0x`; this is a
live-edge join buffer, not a catch-up queue or persisted recording. A new
producer session clears an older idle pool. Device-local Stop removes only that
browser's lease and does not affect other listeners. The listener that triggered
the epoch has no special ownership: as long as another lease remains, the
encoder, shared queue, speech epoch, and global speed controller continue
unchanged. Removing or expiring the final lease clears all epoch-specific state.
One worker synthesizes each translation once, and one FFmpeg
process writes a continuous 24 kHz mono AAC/HLS timeline shared by all devices.
When no speech is pending, the encoder receives real-time zero PCM so native iOS
playback can continue after the browser is locked.

`--tts-speed` is the backend Kokoro synthesis baseline and remains global for
displayed `1.0x`. The default global Auto controller multiplies that baseline
for each new sentence before Kokoro synthesis. With the recommended baseline
`1.05`, displayed `1.2x`, `1.4x`, and `1.5x` map to absolute Kokoro speeds
`1.26`, `1.47`, and `1.575`. The selected value is fixed for the full sentence;
already synthesized or published PCM is never retimed. Every `/listen` media
element remains at HTML playback rate `1.0`, so iPhone Safari and desktop
Chrome consume the same server-paced shared stream.

After genuine PCM queue starvation, the encoder may publish active PCM at
`2.0x` for one bounded two-second media burst, then it sustains real-time `1.0x`
publication. Adjacent queued sentences share that burst budget; only another
real empty-queue wait resets it. Tail-finalization carrier remains accelerated,
but it is not speech debt. This keeps long unpublished PCM visible to the
unchanged Auto tiers instead of hiding it in HLS faster than listeners consume
the timeline.

To roll back only the adaptive controller, add
`--disable-tts-global-auto-speed`. This keeps server synthesis fixed at
`--tts-speed` and keeps every browser at media rate `1.0`; it does not restore
the removed per-device rate selector.

The publisher appends a fixed `300ms` zero-PCM sentence pause after each Kokoro
clip. The global translated-speech queue uses `--tts-listener-queue-size` as its
bound, and the FFmpeg PCM handoff has a separate small bound. If encoding falls
behind, this applies backpressure rather than accumulating unbounded audio in
memory. The default HLS playlist retains at most 1200 one-second segments, and
the last listener's 90-second lease expiry closes FFmpeg and removes all files
for that epoch.

`GET /api/tts/live/status` reports `queue_depth` including the item currently in
Kokoro, exposes `synthesis_active`, and reports synthesized FIFO backlog as
`pending_audio_ms`. It also reports conservative de-duplicated unpublished
speech as `translated_audio_backlog_ms`; use
`translated_audio_backlog_count` and `translated_audio_backlog_estimated` to
distinguish queued revisions and estimated durations. The estimate uses the
larger of the language default or recently measured baseline speech time per
character with a 10% safety margin. It excludes device HLS lag, network delay,
and already published media.

At sentence synthesis start, global Auto selects `<6s = 1.0x`,
`6-<15s = 1.2x`, `15-<20s = 1.4x`, or `>=20s = 1.5x` from that conservative
server backlog. The response exposes `speech_epoch_id`, `global_speed_mode`,
`global_speed_multiplier`, and `tts_effective_speed`. A phone that is buffering
cannot change these values. When there are no listeners, the active backlog and
count are zero, the epoch ID is empty, the multiplier is `1.0`, and neither
Kokoro nor FFmpeg is started by status polling.

The publisher also retains at most 256 caption cues for audio already released
to the current HLS encoder epoch. Each cue uses the PCM media timeline submitted
to FFmpeg, includes the AAC 1024-sample presentation delay, removes synthesized
edge silence by waveform activity, and excludes the fixed 300ms sentence pause.
Each cue also reports `discardable_gap_before_ms` for only the wait-generated
tail carrier and `resume_at_ms` for the point that preserves the normal natural
gap before the next sentence. The browser requires that resume point and one
second beyond the next speech start to be in the same buffered range, then makes
one one-shot seek per cue. It does not pause, call play, schedule a retry, or
change the media rate as part of compaction.
`/listen` polls the listener-scoped caption snapshot only while the page is
visible. Safari maps its native media timeline with `getStartDate() +
currentTime`, so a stale device playlist or local network buffer does not move
captions ahead of audible speech. The server live-edge calculation is
only a fallback for media implementations without a valid start date. A failed
caption request must not stop or restart HLS playback. Monitor caption endpoint
errors separately from FFmpeg and audio backlog; a caption failure is a display
degradation, not an audio-stream outage.

With an active listener, translation completion also queues bounded speculative
Kokoro preparation for the exact sentence revision. It does not append PCM to
HLS or bypass `--tts-revision-stable-sec`; stable ordered release consumes the
prepared result only when sentence ID, revision, target language, and text digest
still match. A revision change discards stale audio. Prepared PCM also records
the displayed multiplier and effective Kokoro speed selected when synthesis
began; stable release reuses that exact PCM even if the current Auto tier is now
lower. The fixed eight-entry
cache and the stable queue share one Kokoro worker, and the cache is removed
when the listener epoch ends. Monitor `preparation_queue_depth`,
`preparation_active`, and `prepared_audio_count` separately from `queue_depth`
and `pending_audio_ms`.

`--tts-hls-max-listeners 128` bounds active public leases. Each browser generates
an opaque random listener ID; possession of that ID is a public bearer capability
that can refresh or delete only its matching lease. A new ID above the bound gets
`HTTP 429`, while established listeners continue to refresh. This limit bounds
lease bookkeeping; all devices still share one Kokoro worker and one FFmpeg.

The public listener intentionally does not require a login so a fixed church QR
works on guest phones. The main subtitle page remains authenticated, as do the
ASR WebSocket and management/compatibility APIs. Keep HTTPS enabled, do not put
private meeting content into the QR, and treat the live translated audio as
public to anyone who has the address.

The old `/ws/tts` plus per-job WAV API remains available for one compatibility
cycle but is not used by `/listen`. Do not run old and new listener clients at
the same time when measuring TTS CPU, because a legacy WAV request is a separate
compatibility delivery path.

The legacy shared job registry is bounded by `--tts-max-client-jobs` and jobs expire after
`--tts-job-ttl-sec`; unread jobs are never silently evicted to admit newer work.
`--tts-final-translation-drain-sec` controls the slow-drain warning threshold and
does not discard pending stable translations. When capturing system audio, use a
separate listener device or headphones to avoid feeding synthesized speech back
into ASR.

`--tts-revision-stable-sec` delays spoken publication from the latest source
sentence revision while leaving visible subtitles responsive. The recommended
production value is `3.0`. `--tts-latest-revision-grace-sec 4.0` applies only
to the newest unsealed source. A successor returns its predecessor to the
ordinary window. With `--segment-final-redecode`, only a successful unchanged
or safely compatible segment result seals the segment so ready translations can
publish without an arbitrary timer. A failed or divergent result leaves normal
revision timing active for older sources and keeps the newest source waiting for
a seal or rollback-safe successor. Do not replace
this source-order policy with a global seven-second delay. Tune it only from observed
`tts_late_revision_after_release` timing; do not add punctuation,
language-specific word lists, or frontend timers to infer speech stability.

vLLM may allocate the multimodal processor cache in both the API and EngineCore
processes. Use `--mm-processor-cache-gb 0` for the single-microphone,
single-connection streaming profile because changing live audio has no useful
cross-request processor-cache reuse. A nonzero value can improve repeated
offline or multi-client inputs, but it must be justified with RSS, GTT, and
decode-latency measurements from that workload.

Load and start the service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now voxbridge-8024.service
systemctl --user status voxbridge-8024.service --no-pager -l
systemctl --user status voxbridge-translation.service --no-pager -l
ss -lntp | rg ':(8001|8024)'
```

Only one managed backend should own port `8024`, and only one managed
llama-server should own port `8001`. Do not start either backend manually
beside the user services.

### Install bounded user log rotation

The tracked template checks the service log and subtitle trace every hour,
rotates either file after 512 MiB, retains 21 compressed generations, and uses
`copytruncate` so the running Python process does not need to reopen a file
descriptor. Update the two log paths in `deploy/logrotate/voxbridge.conf` if the
runtime workspace differs from the deployment host.

```bash
mkdir -p ~/.config/voxbridge ~/.config/systemd/user ~/.local/state/voxbridge
cp deploy/logrotate/voxbridge.conf ~/.config/voxbridge/logrotate.conf
cp deploy/systemd/voxbridge-logrotate.service ~/.config/systemd/user/
cp deploy/systemd/voxbridge-logrotate.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voxbridge-logrotate.timer
systemctl --user list-timers voxbridge-logrotate.timer --no-pager
/usr/sbin/logrotate --debug \
  --state ~/.local/state/voxbridge/logrotate.status \
  ~/.config/voxbridge/logrotate.conf
```

For rollback, disable `voxbridge-logrotate.timer`, restore the previous
configuration if one existed, and run `systemctl --user daemon-reload`. Debug
mode parses and reports actions without rotating live logs.

## 5. Put HTTPS in front of port 8024

A generic Caddy configuration is:

```caddyfile
voxbridge.example.com {
    reverse_proxy 127.0.0.1:8024
}
```

Requirements:

- Keep VoxBridge bound to `127.0.0.1:8024`.
- Expose only the HTTPS reverse proxy to the untrusted network.
- Enable `--auth-cookie-secure` only when the browser uses HTTPS/WSS.
- Preserve WebSocket upgrades in the reverse proxy.
- Keep `--disable-debug-file` enabled.

## 6. Verify the deployment

Check systemd and the listener:

```bash
systemctl --user is-active voxbridge-8024.service
systemctl --user is-active voxbridge-translation.service
systemctl --user show voxbridge-8024.service \
  -p MainPID -p NRestarts -p MemoryCurrent --no-pager
ss -lntp | rg ':(8001|8024)'
curl --fail --silent http://127.0.0.1:8001/v1/models
```

Check process topology:

```bash
ps -eo pid,ppid,cmd | rg '[v]oxbridge\.cli\.demo_streaming_ws|VLLM::EngineCore'
```

A single-GPU deployment should normally show one VoxBridge backend and one EngineCore. Investigate duplicate processes before accepting traffic.

An unauthenticated local HTTP request should redirect to login:

```bash
curl -I http://127.0.0.1:8024/
```

Review logs without publishing subtitle content:

```bash
journalctl --user -u voxbridge-8024.service --since '-10 min' --no-pager
```

VoxBridge disables Uvicorn's raw HTTP access log because TTS audio paths contain
opaque job identifiers. Application TTS logs retain only short SHA-256
fingerprints and queue counts; do not replace this with a proxy access log that
records unredacted `/api/tts/` paths.

## 7. Upgrade and rollback

Before an upgrade, stop active browser sessions cleanly. Then:

```bash
cd ~/src/VoxBridge
git fetch --tags origin
git pull --ff-only origin main
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q
systemctl --user restart voxbridge-8024.service
ss -lntp | rg ':8024'
```

For rollback, select a previously verified commit or annotated tag, reinstall the editable package, run tests, and restart once. Never run multiple release processes concurrently to compare them on the same GPU.

## 8. Operational safety

- Prefer graceful WebSocket `finish` and systemd stop before replacement.
- Confirm old backend and EngineCore PIDs exit before starting another model process.
- Monitor `NRestarts`, process RSS, GPU memory, GTT, queue depth, and queue-drop trace fields during long sessions.
- Monitor CPU load, process RSS, HLS queue depth, `/tmp/voxbridge-tts-hls-8024`, and the single FFmpeg child after enabling Kokoro. Adding listeners must not add synthesis workers or FFmpeg processes; high backlog degrades by increasing spoken delay rather than duplicating CPU model runs.
- Compare `silero_shadow_observation` and `silero_shadow_disagreement` with `silent_decode_skipped`, `audio_preroll_replayed`, and subsequent ASR output before allowing a neural VAD to influence control decisions.
- Treat logs, audio, translations, subtitle traces, and screenshots as sensitive meeting data.
- Keep credentials in mode `0600` runtime files or a dedicated secret manager.
- Keep the service on port `8024`; changing the public proxy port does not change the backend contract.
