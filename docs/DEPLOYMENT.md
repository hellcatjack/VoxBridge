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

After a translation backend is configured and verified, append these TTS flags
to the same `ExecStart` command:

```text
--enable-tts
--tts-en-model-path models/kokoro/kokoro-v1.0.onnx
--tts-en-voices-path models/kokoro/voices-v1.0.bin
--tts-zh-model-path models/kokoro/kokoro-v1.1-zh.onnx
--tts-zh-voices-path models/kokoro/voices-v1.1-zh.bin
--tts-zh-vocab-path models/kokoro/config-v1.1-zh.json
--tts-cpu-threads 4
--tts-listener-queue-size 128
```

Translation endpoints and model names are deployment-specific. Keep API keys outside the unit file.
The subtitle page never plays TTS. Each authenticated listener opens `/listen`
on a phone, tablet, or other browser and explicitly selects Start on that device.
Listeners receive future stable translations only; joining does not replay old
jobs. Device-local Stop clears only that browser's FIFO and does not affect other
listeners. One WAV is synthesized and cached per translation under the global
CPU lock, then shared by all listeners assigned to that job.

`--tts-listener-queue-size` bounds unread metadata per device. A listener that
cannot keep up is disconnected without delaying other listeners. The shared job
registry is bounded by `--tts-max-client-jobs` and jobs expire after
`--tts-job-ttl-sec`; unread jobs are never silently evicted to admit newer work.
`--tts-final-translation-drain-sec` controls the slow-drain warning threshold and
does not discard pending stable translations. When capturing system audio, use a
separate listener device or headphones to avoid feeding synthesized speech back
into ASR.

Load and start the service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now voxbridge-8024.service
systemctl --user status voxbridge-8024.service --no-pager -l
ss -lntp | rg ':8024'
```

Only one managed backend should own port `8024`. Do not start a second manual backend beside the user service.

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
systemctl --user show voxbridge-8024.service \
  -p MainPID -p NRestarts -p MemoryCurrent --no-pager
ss -lntp | rg ':8024'
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
- Monitor CPU load and process RSS after enabling Kokoro. TTS synthesis is globally serialized; high backlog degrades by increasing spoken delay rather than starting concurrent CPU model runs.
- Treat logs, audio, translations, subtitle traces, and screenshots as sensitive meeting data.
- Keep credentials in mode `0600` runtime files or a dedicated secret manager.
- Keep the service on port `8024`; changing the public proxy port does not change the backend contract.
