# Deployment

This guide describes a public-facing VoxBridge deployment while keeping the application itself on the fixed local port `8024`.

## 1. Prerequisites

- Linux and Python `>=3.10`.
- A GPU runtime supported by Qwen3-ASR and vLLM.
- Torch plus ROCm or CUDA versions selected for the target accelerator.
- Optional OpenAI-compatible translation API.
- HTTPS reverse proxy for untrusted networks.

Install accelerator-specific packages according to the hardware vendor and upstream Qwen3-ASR/vLLM documentation. Do not copy a Torch or Triton index intended for different hardware.

## 2. Install in `.venv`

```bash
git clone https://github.com/hellcatjack/VoxBridge.git
cd VoxBridge
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

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

Translation endpoints and model names are deployment-specific. Keep API keys outside the unit file.

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
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
systemctl --user restart voxbridge-8024.service
ss -lntp | rg ':8024'
```

For rollback, select a previously verified commit or annotated tag, reinstall the editable package, run tests, and restart once. Never run multiple release processes concurrently to compare them on the same GPU.

## 8. Operational safety

- Prefer graceful WebSocket `finish` and systemd stop before replacement.
- Confirm old backend and EngineCore PIDs exit before starting another model process.
- Monitor `NRestarts`, process RSS, GPU memory, GTT, queue depth, and queue-drop trace fields during long sessions.
- Treat logs, audio, translations, subtitle traces, and screenshots as sensitive meeting data.
- Keep credentials in mode `0600` runtime files or a dedicated secret manager.
- Keep the service on port `8024`; changing the public proxy port does not change the backend contract.
