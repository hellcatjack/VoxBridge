# Security Scan Report

Date: 2026-02-25
Scope: `VoxBridge/` source and docs (excluding `.git` and binary audio/image artifacts)

## Scan Commands

```bash
cd /data/Qwen3-ASR/VoxBridge

# Secret patterns
rg -n --hidden --glob '!.git' -i \
  '(api[_-]?key|secret|token|password|authorization|bearer|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})'

# Private network / endpoint exposure
rg -n --hidden --glob '!.git' -i \
  '(192\\.168\\.|10\\.[0-9]+\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.|localhost:[0-9]{2,5}|0\\.0\\.0\\.0:[0-9]{2,5})'

# Email addresses
rg -n --hidden --glob '!.git' -i \
  '[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}'
```

## Findings

1. No leaked API keys, private keys, or credential files were detected.
2. One hardcoded private network default endpoint was found and fixed:
   - `voxbridge/cli/demo_streaming_ws.py`
   - changed from `http://192.168.1.31:8001` to `http://127.0.0.1:8001`
3. Author email exists in package metadata (`pyproject.toml`) and is treated as intentional public maintainer info.

## Hardening Checklist

- Keep `.env`, logs, and temporary traces out of version control.
- Inject API token at runtime via CLI/env; do not commit real token values.
- Re-run this scan before each release tag.
- If subtitle trace logs include meeting content, treat them as sensitive business data.
