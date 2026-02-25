# VoxBridge

VoxBridge is an extracted, standalone real-time subtitle system built on top of Qwen3-ASR.

## What Is Included

- WebSocket streaming ASR backend
- Sentence-level subtitle pool and rolling window logic
- Real-time bilingual translation pipeline
- Backpressure and segment policy utilities
- Self-check and trace tools for subtitle quality debugging

## Runtime Constraints

- Service port: `8024`
- Python interpreter: project-local `.venv` (`.venv/bin/python`)

## Quick Start

```bash
cd VoxBridge

# Install dependencies into local .venv
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

Start service:

```bash
.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --backend vllm \
  --port 8024
```

Verify listen status:

```bash
ss -lntp | rg ':8024'
```

## Tests

```bash
cd VoxBridge
PYTHONPATH=. ../.venv/bin/python -m pytest -q tests
```
