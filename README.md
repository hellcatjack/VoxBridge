# VoxBridge

VoxBridge 是一个独立整理后的实时语音识别与双语字幕系统（ASR + Translation + WebSocket UI）。

## 运行约束

- 本地服务端口固定为 `8024`。
- 本仓库环境固定使用上级目录 `.venv`：`../.venv/bin/python`。

## 目录结构

- `voxbridge/cli/demo_streaming_ws.py`：前后端一体的 WebSocket 流式服务入口
- `voxbridge/streaming/`：切片策略、背压、文本池逻辑
- `tools/subtitle_ws_selfcheck.py`：后端自测工具
- `tests/`：核心逻辑与协议测试
- `docs/SECURITY_SCAN.md`：信息泄露扫描记录

## 安装

```bash
cd /data/Qwen3-ASR/VoxBridge
../.venv/bin/python -m pip install -e .
```

## 启动（8024）

```bash
cd /data/Qwen3-ASR/VoxBridge
../.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --backend vllm \
  --port 8024
```

## 端口确认

```bash
ss -lntp | rg ':8024'
```

## 自测

后端链路自测（识别 + 翻译事件）：

```bash
cd /data/Qwen3-ASR/VoxBridge
PYTHONPATH=. ../.venv/bin/python tools/subtitle_ws_selfcheck.py \
  --ws-url ws://127.0.0.1:8024/ws \
  --wav /data/Qwen3-ASR/audios/repeat22_16k.wav
```

单元测试：

```bash
cd /data/Qwen3-ASR/VoxBridge
PYTHONPATH=. ../.venv/bin/python -m pytest -q tests
```

## 翻译后端配置

默认翻译后端可使用 OpenAI 兼容 API，通过参数传入：

- `--translation-backend openai_api`
- `--translation-api-base-url http://127.0.0.1:8001`
- `--translation-api-model <model-name>`
- `--translation-api-key <token>`（可选）

建议不要在代码中硬编码 API Token，统一使用启动参数或部署环境注入。
