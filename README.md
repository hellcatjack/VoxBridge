# VoxBridge

VoxBridge 是一个独立整理后的实时语音识别与双语字幕系统，提供 Qwen3-ASR 流式识别、OpenAI 兼容翻译后端和浏览器字幕界面。

## Runtime Rules

- 本地服务端口固定为 `8024`。
- Python 环境固定使用上级项目虚拟环境：`../.venv/bin/python`。
- 启动后必须确认 `8024` 已监听：`ss -lntp | rg ':8024'`。
- 不要在代码或文档中提交 API Token、会议日志或字幕 trace 原文。

## Features

- WebSocket 流式 ASR，前端持续推送音频 chunk，后端负责 streaming state 和切段逻辑。
- 双语字幕界面：英文区域占上方 `2/3`，中文区域占下方 `1/3`。
- 最近 100 条字幕可滚动查询；默认自动跟随最新字幕，用户手动滚动时暂停自动滚动，并提供“最新”按钮回到底部。
- 护眼浅色 UI；开始识别后顶部控制栏自动隐藏，右上角保留“控制”按钮用于临时展开。
- 支持麦克风输入和浏览器整屏共享系统音频输入。
- 支持翻译方向选择：
  - `中文 -> 英文`：ASR 强制 `Chinese`，翻译源语言为中文，目标语言为英文。
  - `英文 -> 中文`：ASR 强制 `English`，翻译源语言为英文，目标语言为中文。
  - 翻译方向需要在开始前选择；会话运行中选择框会锁定，避免 ASR 语言和翻译方向半切换。

## Layout

- `voxbridge/cli/demo_streaming_ws.py`：前后端一体的 WebSocket 流式服务入口。
- `voxbridge/streaming/`：切片策略、背压、文本池逻辑。
- `tools/subtitle_ws_selfcheck.py`：WebSocket 后端自测工具。
- `tests/`：核心逻辑与前端模板协议测试。
- `docs/SECURITY_SCAN.md`：信息泄露扫描记录。

## Install

```bash
cd /data/Qwen3-ASR/VoxBridge
../.venv/bin/python -m pip install -e .
```

## Start On Port 8024

最小启动示例：

```bash
cd /data/Qwen3-ASR/VoxBridge
../.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-0.6B \
  --backend vllm \
  --port 8024
```

当前推荐通过用户级 systemd 服务管理：

```bash
systemctl --user status voxbridge-8024.service --no-pager -l
systemctl --user restart voxbridge-8024.service
ss -lntp | rg ':8024'
```

## Translation Backend

翻译后端使用 OpenAI 兼容 API，通过启动参数注入：

```bash
--enable-translation \
--translation-backend openai_api \
--translation-api-base-url http://127.0.0.1:8001 \
--translation-api-model <model-name> \
--translation-api-key <token>
```

`--translation-api-key` 可选；如果本地兼容 API 不需要鉴权，可以不传。不要把真实 Token 写入仓库。

## Tests

单元测试：

```bash
cd /data/Qwen3-ASR/VoxBridge
PYTHONPATH=. ../.venv/bin/python -m pytest -q tests
```

后端链路自测：

```bash
cd /data/Qwen3-ASR/VoxBridge
PYTHONPATH=. ../.venv/bin/python tools/subtitle_ws_selfcheck.py \
  --ws-url ws://127.0.0.1:8024/ws \
  --wav /data/Qwen3-ASR/audios/repeat22_16k.wav
```

浏览器自测建议使用 Playwright 打开 `http://127.0.0.1:8024`，重点确认：

- `中文 -> 英文` 启动 payload 使用 `language: "Chinese"`。
- `英文 -> 中文` 启动 payload 使用 `language: "English"`。
- 窄屏下最新原文和翻译字幕仍自动显示在底部。
- 运行中控制栏自动隐藏，点击“控制”可临时展开。

## Security

提交前运行 `docs/SECURITY_SCAN.md` 中的扫描命令。会议音频、字幕 trace、日志和临时截图可能包含敏感内容，不应提交。
