# AGENTS Instructions

## Critical Runtime Rule
- 本地启动服务时，必须使用端口 `8024`。
- 禁止改为 `8000/8001/8080` 等其它端口，除非用户明确要求。

## Local Environment
- 本子项目位于 `/data/Qwen3-ASR/VoxBridge`。
- 系统 Python 环境固定为上级项目 `.venv`（即 `../.venv/bin/python`，对应 `/data/Qwen3-ASR/.venv/bin/python`）。
- 所有 Python 命令默认通过 `../.venv/bin/python` 执行，禁止回退到系统全局 Python。
- 启动后需确认 `8024` 端口处于监听状态（`ss -lntp | rg ':8024'`）。

## Service Start Baseline
- 推荐启动方式：`../.venv/bin/python -m voxbridge.cli.demo_streaming_ws --port 8024`
- 线上本机推荐使用用户级服务：`voxbridge-8024.service`。
- 若已有旧进程占用，先清理旧进程，再重启到 `8024`。

## UI And Direction Notes
- 翻译方向默认 `中文 -> 英文`。
- `英文 -> 中文` 会在启动时强制 ASR language 为 `English`；`中文 -> 英文` 会强制 ASR language 为 `Chinese`。
- 翻译方向必须在开始前选择，运行中下拉框会被锁定。
- 前端为护眼浅色主题，开始识别后顶部控制栏会自动隐藏。
