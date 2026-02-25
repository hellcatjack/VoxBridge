# AGENTS Instructions

## Critical Runtime Rule
- 本地启动服务时，必须使用端口 `8024`。
- 禁止改为 `8000/8001/8080` 等其它端口，除非用户明确要求。

## Local Environment
- 系统 Python 环境固定为项目内 `.venv`（即 `.venv/bin/python`）。
- 所有 Python 命令默认通过 `.venv` 执行，禁止回退到系统全局 Python。
- 启动后需确认 `8024` 端口处于监听状态（`ss -lntp | rg ':8024'`）。

## Service Start Baseline
- 推荐启动方式：`python -m voxbridge.cli.demo_streaming_ws --port 8024`
- 若已有旧进程占用，先清理旧进程，再重启到 `8024`。
