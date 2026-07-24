# VoxBridge

VoxBridge 是一个面向会议、新闻和演讲场景的实时语音识别与双语字幕服务。浏览器负责采集音频和显示字幕，后端负责 Qwen3-ASR 流式状态、切段、句子稳定性和翻译调度。

## 核心能力

- 支持麦克风输入，以及浏览器“整屏共享 + 系统音频”输入。
- 支持 `中文 -> 英文` 和 `英文 -> 中文`；启动后锁定方向，避免 ASR 语言与翻译方向漂移。
- 每个 WebSocket 会话维护独立的 streaming state；VAD、硬时长轮转、最终 flush 和重叠处理均由后端负责。
- 使用有界音频队列和背压机制，避免长时间会话无限积压。
- 后端通过 `sentence_id` 和 `revision` 明确区分新句、稳定修订和对应翻译，前端不使用固定词表猜测稳定性。
- 支持页面输入会话级专业术语 Context，也支持后端加载有界、分时段的 context schedule。
- 最近 100 条字幕可滚动查询；默认跟随最新内容，用户手动滚动时暂停自动跟随。
- 支持上下字幕字体调整、移动端布局、控制栏自动隐藏和返回最新字幕按钮。
- 可选单用户登录、Secure Cookie、结构化字幕 trace 和文本池诊断日志。

## 工作流程

```text
Browser PCM audio
  -> WebSocket /ws
  -> bounded audio queue
  -> backend VAD / hard segment rotation
  -> Qwen3-ASR streaming state
  -> sentence_id + revision
  -> OpenAI-compatible translation API
  -> bilingual subtitle rows
```

浏览器只负责持续发送 PCM 音频和渲染后端事件。生成中的 `partial` 可以变化；已固化句子通过 `sentence_committed` 创建，通过 `sentence_updated` 更新同一个 `sentence_id`。翻译结果绑定具体 `revision`，过期结果不会覆盖新文本。

## 系统要求

- Linux。
- Python `>=3.10`。
- `uv`，用于创建和维护项目内 `.venv`。
- Qwen3-ASR 和 vLLM 支持的 GPU/加速环境；请先按硬件厂商与上游项目说明安装匹配的 Torch、ROCm 或 CUDA 组件。
- 可选 OpenAI 兼容翻译 API。
- VoxBridge 本地服务固定使用端口 `8024`。

项目依赖固定包含 `qwen-asr[vllm]==0.0.6`。GPU 运行时兼容性应在安装 VoxBridge 前单独验证。

## 安装

独立克隆：

```bash
git clone https://github.com/hellcatjack/VoxBridge.git
cd VoxBridge
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e .
```

如果 VoxBridge 位于现有 Qwen3-ASR 工作区内，可直接执行
`uv pip install --python ../.venv/bin/python -e .`，不要重复创建环境。

## 快速启动（8024）

以下配置适合受信任的本机开发环境，默认不启用登录：

```bash
.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-0.6B \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8024
```

确认服务：

```bash
ss -lntp | rg ':8024'
```

浏览器访问 `http://127.0.0.1:8024`。公网部署必须先阅读 [部署指南](docs/DEPLOYMENT.md)，启用认证并使用 HTTPS/WSS。

## 翻译与 Context

启用 OpenAI 兼容翻译后端：

```bash
--enable-translation \
--translation-backend openai_api \
--translation-api-base-url <openai-compatible-base-url> \
--translation-api-model <model-name>
```

如果兼容 API 需要鉴权，再通过运行时参数提供 `--translation-api-key`；不要把真实 Token 写入仓库、systemd unit 或 shell 历史。

页面中的“专业术语 Context”只接受少量人名、地名或专业术语：

- 非空输入覆盖当前 WebSocket 会话的服务端 schedule。
- 空输入显式禁用当前会话的 context。
- 客户端省略 `asr_context_terms` 时，继续使用服务端 schedule。
- Context 是术语偏置，不是逐字稿；不要粘贴句子、完整演讲稿或机密转写。
- 日志只记录启用状态、数量、字符数和 SHA-256 指纹，不回显术语正文。

完整字段和 schedule 格式见 [后端 API](docs/API.md)。

## 公网认证

认证默认关闭。公网运行时应启用：

```bash
--auth-enabled \
--auth-username admin \
--auth-cookie-secure \
--disable-debug-file
```

密码使用 PBKDF2 哈希，通过 `VOXBRIDGE_AUTH_PASSWORD_HASH` 注入。`--auth-cookie-secure` 只能用于 HTTPS/WSS。完整的用户级 systemd 与反向代理示例见 [部署指南](docs/DEPLOYMENT.md)。

## 测试

```bash
.venv/bin/python -m pytest -q
```

使用真实 16 kHz、单声道、PCM16 WAV 进行 WebSocket 回放：

```bash
.venv/bin/python tools/subtitle_ws_selfcheck.py \
  --ws-url ws://127.0.0.1:8024/ws \
  --wav <path-to-16k-mono-pcm16.wav>
```

认证启用时，后端回放工具需要相应的已认证会话；浏览器级验证建议通过正常登录页面使用 Playwright。

## 文档

- [CHANGELOG.md](CHANGELOG.md)：版本变更。
- [docs/API.md](docs/API.md)：HTTP/WebSocket、稳定性、revision、context 与 trace 协议。
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)：`.venv`、用户级 systemd、HTTPS、认证与运维检查。
- [docs/SECURITY_SCAN.md](docs/SECURITY_SCAN.md)：发布前信息泄露扫描范围和结果。

## 安全边界

- 不提交 API Token、密码哈希、`.env`、会议音频、字幕 trace、日志、截图或模型文件。
- 公网部署关闭 `/__debug/file`，并通过 HTTPS 反向代理访问仅监听回环地址的 `8024`。
- 字幕 trace 可能包含会议原文，应按敏感业务数据处理。
- 发布前执行 [安全扫描](docs/SECURITY_SCAN.md) 中的检查。

## License

Apache License 2.0，见 [LICENSE](LICENSE)。
