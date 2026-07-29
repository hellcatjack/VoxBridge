# VoxBridge

VoxBridge 是一个面向会议、新闻和演讲场景的实时语音识别与双语字幕服务。浏览器负责采集音频和显示字幕，后端负责 Qwen3-ASR 流式状态、切段、句子稳定性和翻译调度。

## 核心能力

- 支持麦克风输入，以及浏览器“整屏共享 + 系统音频”输入。
- 支持 `中文 -> 英文` 和 `英文 -> 中文`；启动后锁定方向，避免 ASR 语言与翻译方向漂移。
- 每个 WebSocket 会话维护独立的 streaming state；VAD、硬时长轮转、最终 flush 和重叠处理均由后端负责。
- 使用有界音频队列和背压机制，避免长时间会话无限积压。
- 静音省算力期间保留最近 400 ms 未送入 ASR 的音频，恢复时一次补送，保护轻声起音；可选 Silero VAD 只写影子日志，不参与在线切段。
- 后端通过 `sentence_id` 和 `revision` 明确区分新句、稳定修订和对应翻译，前端不使用固定词表猜测稳定性。
- 支持页面输入会话级专业术语 Context，也支持后端加载有界、分时段的 context schedule。
- 最近 100 条字幕可滚动查询；默认跟随最新内容，用户手动滚动时暂停自动跟随。
- 支持上下字幕字体调整、移动端布局、控制栏自动隐藏和返回最新字幕按钮。
- 可选使用 CPU-only Kokoro-82M，在独立 `/listen` 页面向多个设备广播稳定译文并各自严格 FIFO 朗读。
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
  -> stable translation broadcast job
  -> /ws/tts listener snapshot
  -> shared Kokoro WAV synthesis
  -> independent device FIFO playback
```

浏览器只负责持续发送 PCM 音频和渲染后端事件。生成中的 `partial` 可以变化；已固化句子或长句中的稳定子句通过 `sentence_committed` 创建，通过 `sentence_updated` 更新同一个 `sentence_id`。稳定子句只影响字幕与翻译单元，不会在逗号处切换 ASR state。翻译结果绑定具体 `revision`，过期结果不会覆盖新文本。

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

需要译文朗读时，额外安装不含 GPU runtime 的 TTS extra：

```bash
uv pip install --python .venv/bin/python -e '.[tts]'
```

需要启用 Silero VAD 影子观测时，在已经验证好的加速环境中只安装其包和 ONNX 资源：

```bash
uv pip install --python .venv/bin/python --no-deps 'silero-vad==6.2.1'
```

不要在 ROCm 环境中直接解析安装 `.[vad-shadow]`；Silero 的通用依赖可能让解析器用 CUDA 版 Torch/Triton 替换现有 ROCm 版本。安装后必须重新核对 `torch`、`triton`、`torchaudio` 和 `onnxruntime` 版本。

安装前建议先加 `--dry-run`，确认求解结果不会卸载或替换已有的 Torch、Triton、ROCm/CUDA 包。

如果 VoxBridge 位于现有 Qwen3-ASR 工作区内，可直接执行
`uv pip install --python ../.venv/bin/python -e .`，不要重复创建环境。

## 快速启动（8024）

以下配置适合受信任的本机开发环境，默认不启用登录：

```bash
.venv/bin/python -m voxbridge.cli.demo_streaming_ws \
  --asr-model-path Qwen/Qwen3-ASR-0.6B \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8024 \
  --mm-processor-cache-gb 0.5
```

确认服务：

```bash
ss -lntp | rg ':8024'
```

浏览器访问 `http://127.0.0.1:8024`。公网部署必须先阅读 [部署指南](docs/DEPLOYMENT.md)，启用认证并使用 HTTPS/WSS。

生产 trace 已开启时，可以追加以下参数观测 VAD，而不改变当前 SNR 切段判断：

```bash
--silent-decode-pre-roll-sec 0.4 \
--silero-vad-shadow \
--silero-vad-shadow-threshold 0.5 \
--silero-vad-shadow-log-sec 1.0
```

pre-roll 只缓存现有门控原本会跳过的音频，恢复推理后立即清空。Silero 加载或推理失败只会产生 `silero_shadow_unavailable`，不会阻断 ASR、翻译或 TTS。

## 翻译与 Context

启用 OpenAI 兼容翻译后端：

```bash
--enable-translation \
--translation-backend openai_api \
--translation-api-base-url <openai-compatible-base-url> \
--translation-api-model <model-name>
```

中文→英文翻译会强制采用 ESV 的基督教、圣经与神学专业英文用语。该策略只规范术语和可确认的经文措辞，不补全、扩写或用模型记忆改写演讲者原文。

如果兼容 API 需要鉴权，再通过运行时参数提供 `--translation-api-key`；不要把真实 Token 写入仓库、systemd unit 或 shell 历史。

页面中的“专业术语 Context”只接受少量人名、地名或专业术语：

- 非空输入覆盖当前 WebSocket 会话的服务端 schedule。
- 空输入显式禁用当前会话的 context。
- 客户端省略 `asr_context_terms` 时，继续使用服务端 schedule。
- Context 是术语偏置，不是逐字稿；不要粘贴句子、完整演讲稿或机密转写。
- 日志只记录启用状态、数量、字符数和 SHA-256 指纹，不回显术语正文。

完整字段和 schedule 格式见 [后端 API](docs/API.md)。

## 译文朗读（Kokoro-82M）

译文朗读必须同时启用翻译和 `--enable-tts`。主字幕页不播放音频，只提供独立 `/listen` 页面入口；手机、平板或其他电脑通过同一 HTTPS 地址登录后打开 `/listen`，点击该设备自己的 Start 即可监听。多个设备可以同时连接，彼此的 Start、Stop、下载和播放进度互不干扰。

监听连接是 future-only：仅接收连接后产生的稳定译文，不回放连接前的历史任务。后端只为 `is_stable: true` 的完整译文创建广播任务；多个翻译 worker 即使乱序完成，也会按源句顺序发布。每条译文只合成一次共享 WAV，每个设备都按严格 FIFO 下载完整音频、确认接收、等待播放结束，再处理下一条。慢设备只增加自己的待播延迟；队列溢出时只断开该监听者。

每台监听设备可在 `/listen` 独立选择 `0.8x`、`0.9x`、`1.0x`、`1.1x` 或 `1.2x` 朗读速度。选择保存在该浏览器中，播放期间修改会立即生效，Stop、重连和刷新不会重置；旧版页面保存的其它倍速会安全回退到 `1.0x`。该设置不影响其他设备，也不会让后端重复合成共享 WAV。

监听页采用有界单条预取：当前译文朗读期间只提前合成并下载严格 FIFO 中的下一条，不会把整个待播队列载入内存。下一条到达播放位置时会直接复用已准备的 WAV；Stop、断线或刷新会取消尚未完成的预取。若 Kokoro 在当前音频结束前仍未完成下一条合成，播放器会继续等待，因此该机制显著减少正常切换停顿，但不承诺采样级无缝衔接。

每条成功朗读结束后固定保留 `300ms` 句号停顿，再播放已预取的下一条。该停顿不分析中英文文本、不随朗读倍速变化，也不会暂停后台合成和下载；Stop、断线和刷新会立即取消尚未结束的等待。

模型资产不进入 Git。部署时在 `models/kokoro/` 放置：

- `kokoro-v1.0.onnx` 与 `voices-v1.0.bin`：英语 Kokoro v1.0。
- `kokoro-v1.1-zh.onnx`、`voices-v1.1-zh.bin` 与 `config-v1.1-zh.json`：中文 Kokoro v1.1-zh。

模型来源应使用 [kokoro-onnx 官方 release](https://github.com/thewh1teagle/kokoro-onnx/releases) 和 [Kokoro-82M-v1.1-zh 官方模型页](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)。完整启动参数见 [部署指南](docs/DEPLOYMENT.md)。

主字幕页 Stop 会先等待尚未完成的稳定译文并发布朗读任务，但不会等待音频合成或设备播放。监听页 Stop 只停止当前设备并清空它的本地待播队列，不影响其他设备；重新 Start 后仍只接收后续译文。公开部署必须启用登录认证，避免未授权设备接入译文广播。使用系统音频输入时，建议让朗读设备与采集设备分离或使用耳机，避免回采。

可见字幕和译文继续实时更新；朗读采用更严格的后端稳定门。普通来源仍使用 `--tts-revision-stable-sec 3.0`，只有当前最新且尚未封存的来源再使用 `--tts-latest-revision-grace-sec 4.0` 防止发布后修订。这不是全局 7 秒延时：注册后继来源后，前一句立即恢复普通 3 秒规则；后端完成分段 final reconcile 后会封存本段来源并放行已就绪译文。若同一句在保护期内收到更高 revision，旧译文不会进入朗读队列。正常 Stop 仍会在最终 ASR 与翻译完成后排空最后的 ready 版本。

vLLM 的多模态处理器缓存可能同时存在于 API 进程和 EngineCore。建议显式使用 `--mm-processor-cache-gb 0.5`，把单进程缓存预算从 vLLM 默认值压低到 0.5 GiB；单引擎部署的理论总预算约为该值的两倍。设为 `0` 可以继续降低主机内存，但可能增加重复音频预处理开销。

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
