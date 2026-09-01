# VoxBridge

VoxBridge 是一个面向会议、新闻和演讲场景的实时语音识别与双语字幕服务。浏览器负责采集音频和显示字幕，后端负责 Qwen3-ASR 流式状态、切段、句子稳定性和翻译调度。

## 核心能力

- 支持麦克风输入，以及浏览器“整屏共享 + 系统音频”输入。
- 支持 `中文 -> 英文` 和 `英文 -> 中文`；启动后锁定方向，避免 ASR 语言与翻译方向漂移。
- 每个 WebSocket 会话维护独立的 streaming state；VAD、硬时长轮转、最终 flush 和重叠处理均由后端负责。
- 使用有界音频队列和背压机制，避免长时间会话无限积压。
- 浏览器保留短停顿原始 PCM；确认超过 700 ms 的长静音时保留前 400 ms 低能量句尾、把其余静音压缩为有序控制事件，恢复时补送最近 400 ms pre-roll，后端仍独占 VAD、切段和固化决策。
- 后端静音省算力期间不会把前端静音控制事件送入 ASR；全程只有静音控制事件的 state 也不会在 Stop 时调用 ASR finish。可选 Silero VAD 继续只影响 context 恢复保护，不改变全局在线切段。
- 后端通过 `sentence_id` 和 `revision` 明确区分新句、稳定修订和对应翻译，前端不使用固定词表猜测稳定性。
- 支持页面输入会话级专业术语 Context，也支持后端加载有界、分时段的 context schedule。
- 最近 100 条字幕可滚动查询；默认跟随最新内容，用户手动滚动时暂停自动跟随。
- 支持上下字幕字体调整、移动端布局、控制栏自动隐藏和返回最新字幕按钮。
- 可选使用 CPU-only Kokoro-82M，在独立 `/listen` 页面向多个设备广播稳定译文并各自严格 FIFO 朗读。
- 可选单用户登录、Secure Cookie、结构化字幕 trace 和文本池诊断日志。

## 工作流程

```text
Browser PCM + audio activity events
  -> WebSocket /ws
  -> bounded audio queue
  -> backend VAD / hard segment rotation
  -> Qwen3-ASR streaming state
  -> sentence_id + revision
  -> OpenAI-compatible translation API
  -> bilingual subtitle rows
  -> stable translation HLS queue
  -> one shared Kokoro synthesis worker
  -> one continuous AAC/HLS live stream
  -> native playback on every listener device
```

浏览器只负责音频传输优化和渲染后端事件，不判断句子边界。小于 700 ms 的短停顿会连同语音原样发送；确认长静音时仅保留前 400 ms 作为低能量句尾保护，其余静音通过 `audio_silence` 推进后端静音时钟而不进入模型。后端在 final 前只补解码尚未送入 ASR 的有界尾音，不在持续静音期间反复推理。生成中的 `partial` 可以变化；已固化句子或长句中的稳定子句通过 `sentence_committed` 创建，通过 `sentence_updated` 更新同一个 `sentence_id`。稳定子句只影响字幕与翻译单元，不会在逗号处切换 ASR state。翻译结果绑定具体 `revision`，过期结果不会覆盖新文本。

## 系统要求

- Linux。
- Python `>=3.10`。
- `uv`，用于创建和维护项目内 `.venv`。
- Qwen3-ASR 和 vLLM 支持的 GPU/加速环境；请先按硬件厂商与上游项目说明安装匹配的 Torch、ROCm 或 CUDA 组件。
- 可选 OpenAI 兼容翻译 API。
- VoxBridge 本地服务固定使用端口 `8024`。

项目依赖固定包含 `qwen-asr[vllm]==0.0.6`。GPU 运行时兼容性应在安装 VoxBridge 前单独验证。

### macOS 浏览器客户端

- Safari 14.1+、macOS Chrome 和 Edge 均可使用麦克风输入。采集优先使用 AudioWorklet；不可用或初始化失败时回退到 ScriptProcessor。
- 麦克风和屏幕采集必须通过 HTTPS/WSS，只有 `localhost` 本地开发例外。
- “系统声音”依赖浏览器和 macOS 对 `getDisplayMedia` 音轨的实际支持。页面会检查返回的 audio track；浏览器没有提供音轨时会明确终止启动，不会建立一个伪静音会话。
- 需要采集浏览器标签页或其它应用声音时优先使用当前版 Chrome/Edge，并在共享选择器中确认音频选项。Safari 可稳定用于麦克风输入，但不能假定所有 Safari/macOS 组合都支持系统音频共享。

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
  --mm-processor-cache-gb 0 \
  --segment-final-redecode
```

确认服务：

```bash
ss -lntp | rg ':8024'
```

浏览器访问 `http://127.0.0.1:8024`。公网部署必须先阅读 [部署指南](docs/DEPLOYMENT.md)，启用认证并使用 HTTPS/WSS。

生产环境可安装仓库中的用户级 systemd 依赖，使
`systemctl --user start|stop|restart voxbridge-8024.service` 同步管理 Q8_0
翻译 API，并在启动 ASR 前等待翻译模型真正就绪。具体安装步骤见
[部署指南](docs/DEPLOYMENT.md)。

生产 trace 已开启时，可以追加以下参数观测 VAD，并仅用强 Silero 语音证据救回原本会被 SNR 门丢弃的轻声音频：

```bash
--silent-decode-pre-roll-sec 0.4 \
--silero-vad-shadow \
--silero-vad-rescue \
--silero-vad-shadow-threshold 0.5 \
--silero-vad-shadow-log-sec 1.0
```

pre-roll 只缓存现有门控原本会跳过的音频：恢复推理时重放一次；若直接到达段落端点，则在 final 前作为有界尾音解码一次。`--silero-vad-rescue` 按整批累计语音证据阻止“前部语音、后部静音”的合并批次被整体跳过，但不覆盖能量 VAD 已确认的静音端点，也不直接提交文本；Silero 加载或推理失败只会产生 `silero_shadow_unavailable`，不会阻断 ASR、翻译或 TTS。

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

译文朗读必须同时启用翻译和 `--enable-tts`，并要求系统可执行文件中存在 `ffmpeg`。主字幕页不播放音频，只提供独立 `/listen` 页面入口；该入口免登录，固定公开地址为 `https://ushome.amycat.com:18024/listen`，主页面默认显示指向它的本地静态二维码。Pittsburgh Christian Church South（PCCS）专用监听页全部使用英文，并固定在一个浏览器视口内，不产生页面滚动条。手机、平板或其他电脑扫码后点击该设备自己的 Start 即可监听。

监听页把一个持续的音频元素绑定到这次用户点击：iPhone/Safari 使用原生 HLS，以便锁屏后不依赖 JavaScript 唤醒；具备 MSE AAC 解码能力的桌面 Chrome、Edge 和 Firefox 使用项目内固定版本的 hls.js，避免依赖不可靠的桌面原生 HLS 探测。后端在没有译文时持续写入听感静音但可解码的 AAC 载波，两种客户端都不会切换逐句音频文件或创建额外的 Kokoro/FFmpeg 任务。主字幕页、ASR WebSocket、登录和管理接口仍受原有认证保护；只有监听页、二维码、本地 hls.js 资产、共享直播状态和 listener-scoped HLS 能力端点公开。

共享 HLS 使用 bounded pre-listener backlog：当前 ASR 会话中已经通过 revision 稳定门、但发布时尚无监听设备的译文，会保留在最多 128 项的有界待播池中，但此时不会启动 Kokoro 或 FFmpeg，也不会把这些旧句计算成新听众的积压。首台设备加入新的直播时间线时，后端会丢弃过期待播项，只保留最新一条稳定译文交给同一个 Kokoro worker，并以显示 `1.0x` 朗读这条入场句，随后继续按源句顺序朗读新译文；因此扫码加入或重新 Start 默认从直播最新内容开始，而不会从会议开头补播。超过上限时仍只淘汰最旧的未播项。新 ASR 会话开始时，如果没有活跃监听或编码器，会清除上一会话的待播池，防止跨会议串音。

多个翻译 worker 即使乱序完成，也会按源句顺序发布。无论同时连接多少台设备，每条译文只由一个 Kokoro worker 合成一次，也只进入一个 FFmpeg 编码器；设备读取同一套 HLS 分片。首台设备只是触发新的共享 speech epoch，并不拥有语速控制器；只要仍有一台设备在线，原首台设备退出也不会重置时间线、积压或语速。最后一个租约离开或超过 90 秒未刷新后，后端关闭共享编码器并清理该 epoch 的队列、字幕和临时分片。之后产生的稳定译文只进入空闲有界池，等待下一台设备从现场重新开始。

`/listen` 的 Playback Speed 现在是只读的全局状态。所有浏览器的 HTML 音频元素始终以 `1.0x` 消费同一条 HLS；Auto 倍速由服务器在每句开始合成时通过 Kokoro 原生 `speed` 参数决定，并固定到该句结束。这样 Windows Chrome 与 iPhone Safari 听到完全相同的句序、声音速度和共享时间线，单台设备的网络缓冲不会改变其他听众的语速，也不会产生额外的合成或编码任务。

如需回滚自适应控制，可在启动参数中加入 `--disable-tts-global-auto-speed`：Kokoro 将固定使用 `--tts-speed`，所有客户端仍保持媒体播放率 `1.0x`，不会恢复已移除且在 iPhone 上不可靠的客户端独立变速。

共享 HLS 在服务器端保持严格 FIFO。Kokoro 输出的 PCM 后固定追加 `300ms` 句号停顿，再写入同一条连续音频时间线；停顿不分析中英文文本。语音队列从真正空闲恢复时，FFmpeg 写入端只允许最前面的 2 秒 PCM 以 `2.0x` 短促追赶，之后必须按实时 `1.0x` 持续发布；相邻句子共享同一个短促额度，不能每句重新获得 2 秒。这样长积压不会因为服务器提前写进 HLS 而从 Auto 计算中消失。译文队列和 FFmpeg 前的 PCM 队列都有上限，编码变慢时通过背压限制内存增长，而不是按监听设备复制音频或启动额外合成任务。`GET /api/tts/live/status` 的 `pending_audio_ms` 显示已经合成、仍在等待写入实时 HLS 时间线的音频时长；`translated_audio_backlog_ms` 是当前 speech epoch 尚未发布到 HLS 的保守最大语音时长，包括精确 PCM 和去重后的待合成句。未合成句使用“语言默认值”和“近期实测字符/时长加 10% 安全余量”两者中较大的估值，并统一按显示 `1.0x` 估算，避免低估。没有听众时这些公开积压字段固定为零。

`/listen` 的 Live Audio 区域会显示当前设备正在朗读的译文，而不是后端最新生成的译文；`Speech backlog` 仅表示服务器尚未发布的共享语音债务，不包含某台手机的网络延迟或 HLS playhead。Auto 对下一句按该保守积压选择 `<10s 1.0x`、`10–<30s 1.2x`、`30–<40s 1.4x`、`>=40s 1.5x`；已开始或已经发布的 PCM 不会被变速。后端按实际写入 FFmpeg 的 PCM 媒体样本建立最多 256 条绝对时间字幕提示，计入 AAC 编码帧延迟，并通过与语言无关的波形活动检测排除 Kokoro 首尾静音。Safari 使用原生 `getStartDate() + currentTime` 定位该设备正在播放的节目时间，因此网络缓存或播放列表更新进度不会把服务器的新字幕提前显示。句间停顿时保留上一句，不清空、变灰或闪烁。若等待新译文期间为完成 HLS 分片而产生了临时静音载波，字幕提示会把它和正常 `300ms` 句间停顿分开；浏览器只有在跳转目标及下一句后 1 秒音频已位于同一缓冲区时才执行一次 seek。该处理不调用 pause/play、不循环重试，也不改变媒体播放率。字幕元数据轮询只在页面前台运行，失败或锁屏暂停不会阻塞原生 HLS 音频。

有活跃监听设备时，后端会在译文生成后按精确的 `sentence_id + revision + target_language + text hash` 提前执行 Kokoro，但不会提前把音频写入 HLS。只有原有 revision 稳定门按源顺序放行后，缓存命中的 PCM 才能进入直播时间线；译文修订会立即使旧缓存失效。预合成结果同时记录开始合成时所选的显示倍速和有效 Kokoro speed；稳定门放行时直接复用这份 PCM，不会因为此刻未发布积压已经降低而重新合成为慢速版本。缓存最多保留 8 项，仍由同一个 Kokoro worker 处理，因此不会因设备数量增加 CPU 消耗，也不会降低 3 秒稳定门。

模型资产不进入 Git。部署时在 `models/kokoro/` 放置：

- `kokoro-v1.0.onnx` 与 `voices-v1.0.bin`：英语 Kokoro v1.0。
- `kokoro-v1.1-zh.onnx`、`voices-v1.1-zh.bin` 与 `config-v1.1-zh.json`：中文 Kokoro v1.1-zh。

模型来源应使用 [kokoro-onnx 官方 release](https://github.com/thewh1teagle/kokoro-onnx/releases) 和 [Kokoro-82M-v1.1-zh 官方模型页](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)。完整启动参数见 [部署指南](docs/DEPLOYMENT.md)。

主字幕页 Stop 会先等待尚未完成的稳定译文并发布朗读任务，但不会等待音频合成或设备播放。监听页 Stop 只停止当前设备，不影响其他设备或共享时间线；重新 Start 后从当时的 live edge 继续收听。公开部署必须为主字幕和 ASR 启用登录认证并使用 HTTPS；公开监听直播本身不含访问控制，因此应将二维码和地址视作公开会议信息。随机 listener ID 只作为短期 bearer capability，默认最多保留 128 个并发租约，避免公开入口被无界占用。使用系统音频输入时，建议让朗读设备与采集设备分离或使用耳机，避免回采。

可见字幕和译文继续实时更新；朗读采用更严格的后端稳定门。推荐启用 `--segment-final-redecode`：自然 VAD 端点在固化、翻译和 TTS 封存前，对当前段音频执行一次有界的一次性解码，用相容结果修复流式句尾。讲话中的 hard cut 不执行该阻塞式重解码，避免实时音频在轮转期间堆满队列后被丢弃。段尾校验同时比较上一段待拼接前缀和当前段组成的有效文本池；若完整句数量回退，就保留最后一次流式结果，避免句界重排吞掉已显示末句。启用后，最新且未封存的来源不会仅因固定计时到期而朗读；它必须等到段校验成功，或后继句提供回滚安全证据。校验为空、失败或与流式文本明显偏离时不会覆盖字幕，也不会提前封存 TTS。普通已非最新来源仍使用 `--tts-revision-stable-sec 3.0`；未启用段校验时，最新来源继续使用 `--tts-latest-revision-grace-sec 4.0`。这不是全局 7 秒延时；正常 Stop 只 flush 当前 streaming state、排空最终译文，不清空或重建已显示字幕。仅在离线校验场景显式传入 `--final-redecode-on-stop`，才恢复全会话重识别。

vLLM 的多模态处理器缓存可能同时存在于 API 进程和 EngineCore。单麦克风、`--max-connections 1` 的生产模式建议显式使用 `--mm-processor-cache-gb 0`：连续音频请求没有可复用的跨请求处理结果，禁用缓存可避免长会话逐步填充主机缓存。多连接或重复离线音频部署应先比较 RSS、GTT 和解码延迟，再决定是否启用非零缓存。

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
.venv/bin/python -m tools.subtitle_ws_selfcheck \
  --ws-url ws://127.0.0.1:8024/ws \
  --wav <path-to-16k-mono-pcm16.wav>
```

已有 YouTube `json3` 参考字幕和 WebSocket 事件时，可检查疑似整句空洞、
强对齐后的句尾缺失、重复固化和翻译 ID 缺口：

```bash
.venv/bin/python -m tools.subtitle_reference_coverage \
  --reference-json3 <reference.zh-Hans.json3> \
  --events-jsonl <websocket-events.jsonl> \
  --duration-sec 180
```

参考字幕仅用于离线诊断，不参与后端切段、稳定性或文本固化。工具采用模糊
覆盖而非逐字一致判定；未完整播放到结束的边界 cue 会被排除。

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
