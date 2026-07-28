"""Standalone browser UI for live translated-speech listeners."""

TTS_LISTENER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>译文实时朗读</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #25352e;
      --muted: #66776c;
      --paper: #f7f5eb;
      --panel: rgba(255, 255, 248, 0.9);
      --line: rgba(89, 112, 95, 0.2);
      --accent: #386f5d;
      --accent-soft: #dceade;
      --warn: #9a632f;
      --danger: #934f49;
      --shadow: 0 26px 80px rgba(58, 75, 62, 0.13);
    }

    * { box-sizing: border-box; }

    html, body { min-height: 100%; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Avenir Next", "Noto Sans SC", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at 15% 12%, rgba(217, 235, 213, 0.9), transparent 36%),
        radial-gradient(circle at 88% 88%, rgba(231, 220, 188, 0.62), transparent 38%),
        linear-gradient(145deg, #f5f3e8 0%, #edf3e7 52%, #f3efe2 100%);
      display: grid;
      place-items: center;
      padding: max(18px, env(safe-area-inset-top)) 18px max(18px, env(safe-area-inset-bottom));
    }

    main {
      width: min(720px, 100%);
      min-height: min(660px, calc(100vh - 36px));
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      padding: clamp(24px, 5vw, 52px);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 28px;
    }

    .eyebrow {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
      font-size: clamp(38px, 8vw, 72px);
      font-weight: 700;
      line-height: 1.05;
      letter-spacing: -0.035em;
    }

    .intro {
      max-width: 32em;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: clamp(15px, 2.4vw, 18px);
      line-height: 1.75;
    }

    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .status-card {
      min-height: 112px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(248, 250, 242, 0.74);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 12px;
    }

    .status-card span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
    }

    .status-card strong {
      font-size: clamp(18px, 3vw, 24px);
      line-height: 1.2;
      overflow-wrap: anywhere;
    }

    .status-card[data-state="ok"] strong { color: var(--accent); }
    .status-card[data-state="warn"] strong { color: var(--warn); }
    .status-card[data-state="error"] strong { color: var(--danger); }

    .now-playing {
      min-height: 108px;
      padding: 20px;
      border-radius: 22px;
      background: linear-gradient(135deg, #315f50, #477b67);
      color: #f8fff6;
      box-shadow: 0 18px 45px rgba(52, 94, 76, 0.2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .now-playing small {
      display: block;
      margin-bottom: 7px;
      color: rgba(244, 255, 246, 0.72);
      font-weight: 700;
      letter-spacing: 0.08em;
    }

    .now-playing strong {
      display: block;
      font-size: clamp(20px, 4vw, 30px);
    }

    .pulse {
      width: 54px;
      height: 54px;
      flex: 0 0 auto;
      border-radius: 50%;
      border: 1px solid rgba(255, 255, 255, 0.28);
      background: rgba(255, 255, 255, 0.11);
      position: relative;
    }

    .pulse::before, .pulse::after {
      content: "";
      position: absolute;
      top: 50%;
      width: 4px;
      border-radius: 99px;
      background: #efffed;
      transform: translateY(-50%);
    }

    .pulse::before { left: 18px; height: 16px; }
    .pulse::after { right: 18px; height: 28px; }
    .now-playing[data-playing="true"] .pulse { animation: breathe 1.4s ease-in-out infinite; }

    @keyframes breathe {
      50% { transform: scale(1.08); box-shadow: 0 0 0 9px rgba(255, 255, 255, 0.08); }
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .playback-settings {
      min-height: 64px;
      margin-bottom: 12px;
      padding: 10px 12px 10px 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(248, 250, 242, 0.74);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .playback-settings label {
      color: var(--muted);
      font-size: 14px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }

    .playback-settings select {
      min-width: 112px;
      min-height: 42px;
      padding: 0 34px 0 14px;
      border: 1px solid rgba(56, 111, 93, 0.3);
      border-radius: 12px;
      color: var(--ink);
      background: #fbfcf6;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    button, .back-link {
      min-height: 54px;
      border: 1px solid transparent;
      border-radius: 15px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
    }

    button:hover:not(:disabled), .back-link:hover { transform: translateY(-1px); }
    button:disabled { cursor: default; opacity: 0.45; }

    #startListening {
      color: #f8fff6;
      background: var(--accent);
      box-shadow: 0 12px 28px rgba(56, 111, 93, 0.22);
    }

    #stopListening {
      color: var(--danger);
      border-color: rgba(147, 79, 73, 0.28);
      background: rgba(255, 250, 244, 0.8);
    }

    .back-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: auto;
      margin-top: 14px;
      color: var(--muted);
      text-decoration: none;
      font-size: 13px;
    }

    @media (max-width: 600px) {
      body { padding: 0; place-items: stretch; }
      main { min-height: 100svh; border: 0; border-radius: 0; padding: 28px 18px; }
      .status-grid { grid-template-columns: 1fr; }
      .status-card { min-height: 76px; flex-direction: row; align-items: center; }
      .playback-settings { width: 100%; }
      .playback-settings select { min-width: 0; width: min(132px, 42vw); }
      .actions { position: sticky; bottom: 0; padding-bottom: env(safe-area-inset-bottom); }
    }
  </style>
</head>
<body>
  <main>
    <section>
      <p class="eyebrow">VoxBridge Live Audio</p>
      <h1>译文实时朗读</h1>
      <p class="intro">点击开始后，本设备只朗读加入之后产生的稳定译文。停止或断线不会影响其他收听设备。</p>
    </section>

    <section class="status-grid" aria-live="polite">
      <div id="connectionCard" class="status-card" data-state="warn">
        <span>连接</span><strong id="connectionStatus">尚未开始</strong>
      </div>
      <div id="producerCard" class="status-card" data-state="warn">
        <span>会议</span><strong id="producerStatus">等待状态</strong>
      </div>
      <div id="queueCard" class="status-card">
        <span>待播队列</span><strong id="queueStatus">0 条</strong>
      </div>
    </section>

    <section id="nowPlaying" class="now-playing" data-playing="false" aria-live="polite">
      <div>
        <small>当前播放</small>
        <strong id="playbackStatus">等待开始</strong>
      </div>
      <div class="pulse" aria-hidden="true"></div>
    </section>

    <audio id="ttsPlayback" preload="auto" hidden></audio>

    <section>
      <div class="playback-settings">
        <label for="playbackRate">朗读速度</label>
        <select id="playbackRate" aria-label="朗读速度">
          <option value="0.75">0.75x</option>
          <option value="1" selected>1.0x</option>
          <option value="1.25">1.25x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2.0x</option>
        </select>
      </div>
      <div class="actions">
        <button id="startListening" type="button">开始收听</button>
        <button id="stopListening" type="button" disabled>停止收听</button>
      </div>
      <a class="back-link" href="/">返回字幕页面</a>
    </section>
  </main>

  <script>
  (() => {
    const startButton = document.getElementById("startListening");
    const stopButton = document.getElementById("stopListening");
    const connectionCard = document.getElementById("connectionCard");
    const producerCard = document.getElementById("producerCard");
    const connectionStatus = document.getElementById("connectionStatus");
    const producerStatus = document.getElementById("producerStatus");
    const queueStatus = document.getElementById("queueStatus");
    const playbackStatus = document.getElementById("playbackStatus");
    const nowPlaying = document.getElementById("nowPlaying");
    const playbackRateInput = document.getElementById("playbackRate");
    const playbackElement = document.getElementById("ttsPlayback");
    const PLAYBACK_RATE_STORAGE_KEY = "voxbridge.ttsPlaybackRate";
    const SUPPORTED_PLAYBACK_RATES = new Set([0.75, 1, 1.25, 1.5, 2]);
    const SILENT_WAV_DATA_URL =
      "data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA==";

    function normalizePlaybackRate(value) {
      const parsed = Number(value);
      return SUPPORTED_PLAYBACK_RATES.has(parsed) ? parsed : 1;
    }

    function readPlaybackRate() {
      try {
        return normalizePlaybackRate(
          window.localStorage.getItem(PLAYBACK_RATE_STORAGE_KEY)
        );
      } catch (error) {
        return 1;
      }
    }

    let playbackRate = readPlaybackRate();
    playbackRateInput.value = String(playbackRate);

    let socket = null;
    let listenerId = "";
    let queue = [];
    let currentJob = null;
    let abortController = null;
    let activeObjectUrl = "";
    let cancelActivePlayback = null;
    let generation = 0;
    let heartbeat = null;
    const seenJobIds = new Set();

    function wsUrl(path) {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${window.location.host}${path}`;
    }

    function setCard(card, value) {
      card.dataset.state = value || "";
    }

    function updateQueueStatus() {
      queueStatus.textContent = `${queue.length} 条`;
    }

    function send(message) {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message));
      }
    }

    function applyPlaybackRate() {
      playbackElement.defaultPlaybackRate = playbackRate;
      playbackElement.playbackRate = playbackRate;
      if ("preservesPitch" in playbackElement) playbackElement.preservesPitch = true;
      if ("mozPreservesPitch" in playbackElement) playbackElement.mozPreservesPitch = true;
      if ("webkitPreservesPitch" in playbackElement) {
        playbackElement.webkitPreservesPitch = true;
      }
    }

    function releaseActiveObjectUrl() {
      if (!activeObjectUrl) return;
      window.URL.revokeObjectURL(activeObjectUrl);
      activeObjectUrl = "";
    }

    function stopActivePlayback() {
      if (cancelActivePlayback) {
        const cancel = cancelActivePlayback;
        cancelActivePlayback = null;
        cancel();
      }
      playbackElement.pause();
      playbackElement.removeAttribute("src");
      playbackElement.load();
      releaseActiveObjectUrl();
    }

    async function unlockPlaybackElement() {
      playbackElement.muted = true;
      playbackElement.src = SILENT_WAV_DATA_URL;
      try {
        await playbackElement.play();
      } finally {
        playbackElement.pause();
        playbackElement.muted = false;
        playbackElement.removeAttribute("src");
        playbackElement.load();
        applyPlaybackRate();
      }
    }

    applyPlaybackRate();

    async function fetchAudio(job, signal) {
      let lastError = null;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const response = await fetch(
          `/api/tts/broadcast/jobs/${encodeURIComponent(job.job_id)}/audio`,
          {
            method: "POST",
            credentials: "same-origin",
            cache: "no-store",
            signal,
            headers: { "X-TTS-Listener-ID": listenerId },
          }
        );
        if (response.ok) {
          const audioBytes = await response.arrayBuffer();
          send({ type: "tts_received", job_id: job.job_id });
          return audioBytes;
        }
        lastError = new Error(`audio request failed: ${response.status}`);
        if (response.status !== 503 || attempt > 0) break;
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
      throw lastError || new Error("audio request failed");
    }

    async function playAudioBuffer(buffer, localGeneration) {
      if (localGeneration !== generation) return;
      const audioBlob = new Blob([buffer], { type: "audio/wav" });
      activeObjectUrl = window.URL.createObjectURL(audioBlob);
      playbackElement.src = activeObjectUrl;
      playbackElement.load();
      applyPlaybackRate();
      try {
        await new Promise((resolve, reject) => {
          let settled = false;
          const settle = (error) => {
            if (settled) return;
            settled = true;
            playbackElement.removeEventListener("ended", onEnded);
            playbackElement.removeEventListener("error", onError);
            cancelActivePlayback = null;
            if (error) reject(error);
            else resolve();
          };
          const onEnded = () => settle();
          const onError = () => settle(new Error("audio playback failed"));
          cancelActivePlayback = () => {
            settle(new DOMException("playback stopped", "AbortError"));
          };
          playbackElement.addEventListener("ended", onEnded, { once: true });
          playbackElement.addEventListener("error", onError, { once: true });
          const playPromise = playbackElement.play();
          if (playPromise) playPromise.catch(onError);
        });
      } finally {
        playbackElement.removeAttribute("src");
        playbackElement.load();
        releaseActiveObjectUrl();
      }
    }

    async function pumpQueue() {
      if (currentJob || queue.length === 0) return;
      const localGeneration = generation;
      currentJob = queue.shift();
      updateQueueStatus();
      abortController = new AbortController();
      playbackStatus.textContent = `正在朗读 · ${currentJob.target_language || "译文"}`;
      nowPlaying.dataset.playing = "true";
      try {
        const audioBytes = await fetchAudio(currentJob, abortController.signal);
        await playAudioBuffer(audioBytes, localGeneration);
      } catch (error) {
        if (error && error.name !== "AbortError" && localGeneration === generation) {
          playbackStatus.textContent = "本条音频不可用，继续下一条";
        }
      } finally {
        if (localGeneration !== generation) return;
        abortController = null;
        currentJob = null;
        nowPlaying.dataset.playing = "false";
        playbackStatus.textContent = queue.length > 0 ? "准备下一条" : "等待新译文";
        updateQueueStatus();
        void pumpQueue();
      }
    }

    function clearHeartbeat() {
      if (heartbeat !== null) {
        window.clearInterval(heartbeat);
        heartbeat = null;
      }
    }

    function resetLocalPlayback() {
      generation += 1;
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
      stopActivePlayback();
      queue = [];
      currentJob = null;
      seenJobIds.clear();
      nowPlaying.dataset.playing = "false";
      playbackStatus.textContent = "等待开始";
      updateQueueStatus();
    }

    function stopListening() {
      clearHeartbeat();
      const closingSocket = socket;
      socket = null;
      listenerId = "";
      if (closingSocket && closingSocket.readyState < WebSocket.CLOSING) {
        closingSocket.close(1000, "listener stopped");
      }
      resetLocalPlayback();
      connectionStatus.textContent = "已停止";
      producerStatus.textContent = "等待状态";
      setCard(connectionCard, "warn");
      setCard(producerCard, "warn");
      startButton.disabled = false;
      stopButton.disabled = true;
    }

    async function startListening() {
      if (socket) return;
      resetLocalPlayback();
      await unlockPlaybackElement();
      startButton.disabled = true;
      stopButton.disabled = false;
      connectionStatus.textContent = "正在连接";
      setCard(connectionCard, "warn");

      const activeSocket = new WebSocket(wsUrl("/ws/tts"));
      socket = activeSocket;
      activeSocket.addEventListener("open", () => {
        if (socket !== activeSocket) return;
        heartbeat = window.setInterval(() => send({ type: "ping" }), 20000);
      });
      activeSocket.addEventListener("message", (event) => {
        if (socket !== activeSocket) return;
        let message;
        try { message = JSON.parse(event.data); } catch (error) { return; }
        if (message.type === "tts_listener_ready") {
          listenerId = String(message.listener_id || "");
          connectionStatus.textContent = message.tts_available ? "已连接" : "服务不可用";
          setCard(connectionCard, message.tts_available ? "ok" : "error");
          producerStatus.textContent = message.producer_active ? "会议进行中" : "等待会议";
          setCard(producerCard, message.producer_active ? "ok" : "warn");
          playbackStatus.textContent = message.tts_available ? "等待新译文" : "朗读服务不可用";
          return;
        }
        if (message.type === "producer_status") {
          producerStatus.textContent = message.active ? "会议进行中" : "会议已停止";
          setCard(producerCard, message.active ? "ok" : "warn");
          return;
        }
        if (message.type === "tts_job" && message.is_stable === true) {
          const jobId = String(message.job_id || "");
          if (!jobId || seenJobIds.has(jobId)) return;
          seenJobIds.add(jobId);
          const job = message;
          queue.push(job);
          updateQueueStatus();
          void pumpQueue();
          return;
        }
        if (message.type === "error") {
          connectionStatus.textContent = "连接错误";
          setCard(connectionCard, "error");
        }
      });
      activeSocket.addEventListener("close", () => {
        if (socket !== activeSocket) return;
        clearHeartbeat();
        socket = null;
        listenerId = "";
        resetLocalPlayback();
        connectionStatus.textContent = "连接已断开，请重新开始";
        setCard(connectionCard, "error");
        startButton.disabled = false;
        stopButton.disabled = true;
      });
      activeSocket.addEventListener("error", () => {
        if (socket !== activeSocket) return;
        connectionStatus.textContent = "网络异常";
        setCard(connectionCard, "error");
      });
    }

    startButton.addEventListener("click", () => {
      startListening().catch(() => {
        stopListening();
        connectionStatus.textContent = "启动失败";
        setCard(connectionCard, "error");
      });
    });
    stopButton.addEventListener("click", stopListening);
    playbackRateInput.addEventListener("change", () => {
      playbackRate = normalizePlaybackRate(playbackRateInput.value);
      playbackRateInput.value = String(playbackRate);
      try {
        window.localStorage.setItem(PLAYBACK_RATE_STORAGE_KEY, String(playbackRate));
      } catch (error) {}
      applyPlaybackRate();
    });
    window.addEventListener("beforeunload", () => {
      if (socket) socket.close(1000, "page closed");
      stopActivePlayback();
    });
  })();
  </script>
</body>
</html>
"""


__all__ = ["TTS_LISTENER_HTML"]
