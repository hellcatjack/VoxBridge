"""Standalone browser UI for live translated-speech listeners."""

TTS_LISTENER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>PCCS Live Translation</title>
  <style>
    :root {
      color-scheme: light;
      --forest: #12241e;
      --forest-soft: #1c382f;
      --cream: #f4f5f1;
      --paper: #fffdf7;
      --sage: #dfe9e3;
      --sage-deep: #b9cec3;
      --mustard: #f4cf43;
      --coral: #d86456;
      --ink: #17201c;
      --muted: #66736d;
      --line: rgba(18, 36, 30, 0.18);
      --shadow: 0 24px 70px rgba(18, 36, 30, 0.18);
    }

    * { box-sizing: border-box; }

    html, body {
      width: 100%;
      height: 100%;
      height: 100dvh;
      margin: 0;
      overflow: hidden;
      overscroll-behavior: none;
    }

    body {
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      background:
        linear-gradient(90deg, rgba(18, 36, 30, 0.035) 1px, transparent 1px),
        linear-gradient(rgba(18, 36, 30, 0.035) 1px, transparent 1px),
        radial-gradient(circle at 14% 12%, rgba(244, 207, 67, 0.18), transparent 28%),
        linear-gradient(145deg, #eef2ec 0%, var(--cream) 48%, #e4ece7 100%);
      background-size: 28px 28px, 28px 28px, auto, auto;
      display: grid;
      place-items: center;
      padding:
        max(10px, env(safe-area-inset-top))
        max(10px, env(safe-area-inset-right))
        max(10px, env(safe-area-inset-bottom))
        max(10px, env(safe-area-inset-left));
    }

    main {
      width: min(760px, 100%);
      height: min(790px, 100%);
      min-height: 0;
      padding: clamp(16px, 3.2vh, 30px);
      border: 1px solid rgba(18, 36, 30, 0.2);
      border-radius: clamp(18px, 3vw, 28px);
      background: rgba(255, 253, 247, 0.94);
      box-shadow: var(--shadow);
      display: grid;
      grid-template-rows: auto auto auto minmax(72px, 1fr) auto;
      gap: clamp(8px, 1.8vh, 16px);
      overflow: hidden;
    }

    .brand {
      min-width: 0;
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr) auto;
      align-items: center;
      gap: 11px;
    }

    .brand-mark {
      width: 44px;
      height: 44px;
      border-radius: 50% 50% 46% 54%;
      background: var(--forest);
      color: var(--mustard);
      display: grid;
      place-items: center;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 23px;
      font-weight: 700;
      transform: rotate(-3deg);
    }

    .brand-copy {
      min-width: 0;
      display: grid;
      gap: 0;
      line-height: 1.04;
    }

    .brand-copy span {
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.09em;
      text-transform: uppercase;
    }

    .brand-copy strong {
      color: var(--forest);
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(15px, 2.5vw, 19px);
      font-weight: 700;
    }

    .live-tag {
      justify-self: end;
      padding: 7px 10px;
      border: 1px solid rgba(216, 100, 86, 0.32);
      border-radius: 999px;
      color: #aa4338;
      background: rgba(216, 100, 86, 0.08);
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.13em;
      white-space: nowrap;
    }

    .hero {
      min-width: 0;
      padding: clamp(13px, 2.4vh, 22px);
      border-radius: 18px;
      color: #f9fbf7;
      background:
        radial-gradient(circle at 92% 14%, rgba(244, 207, 67, 0.18), transparent 24%),
        linear-gradient(135deg, var(--forest), var(--forest-soft));
      overflow: hidden;
    }

    .eyebrow {
      margin: 0 0 6px;
      color: var(--mustard);
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.15em;
    }

    h1 {
      max-width: 18em;
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(25px, 5vw, 44px);
      font-weight: 500;
      line-height: 1.04;
      letter-spacing: -0.025em;
    }

    .intro {
      max-width: 45em;
      margin: 8px 0 0;
      color: rgba(249, 251, 247, 0.72);
      font-size: clamp(11px, 1.8vw, 14px);
      line-height: 1.45;
    }

    .status-grid {
      min-width: 0;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
    }

    .status-card {
      min-width: 0;
      min-height: 54px;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(223, 233, 227, 0.52);
      display: grid;
      align-content: center;
      gap: 2px;
    }

    .status-card span {
      color: var(--coral);
      font-size: 8px;
      font-weight: 900;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .status-card strong {
      min-width: 0;
      color: var(--forest);
      font-size: clamp(12px, 2vw, 15px);
      line-height: 1.15;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .status-card[data-state="ok"] strong { color: #226b4d; }
    .status-card[data-state="warn"] strong { color: #8a6220; }
    .status-card[data-state="error"] strong { color: #a44339; }

    .now-playing {
      min-width: 0;
      min-height: 0;
      padding: clamp(13px, 2.4vh, 21px);
      border: 1px solid rgba(18, 36, 30, 0.12);
      border-radius: 17px;
      background: linear-gradient(135deg, #e5eee8, #d9e7df);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      overflow: hidden;
    }

    .now-playing-copy {
      min-width: 0;
      min-height: 0;
      flex: 1 1 auto;
      display: grid;
      align-content: center;
      gap: 4px;
    }

    .now-playing small {
      display: block;
      margin-bottom: 4px;
      color: var(--coral);
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.14em;
    }

    .now-playing strong {
      display: block;
      min-width: 0;
      max-height: 4.6em;
      color: var(--forest);
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(17px, 3.4vw, 28px);
      font-weight: 500;
      line-height: 1.15;
      overflow: hidden;
      overflow-wrap: break-word;
      text-wrap: pretty;
      transition: color 180ms ease, opacity 180ms ease;
    }

    .now-playing[data-speaking="false"] strong {
      color: #53635c;
      opacity: 0.72;
    }

    .playback-state {
      min-width: 0;
      color: var(--muted);
      font-size: clamp(9px, 1.6vw, 12px);
      font-weight: 800;
      line-height: 1.25;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .caption-reveal {
      animation: captionReveal 220ms ease-out;
    }

    .pulse {
      width: clamp(40px, 7vw, 54px);
      height: clamp(40px, 7vw, 54px);
      flex: 0 0 auto;
      border-radius: 50%;
      background: var(--mustard);
      box-shadow: inset 0 0 0 1px rgba(18, 36, 30, 0.12);
      position: relative;
    }

    .pulse::before, .pulse::after {
      content: "";
      position: absolute;
      top: 50%;
      width: 4px;
      border-radius: 99px;
      background: var(--forest);
      transform: translateY(-50%);
    }

    .pulse::before { left: 34%; height: 29%; }
    .pulse::after { right: 34%; height: 48%; }
    .now-playing[data-playing="true"] .pulse { animation: breathe 1.4s ease-in-out infinite; }

    @keyframes breathe {
      50% { transform: scale(1.06); box-shadow: 0 0 0 7px rgba(244, 207, 67, 0.16); }
    }

    @keyframes captionReveal {
      from { opacity: 0.58; transform: translateY(3px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
      .caption-reveal, .now-playing[data-playing="true"] .pulse { animation: none; }
    }

    .controls {
      min-width: 0;
      display: grid;
      gap: 8px;
    }

    .playback-settings {
      min-height: 46px;
      padding: 6px 7px 6px 13px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(244, 245, 241, 0.9);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .playback-settings label {
      color: var(--forest);
      font-size: 12px;
      font-weight: 850;
      letter-spacing: 0.03em;
    }

    .playback-settings select {
      min-width: 104px;
      height: 34px;
      padding: 0 30px 0 12px;
      border: 1px solid rgba(18, 36, 30, 0.24);
      border-radius: 9px;
      color: var(--forest);
      background: var(--paper);
      font: inherit;
      font-size: 12px;
      font-weight: 850;
      cursor: pointer;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    button {
      min-width: 0;
      min-height: 46px;
      padding: 8px 12px;
      border: 1px solid transparent;
      border-radius: 11px;
      font: inherit;
      font-size: 12px;
      font-weight: 900;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
    }

    button:hover:not(:disabled) { transform: translateY(-1px); }
    button:disabled { cursor: default; opacity: 0.42; }

    #startListening {
      color: var(--forest);
      background: var(--mustard);
      box-shadow: 0 8px 20px rgba(189, 148, 17, 0.2);
    }

    #stopListening {
      color: var(--cream);
      background: var(--forest);
    }

    #resumeListening {
      grid-column: 1 / -1;
      color: #fffdf7;
      background: var(--coral);
    }

    #resumeListening[hidden] { display: none; }

    @media (max-width: 480px) {
      body {
        padding:
          max(6px, env(safe-area-inset-top))
          max(6px, env(safe-area-inset-right))
          max(6px, env(safe-area-inset-bottom))
          max(6px, env(safe-area-inset-left));
      }
      main { padding: 12px; border-radius: 16px; gap: 8px; }
      .brand { grid-template-columns: 38px minmax(0, 1fr) auto; gap: 8px; }
      .brand-mark { width: 38px; height: 38px; font-size: 20px; }
      .brand-copy span { display: none; }
      .brand-copy strong { font-size: 14px; }
      .live-tag { padding: 6px 7px; font-size: 7px; }
      .hero { padding: 13px; }
      h1 { font-size: clamp(23px, 8.2vw, 30px); }
      .intro { font-size: 10px; line-height: 1.35; }
      .status-card { min-height: 48px; padding: 7px 8px; }
      .status-card span { font-size: 7px; }
      .status-card strong { font-size: 11px; }
      .now-playing { padding: 12px; }
      .now-playing strong { font-size: 19px; }
    }

    @media (max-height: 650px) {
      main { padding: 11px; gap: 7px; }
      .hero { padding: 11px 13px; }
      .intro { display: none; }
      .status-card { min-height: 44px; padding-block: 6px; }
      .now-playing { padding: 10px 13px; }
      .playback-settings { min-height: 40px; }
      button { min-height: 40px; }
    }

    @media (min-width: 700px) and (max-height: 500px) {
      main {
        width: min(1020px, 100%);
        grid-template-columns: minmax(240px, 0.8fr) minmax(420px, 1.35fr);
        grid-template-rows: auto auto minmax(0, 1fr);
        column-gap: 12px;
        row-gap: 7px;
      }
      .brand { grid-column: 1; grid-row: 1; }
      .hero { grid-column: 1; grid-row: 2 / 4; align-self: stretch; }
      .hero h1 { font-size: clamp(27px, 4.2vw, 38px); }
      .status-grid { grid-column: 2; grid-row: 1; }
      .now-playing { grid-column: 2; grid-row: 2; min-height: 58px; }
      .controls { grid-column: 2; grid-row: 3; align-self: end; }
      .controls { grid-template-columns: minmax(170px, 0.52fr) minmax(0, 1fr); }
      .actions { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      #resumeListening { grid-column: 1 / -1; }
      .brand-copy strong { font-size: 15px; }
      .brand-copy span, .intro { display: none; }
      .playback-settings { height: 100%; }
    }

    @media (max-height: 430px) and (min-width: 700px) {
      .brand-mark { width: 36px; height: 36px; font-size: 19px; }
      .brand { grid-template-columns: 36px minmax(0, 1fr); }
      .live-tag { display: none; }
      .eyebrow { margin-bottom: 4px; }
      .status-card { min-height: 40px; }
      .now-playing small { display: none; }
    }
  </style>
</head>
<body>
  <main>
    <header class="brand">
      <div class="brand-mark" aria-hidden="true">P</div>
      <div class="brand-copy">
        <span>Pittsburgh Christian Church</span>
        <strong>Pittsburgh Christian Church South</strong>
      </div>
      <div class="live-tag">LIVE TRANSLATION</div>
    </header>

    <section class="hero">
      <p class="eyebrow">SUNDAY WORSHIP / SHARED AUDIO</p>
      <h1>Hear the message in your language.</h1>
      <p class="intro">Start once, then keep listening from the lock screen. Every device joins the same live translated-audio stream.</p>
    </section>

    <section class="status-grid" aria-live="polite">
      <div id="connectionCard" class="status-card" data-state="warn">
        <span>Connection</span><strong id="connectionStatus">Not started</strong>
      </div>
      <div id="producerCard" class="status-card" data-state="warn">
        <span>Service</span><strong id="producerStatus">Waiting</strong>
      </div>
      <div id="queueCard" class="status-card">
        <span>Listeners</span><strong id="queueStatus">Not joined</strong>
      </div>
    </section>

    <section id="nowPlaying" class="now-playing" data-playing="false" data-speaking="false">
      <div class="now-playing-copy">
        <small>LIVE AUDIO</small>
        <strong id="liveCaption" aria-live="polite" aria-atomic="true">Waiting to start</strong>
        <span id="playbackStatus" class="playback-state">Start listening to join the shared stream</span>
      </div>
      <div class="pulse" aria-hidden="true"></div>
    </section>

    <audio id="ttsPlayback" preload="none" playsinline hidden></audio>

    <section class="controls">
      <div class="playback-settings">
        <label for="playbackRate">Playback speed</label>
        <select id="playbackRate" aria-label="Playback speed">
          <option value="0.8">0.8x</option>
          <option value="0.9">0.9x</option>
          <option value="1" selected>1.0x</option>
          <option value="1.1">1.1x</option>
          <option value="1.2">1.2x</option>
        </select>
      </div>
      <div class="actions">
        <button id="startListening" type="button">Start Listening</button>
        <button id="stopListening" type="button" disabled>Stop Listening</button>
        <button id="resumeListening" type="button" hidden>Resume Audio</button>
      </div>
    </section>
  </main>

  <script>
  (() => {
    const startButton = document.getElementById("startListening");
    const stopButton = document.getElementById("stopListening");
    const resumeButton = document.getElementById("resumeListening");
    const connectionCard = document.getElementById("connectionCard");
    const producerCard = document.getElementById("producerCard");
    const connectionStatus = document.getElementById("connectionStatus");
    const producerStatus = document.getElementById("producerStatus");
    const queueStatus = document.getElementById("queueStatus");
    const liveCaption = document.getElementById("liveCaption");
    const playbackStatus = document.getElementById("playbackStatus");
    const nowPlaying = document.getElementById("nowPlaying");
    const playbackRateInput = document.getElementById("playbackRate");
    const playbackElement = document.getElementById("ttsPlayback");
    const PLAYBACK_RATE_STORAGE_KEY = "voxbridge.ttsPlaybackRate";
    const SUPPORTED_PLAYBACK_RATES = new Set([0.8, 0.9, 1, 1.1, 1.2]);
    const CATCH_UP_START_LAG_SEC = 12;
    const CATCH_UP_STOP_LAG_SEC = 5;
    const CATCH_UP_RATE = 1.2;
    const CAPTION_POLL_INTERVAL_MS = 500;

    let playbackRate = readPlaybackRate();
    let listenerId = "";
    let running = false;
    let statusTimer = null;
    let captionTimer = null;
    let captionAbortController = null;
    let catchingUp = false;
    let captionSnapshot = null;
    let captionCueId = "";
    playbackRateInput.value = String(playbackRate);

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

    function setCard(card, value) {
      card.dataset.state = value || "";
    }

    function liveLagSec() {
      const ranges = playbackElement.seekable;
      if (!ranges || ranges.length < 1 || !Number.isFinite(playbackElement.currentTime)) {
        return null;
      }
      const liveEdge = Number(ranges.end(ranges.length - 1));
      if (!Number.isFinite(liveEdge)) return null;
      return Math.max(0, liveEdge - playbackElement.currentTime);
    }

    function estimatedPlaybackAtMs(snapshot) {
      const liveEdgeAtMs = Number(snapshot && snapshot.live_edge_at_ms);
      const lag = liveLagSec();
      if (!Number.isFinite(liveEdgeAtMs) || lag === null) return null;
      return liveEdgeAtMs - lag * 1000;
    }

    function revealCaption() {
      liveCaption.classList.remove("caption-reveal");
      void liveCaption.offsetWidth;
      liveCaption.classList.add("caption-reveal");
    }

    function setLiveCaption(text, cueId = "") {
      const nextText = String(text || "").trim();
      if (!nextText) return;
      const nextCueId = String(cueId || "");
      if (liveCaption.textContent === nextText && captionCueId === nextCueId) return;
      liveCaption.textContent = nextText;
      captionCueId = nextCueId;
      revealCaption();
    }

    function applyCaptionSnapshot(snapshot, requestListenerId) {
      if (!running || !requestListenerId || requestListenerId !== listenerId) return;
      const playheadAtMs = estimatedPlaybackAtMs(snapshot);
      if (playheadAtMs === null) return;
      const cues = Array.isArray(snapshot && snapshot.cues) ? snapshot.cues : [];
      let selected = null;
      for (const cue of cues) {
        const startAtMs = Number(cue && cue.start_at_ms);
        if (!Number.isFinite(startAtMs) || startAtMs > playheadAtMs) continue;
        if (selected === null || startAtMs >= Number(selected.start_at_ms)) {
          selected = cue;
        }
      }
      if (selected === null) {
        if (!captionCueId) setLiveCaption("Waiting for translated speech");
        nowPlaying.dataset.speaking = "false";
        return;
      }
      setLiveCaption(selected.text, selected.cue_id);
      const endAtMs = Number(selected.end_at_ms);
      nowPlaying.dataset.speaking = String(
        nowPlaying.dataset.playing === "true"
          && Number.isFinite(endAtMs)
          && playheadAtMs < endAtMs
      );
    }

    function refreshCaptionForPlayhead() {
      if (captionSnapshot !== null) {
        applyCaptionSnapshot(captionSnapshot, listenerId);
      }
    }

    async function pollCaptions() {
      if (!running || document.hidden || !listenerId || captionAbortController) return;
      const requestListenerId = listenerId;
      const controller = new AbortController();
      captionAbortController = controller;
      try {
        const response = await fetch(
          `/api/tts/live/${encodeURIComponent(requestListenerId)}/captions`,
          {
            credentials: "same-origin",
            cache: "no-store",
            signal: controller.signal,
          }
        );
        if (!response.ok) throw new Error(`caption request failed: ${response.status}`);
        const snapshot = await response.json();
        if (!running || requestListenerId !== listenerId) return;
        captionSnapshot = snapshot;
        applyCaptionSnapshot(snapshot, requestListenerId);
      } catch (error) {
        // Caption metadata is advisory; native HLS playback remains independent.
      } finally {
        if (captionAbortController === controller) {
          captionAbortController = null;
        }
      }
    }

    function effectivePlaybackRate() {
      if (catchingUp) return CATCH_UP_RATE;
      if (document.hidden) return Math.max(1, playbackRate);
      return playbackRate;
    }

    function applyPlaybackRate() {
      const effectiveRate = effectivePlaybackRate();
      playbackElement.defaultPlaybackRate = effectiveRate;
      playbackElement.playbackRate = effectiveRate;
      if ("preservesPitch" in playbackElement) playbackElement.preservesPitch = true;
      if ("mozPreservesPitch" in playbackElement) playbackElement.mozPreservesPitch = true;
      if ("webkitPreservesPitch" in playbackElement) {
        playbackElement.webkitPreservesPitch = true;
      }
    }

    function updateLiveLatencyGuard() {
      if (!running) return;
      const lag = liveLagSec();
      if (lag !== null) {
        if (!catchingUp && lag >= CATCH_UP_START_LAG_SEC) catchingUp = true;
        else if (catchingUp && lag <= CATCH_UP_STOP_LAG_SEC) catchingUp = false;
      }
      applyPlaybackRate();
      if (catchingUp && lag !== null) {
        playbackStatus.textContent = `Catching up / ${Math.ceil(lag)}s behind live`;
      } else {
        playbackStatus.textContent = "Listening to live translation";
      }
    }

    function createListenerId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return `iphone-${window.crypto.randomUUID()}`;
      }
      const bytes = new Uint8Array(16);
      window.crypto.getRandomValues(bytes);
      return `iphone-${Array.from(bytes, (value) =>
        value.toString(16).padStart(2, "0")
      ).join("")}`;
    }

    function setMediaPlaybackState(state) {
      if ("mediaSession" in navigator) {
        navigator.mediaSession.playbackState = state;
      }
    }

    function markPlaying() {
      if (!running) return;
      resumeButton.hidden = true;
      connectionStatus.textContent = "Connected";
      setCard(connectionCard, "ok");
      playbackStatus.textContent = "Listening to live translation";
      nowPlaying.dataset.playing = "true";
      beginCaptionPolling();
      refreshCaptionForPlayhead();
      setMediaPlaybackState("playing");
    }

    function markPlaybackBlocked(error) {
      if (!running) return;
      const blocked = error && error.name === "NotAllowedError";
      connectionStatus.textContent = blocked ? "Tap to continue" : "Audio unavailable";
      setCard(connectionCard, blocked ? "warn" : "error");
      playbackStatus.textContent = blocked
        ? "Tap Resume Audio below"
        : "The shared stream is temporarily unavailable";
      nowPlaying.dataset.playing = "false";
      nowPlaying.dataset.speaking = "false";
      resumeButton.hidden = false;
      setMediaPlaybackState("paused");
    }

    async function pollStatus() {
      if (!running) return;
      try {
        const response = await fetch("/api/tts/live/status", {
          credentials: "same-origin",
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`status request failed: ${response.status}`);
        const status = await response.json();
        if (!running) return;
        producerStatus.textContent = status.producer_active ? "Service live" : "Waiting for service";
        setCard(producerCard, status.producer_active ? "ok" : "warn");
        const listeners = Number(status.listener_count || 0);
        const pendingAudioSec = Math.ceil(Number(status.pending_audio_ms || 0) / 1000);
        queueStatus.textContent = status.encoder_active
          ? `Live / ${listeners} listeners${pendingAudioSec > 0 ? ` / ${pendingAudioSec}s queued` : ""}`
          : "Preparing stream";
      } catch (error) {
        if (!running) return;
        producerStatus.textContent = "Status unavailable";
        setCard(producerCard, "warn");
      }
    }

    function beginStatusPolling() {
      if (statusTimer !== null) window.clearInterval(statusTimer);
      void pollStatus();
      statusTimer = window.setInterval(() => void pollStatus(), 5000);
    }

    function stopStatusPolling() {
      if (statusTimer === null) return;
      window.clearInterval(statusTimer);
      statusTimer = null;
    }

    function beginCaptionPolling() {
      if (captionTimer !== null) return;
      void pollCaptions();
      captionTimer = window.setInterval(
        () => void pollCaptions(),
        CAPTION_POLL_INTERVAL_MS
      );
    }

    function stopCaptionPolling() {
      if (captionTimer !== null) {
        window.clearInterval(captionTimer);
        captionTimer = null;
      }
      if (captionAbortController !== null) {
        captionAbortController.abort();
        captionAbortController = null;
      }
    }

    function startListeningFromGesture() {
      if (running) return;
      running = true;
      listenerId = createListenerId();
      startButton.disabled = true;
      stopButton.disabled = false;
      resumeButton.hidden = true;
      connectionStatus.textContent = "Connecting";
      producerStatus.textContent = "Checking service";
      queueStatus.textContent = "Joining live stream";
      playbackStatus.textContent = "Starting audio";
      setLiveCaption("Waiting for translated speech");
      nowPlaying.dataset.speaking = "false";
      setCard(connectionCard, "warn");
      setCard(producerCard, "warn");

      playbackElement.muted = false;
      playbackElement.playsInline = true;
      playbackElement.src =
        `/api/tts/live/${encodeURIComponent(listenerId)}/index.m3u8`;
      applyPlaybackRate();

      // iOS requires the native stream to start directly inside the user's gesture.
      const playPromise = playbackElement.play();
      beginStatusPolling();
      if (playPromise) {
        playPromise.then(markPlaying).catch(markPlaybackBlocked);
      } else {
        markPlaying();
      }
    }

    function resumeListeningFromGesture() {
      if (!running || !playbackElement.src) return;
      resumeButton.hidden = true;
      connectionStatus.textContent = "Restoring audio";
      setCard(connectionCard, "warn");
      const playPromise = playbackElement.play();
      if (playPromise) {
        playPromise.then(markPlaying).catch(markPlaybackBlocked);
      } else {
        markPlaying();
      }
    }

    function releaseListenerLease(id) {
      if (!id) return;
      fetch(`/api/tts/live/${encodeURIComponent(id)}`, {
        method: "DELETE",
        credentials: "same-origin",
        keepalive: true,
      }).catch(() => {});
    }

    function stopListening() {
      const closingListenerId = listenerId;
      listenerId = "";
      running = false;
      catchingUp = false;
      stopStatusPolling();
      stopCaptionPolling();
      captionSnapshot = null;
      captionCueId = "";
      playbackElement.pause();
      playbackElement.removeAttribute("src");
      playbackElement.load();
      releaseListenerLease(closingListenerId);
      resumeButton.hidden = true;
      nowPlaying.dataset.playing = "false";
      nowPlaying.dataset.speaking = "false";
      connectionStatus.textContent = "Stopped";
      producerStatus.textContent = "Waiting";
      queueStatus.textContent = "Not joined";
      liveCaption.textContent = "Waiting to start";
      playbackStatus.textContent = "Start listening to join the shared stream";
      setCard(connectionCard, "warn");
      setCard(producerCard, "warn");
      startButton.disabled = false;
      stopButton.disabled = true;
      setMediaPlaybackState("none");
    }

    function configureMediaSession() {
      if (!("mediaSession" in navigator)) return;
      try {
        navigator.mediaSession.metadata = new MediaMetadata({
          title: "PCCS Live Translation",
          artist: "Pittsburgh Christian Church South",
          album: "Sunday Worship",
        });
      } catch (error) {}
      try {
        navigator.mediaSession.setActionHandler("play", () => {
          if (!running) startListeningFromGesture();
          else resumeListeningFromGesture();
        });
      } catch (error) {}
      try {
        navigator.mediaSession.setActionHandler("pause", () => {
          playbackElement.pause();
          nowPlaying.dataset.playing = "false";
          nowPlaying.dataset.speaking = "false";
          playbackStatus.textContent = "Paused / resume from the lock screen";
          setMediaPlaybackState("paused");
        });
      } catch (error) {}
    }

    applyPlaybackRate();
    configureMediaSession();

    startButton.addEventListener("click", startListeningFromGesture);
    resumeButton.addEventListener("click", resumeListeningFromGesture);
    stopButton.addEventListener("click", stopListening);
    playbackElement.addEventListener("playing", markPlaying);
    playbackElement.addEventListener("waiting", () => {
      if (!running) return;
      playbackStatus.textContent = "Buffering live audio";
      nowPlaying.dataset.playing = "false";
      nowPlaying.dataset.speaking = "false";
    });
    playbackElement.addEventListener("stalled", () => {
      if (!running) return;
      playbackStatus.textContent = "Reconnecting to live audio";
      nowPlaying.dataset.playing = "false";
      nowPlaying.dataset.speaking = "false";
    });
    playbackElement.addEventListener("error", () => {
      if (!running) return;
      markPlaybackBlocked(playbackElement.error || new Error("media error"));
    });
    playbackRateInput.addEventListener("change", () => {
      playbackRate = normalizePlaybackRate(playbackRateInput.value);
      playbackRateInput.value = String(playbackRate);
      try {
        window.localStorage.setItem(PLAYBACK_RATE_STORAGE_KEY, String(playbackRate));
      } catch (error) {}
      applyPlaybackRate();
    });
    playbackElement.addEventListener("timeupdate", updateLiveLatencyGuard);
    playbackElement.addEventListener("timeupdate", refreshCaptionForPlayhead);
    playbackElement.addEventListener("progress", updateLiveLatencyGuard);
    playbackElement.addEventListener("progress", refreshCaptionForPlayhead);
    document.addEventListener("visibilitychange", () => {
      updateLiveLatencyGuard();
      if (!document.hidden) void pollCaptions();
    });
    window.addEventListener("beforeunload", () => {
      stopStatusPolling();
      stopCaptionPolling();
      releaseListenerLease(listenerId);
      playbackElement.pause();
    });
  })();
  </script>
</body>
</html>
"""


__all__ = ["TTS_LISTENER_HTML"]
