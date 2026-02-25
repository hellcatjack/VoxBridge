# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Browser microphone streaming demo over WebSocket (vLLM backend).
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import signal
import socket
import threading
import time
import re
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from voxbridge.streaming.backpressure import QueueBackpressureController
from voxbridge.streaming.segment_policy import SegmentPolicy
from voxbridge.streaming.text_pool import dedup_segment_join, trim_prefix_overlap

SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"[。！？!?…]+[\"'”’)\]）】》]*|\.+(?=\s|$)[\"'”’)\]）】》]*")
MIN_CJK_SENTENCE_CHARS = 10
_INSTANCE_LOCK_HANDLE: Optional[Any] = None


def _instance_lock_path(port: int) -> Path:
    safe_port = int(port)
    return Path("/tmp") / f"voxbridge_demo_streaming_ws_{safe_port}.lock"


def _acquire_instance_lock_or_raise(port: int, lock_path: Optional[Path] = None):
    target = Path(lock_path) if lock_path is not None else _instance_lock_path(port)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        holder = ""
        with suppress(Exception):
            handle.seek(0)
            holder = handle.read().strip()
        with suppress(Exception):
            handle.close()
        holder_suffix = f" (holder pid: {holder})" if holder else ""
        raise RuntimeError(
            f"another demo_streaming_ws instance is already running for port {int(port)}{holder_suffix}"
        ) from exc
    handle.seek(0)
    handle.truncate(0)
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _release_instance_lock(handle) -> None:
    if handle is None:
        return
    with suppress(Exception):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with suppress(Exception):
        handle.close()


def _assert_port_bindable(host: str, port: int) -> None:
    bind_host = str(host or "0.0.0.0").strip() or "0.0.0.0"
    if bind_host == "*":
        bind_host = "0.0.0.0"
    bind_port = int(port)
    try:
        addr_infos = socket.getaddrinfo(
            bind_host,
            bind_port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
            flags=socket.AI_PASSIVE,
        )
    except socket.gaierror as exc:
        raise RuntimeError(f"invalid bind host '{bind_host}': {exc}") from exc

    last_error: Optional[OSError] = None
    for family, socktype, proto, _, sockaddr in addr_infos:
        probe = socket.socket(family, socktype, proto)
        with suppress(OSError):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(sockaddr)
            return
        except OSError as exc:
            last_error = exc
        finally:
            probe.close()

    if last_error is None:
        raise RuntimeError(f"bind {bind_host}:{bind_port} is not available")
    raise RuntimeError(f"bind {bind_host}:{bind_port} is not available: {last_error}") from last_error


def _list_orphan_enginecore_pids(
    proc_root: Optional[Path] = None,
    current_uid: Optional[int] = None,
) -> List[int]:
    root = Path(proc_root) if proc_root is not None else Path("/proc")
    owner_uid = os.getuid() if current_uid is None else int(current_uid)
    out: List[int] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        name = entry.name
        if not name.isdigit():
            continue
        status_path = entry / "status"
        try:
            text = status_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        status: Dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            status[k.strip()] = v.strip()

        proc_name = str(status.get("Name", ""))
        if not proc_name.startswith("VLLM::EngineCor"):
            continue
        try:
            ppid = int(str(status.get("PPid", "0")).split()[0])
        except ValueError:
            continue
        if ppid != 1:
            continue
        uid_row = str(status.get("Uid", ""))
        if not uid_row:
            continue
        try:
            proc_uid = int(uid_row.split()[0])
        except (IndexError, ValueError):
            continue
        if proc_uid != owner_uid:
            continue
        out.append(int(name))
    return out


def _cleanup_orphan_enginecore_processes(grace_sec: float = 1.2) -> List[int]:
    stale_pids = _list_orphan_enginecore_pids()
    if not stale_pids:
        return []
    for pid in stale_pids:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(int(pid), signal.SIGTERM)

    deadline = time.monotonic() + max(0.05, float(grace_sec))
    alive = set(int(pid) for pid in stale_pids)
    while alive and time.monotonic() < deadline:
        for pid in list(alive):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive.discard(pid)
            except PermissionError:
                alive.discard(pid)
        if alive:
            time.sleep(0.05)

    for pid in list(alive):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    return stale_pids


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))


def _is_chinese_label(text: Any) -> bool:
    ln = str(text or "").strip().lower()
    if not ln:
        return False
    return ("chinese" in ln) or ("中文" in ln) or (ln in {"zh", "zh-cn", "zh-hans", "zh-hant"})


def _is_english_label(text: Any) -> bool:
    ln = str(text or "").strip().lower()
    if not ln:
        return False
    return ("english" in ln) or ("英文" in ln) or (ln in {"en", "en-us", "en-gb"})


def _text_matches_source_language(text: str, source_language: str) -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    if _is_chinese_label(source_language):
        return _has_cjk(src)
    if _is_english_label(source_language):
        return _has_latin(src)
    return True


def _split_sentences_and_tail(text: str) -> Tuple[List[str], str]:
    src = str(text or "").strip()
    if not src:
        return [], ""

    raw_sentences: List[str] = []
    last = 0
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(src):
        end = match.end()
        seg = src[last:end].strip()
        if seg:
            raw_sentences.append(seg)
        last = end
    tail = src[last:].strip()

    sentences: List[str] = []
    carry = ""
    for seg in raw_sentences:
        cur = str(seg or "").strip()
        if not cur:
            continue
        if carry:
            cur = _join_segments([carry, cur])
            carry = ""
        if _has_cjk(cur) and len(cur) < MIN_CJK_SENTENCE_CHARS:
            carry = cur
            continue
        sentences.append(cur)

    if carry:
        tail = _join_segments([carry, tail]) if tail else carry

    return sentences, tail


def _join_segments(segments: List[str]) -> str:
    out = ""
    for seg in segments:
        cur = str(seg or "").strip()
        if not cur:
            continue
        if not out:
            out = cur
            continue
        need_space = bool(re.match(r"[A-Za-z0-9]", out[-1])) and bool(re.match(r"[A-Za-z0-9]", cur[:1]))
        out = f"{out} {cur}" if need_space else f"{out}{cur}"
    return out


class LocalTranslator:
    """
    Lightweight local translation wrapper for zh->en real-time subtitles.

    This uses a local causal LM translation model and generates deterministic output.
    """

    def __init__(
        self,
        model_path: str,
        source_language: str = "Chinese",
        target_language: str = "English",
        max_new_tokens: int = 96,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.source_language = str(source_language or "Chinese")
        self.target_language = str(target_language or "English")
        self.max_new_tokens = max(8, int(max_new_tokens))

        resolved_device = str(device or "cpu").strip().lower()
        if resolved_device not in {"cpu", "cuda", "auto"}:
            resolved_device = "cpu"
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("translation device is cuda but torch.cuda is not available")
        self.device = resolved_device

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.device == "cuda":
            bf16_ok = False
            with suppress(Exception):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            model_kwargs["dtype"] = torch.bfloat16 if bf16_ok else torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._lock = threading.Lock()

    def _build_prompt(self, text: str, source_language: Optional[str] = None, target_language: Optional[str] = None) -> str:
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        return (
            f"请将以下{source}文本翻译为{target}。\n"
            f"要求：忠实原文，不增删；保留专有名词；只输出译文本身，不要解释。\n\n"
            f"原文：\n{text}"
        )

    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> str:
        import torch

        src = str(text or "").strip()
        if not src:
            return ""
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        if not _text_matches_source_language(src, source):
            return ""

        messages = [{"role": "user", "content": self._build_prompt(src, source_language=source, target_language=target)}]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = tokenized_chat.to(self.model.device)

        with self._lock, torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        new_ids = outputs[0][input_ids.shape[-1]:]
        out = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not out:
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        out = out.replace("<target>", "").replace("</target>", "").strip()
        return out


class OpenAIAPITranslator:
    """
    Translation client using an OpenAI-compatible Chat Completions HTTP API.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        source_language: str = "Chinese",
        target_language: str = "English",
        max_new_tokens: int = 96,
        timeout_sec: float = 30.0,
        api_key: str = "",
    ) -> None:
        self.base_url = str(base_url or "").strip()
        if not self.base_url:
            raise ValueError("translation api base_url is empty")
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("translation api model is empty")
        self.source_language = str(source_language or "Chinese")
        self.target_language = str(target_language or "English")
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.api_key = str(api_key or "").strip()
        self._lock = threading.Lock()

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            self.chat_url = normalized
        elif normalized.endswith("/v1"):
            self.chat_url = f"{normalized}/chat/completions"
        else:
            self.chat_url = f"{normalized}/v1/chat/completions"

    def _build_prompt(self, text: str, source_language: Optional[str] = None, target_language: Optional[str] = None) -> str:
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        return (
            f"请将以下{source}文本翻译为{target}。\n"
            f"要求：忠实原文，不增删；保留专有名词；只输出译文本身，不要解释。\n\n"
            f"原文：\n{text}"
        )

    def _extract_content(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str):
                            chunks.append(txt)
                return "".join(chunks).strip()
        return ""

    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        if not _text_matches_source_language(src, source):
            return ""

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_prompt(
                        src,
                        source_language=source,
                        target_language=target,
                    ),
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.chat_url, data=data, headers=headers, method="POST")
        with self._lock:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        out = self._extract_content(payload)
        out = out.replace("<target>", "").replace("</target>", "").strip()
        return out


INDEX_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>语音识别与翻译</title>
  <style>
    :root{
      --bg-a:#0f1114;
      --bg-b:#23252a;
      --ink:#f3f3f2;
      --line:#3b4048;
      --ok:#29b26b;
      --warn:#cc8f28;
      --err:#d75858;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body{
      margin:0;
      font-family: "Avenir Next", "Segoe UI", "Noto Sans SC", "PingFang SC", sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at 18% 16%, #3e4552 0%, transparent 36%),
        radial-gradient(circle at 83% 82%, #2f2622 0%, transparent 34%),
        linear-gradient(160deg, var(--bg-a), var(--bg-b));
    }

    .wrap{
      height: 100%;
      padding: 0;
      display: grid;
      place-items: stretch;
      overflow: hidden;
    }

    .card{
      width: 100vw;
      height: 100vh;
      border:0;
      border-radius: 0;
      background: linear-gradient(180deg, rgba(24, 28, 35, 0.85), rgba(12, 14, 18, 0.94));
      padding: 14px 14px 12px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 14px;
      overflow: hidden;
    }

    h1{
      margin:0;
      font-size: 16px;
      letter-spacing: .8px;
      font-weight: 700;
      color:#d8dde7;
    }

    .row{ display:flex; gap:10px; align-items:center; flex-wrap: wrap; }

    button{
      border:1px solid var(--line);
      border-radius: 10px;
      background: #2a2f36;
      color: #f4f5f6;
      font-weight: 700;
      padding: 9px 15px;
      cursor: pointer;
      transition: background .15s ease, transform .04s ease;
    }
    button:hover{ background:#353b44; }
    button:active{ transform: translateY(1px); }
    button:disabled{ opacity:.55; cursor:not-allowed; }
    button.primary{ border-color:#4f8adf; background:#2b5aa0; }
    button.danger{ border-color:#a44a4a; background:#7d3030; }

    .badge{
      border:1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: #d8dee6;
      background:#272c34;
    }
    .ok{ color: var(--ok); border-color: #3c7f5f; background:#203c31; }
    .warn{ color: var(--warn); border-color: #8b6a35; background:#3c3020; }
    .err{ color: var(--err); border-color: #8f3f3f; background:#3f2525; }

    .direction-toggle{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border:1px solid var(--line);
      border-radius: 10px;
      background:#272c34;
      padding: 7px 10px;
      font-size: 12px;
      color:#d8dde6;
      user-select: none;
    }

    .direction-toggle input{
      margin: 0;
      accent-color: #4f8adf;
      width: 16px;
      height: 16px;
    }

    .subtitle-stage{
      position: relative;
      border:1px solid rgba(255,255,255,0.1);
      border-radius: 14px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(9, 10, 12, 0.02) 0%, rgba(9, 10, 12, 0.76) 68%, rgba(8, 8, 9, 0.96) 100%),
        radial-gradient(circle at 50% -12%, rgba(255, 255, 255, 0.13), transparent 60%),
        linear-gradient(180deg, #2b3139, #14171c 42%, #0b0d10);
      min-height: 0;
      height: 100%;
      display: grid;
      grid-template-rows: 2fr 1fr;
      align-items: stretch;
      padding: 0;
      gap: 0;
    }

    .subtitle-lane{
      position: relative;
      min-height: 0;
      overflow: hidden;
    }

    .subtitle-lane + .subtitle-lane{
      border-top: 1px solid rgba(255,255,255,0.08);
    }

    .subtitle-stack{
      width: 100%;
      height: 100%;
      text-align: center;
      white-space: pre-wrap;
      line-height: 2.3;
      word-break: break-word;
      overflow-wrap: anywhere;
      text-wrap: pretty;
      user-select: text;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-width: none;
      -ms-overflow-style: none;
      padding: 10px 10px 14px;
    }

    .subtitle-stack::-webkit-scrollbar{
      display: none;
      width: 0;
      height: 0;
    }

    .subtitle-line{
      display: block;
      min-height: 1.2em;
    }

    .line-enter{
      animation: subtitle-rise 220ms cubic-bezier(0.2, 0.9, 0.25, 1.0);
    }

    @keyframes subtitle-rise{
      from{
        opacity: 0;
        transform: translateY(12px);
      }
      to{
        opacity: 1;
        transform: translateY(0);
      }
    }

    #translation{
      min-height: 52px;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      font-size: clamp(28px, 3.3vw, 42px);
      font-weight: 750;
      color: #fcfcf7;
      letter-spacing: 0.02em;
      text-shadow:
        0 1px 0 rgba(0, 0, 0, 0.95),
        0 3px 10px rgba(0, 0, 0, 0.75),
        0 0 20px rgba(0, 0, 0, 0.45);
    }

    #text{
      min-height: 34px;
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: clamp(18px, 2.5vw, 28px);
      font-weight: 560;
      color: #f6d9a5;
      text-shadow:
        0 1px 0 rgba(0, 0, 0, 0.95),
        0 2px 8px rgba(0, 0, 0, 0.75);
    }

    #lang{
      display: none;
    }

    @media (max-width: 720px){
      .card{
        min-height: 100vh;
      }
      .subtitle-stage{
        min-height: 0;
        height: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>语音识别与翻译</h1>

      <div class="row">
        <button id="btnStart" class="primary">Start</button>
        <button id="btnStop" class="danger" disabled>Stop</button>
        <span id="status" class="badge warn">Idle</span>
        <label class="direction-toggle" for="translationDirectionToggle">
          <input id="translationDirectionToggle" type="checkbox" />
          <span id="translationDirectionLabel">中文->英文</span>
        </label>
      </div>

      <div class="subtitle-stage">
        <div class="subtitle-lane">
          <div id="translation" class="subtitle-stack"></div>
        </div>
        <div class="subtitle-lane">
          <div id="text" class="subtitle-stack"></div>
        </div>
      </div>
      <div id="lang">-</div>
    </div>
  </div>

<script>
(() => {
  const TARGET_SR = 16000;
  const CHUNK_MS = __CHUNK_MS__;
  const CHUNK_SAMPLES = Math.max(1, Math.round(TARGET_SR * CHUNK_MS / 1000));
  const MAX_WS_BUFFERED_BYTES = 1024 * 1024;
  const MAX_SEND_QUEUE_BYTES = 2 * 1024 * 1024;
  const WEBSOCKET_DRAIN_TIMEOUT_MS = 4000;
  const TAIL_STABILIZE_MS = 700;
  const SUBTITLE_KEEP_MS = 20000;
  const MAX_VISIBLE_ROWS = 4;
  const USE_COMMITTED_SENTENCE_EVENTS = true;
  const SUBTITLE_TRACE_DEFAULT = __SUBTITLE_TRACE__;
  const SUBTITLE_TRACE_MAX_EVENTS = __SUBTITLE_TRACE_MAX_EVENTS__;

  const $ = (id) => document.getElementById(id);
  const btnStart = $("btnStart");
  const btnStop = $("btnStop");
  const statusEl = $("status");
  const langEl = $("lang");
  const textEl = $("text");
  const translationEl = $("translation");
  const translationDirectionToggle = $("translationDirectionToggle");
  const translationDirectionLabel = $("translationDirectionLabel");
  const rawTextEl = $("rawText");
  const languageSelect = $("languageSelect");
  const toggleEchoCancellation = $("toggleEchoCancellation");
  const toggleNoiseSuppression = $("toggleNoiseSuppression");
  const toggleAutoGainControl = $("toggleAutoGainControl");

  let running = false;
  let ws = null;
  let audioCtx = null;
  let mediaStream = null;
  let source = null;
  let processor = null;
  let workletNode = null;
  let sinkGain = null;
  let workletModuleUrl = null;
  let pending = new Float32Array(0);
  let sendQueue = [];
  let queuedBytes = 0;
  let currentSegmentText = "";
  let subtitleSentencePairs = [];
  let currentTextTail = "";
  let currentTranslationTail = "";
  let zhLineNodes = new Map();
  let enLineNodes = new Map();
  let rawAsrText = "";
  let rawAsrLastSnapshot = "";
  let lastPartialSeq = 0;
  let awaitingFinal = false;
  let pendingFinalResolve = null;
  let pendingFinalReject = null;
  let finalTimer = null;
  let watchdogTimer = null;
  let sessionStartedAt = 0;
  let lastCaptureAt = 0;
  let lastChunkSentAt = 0;
  let lastPartialAt = 0;
  let tailStabilizeTimer = null;
  let subtitleTraceEnabled = false;
  let subtitleTraceSeq = 0;
  let subtitleTraceEvents = [];
  let lastPartialTraceSeq = -1;
  let translationDirection = "zh2en";

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  (() => {
    let enabled = !!SUBTITLE_TRACE_DEFAULT;
    try {
      const params = new URLSearchParams(location.search || "");
      const queryFlag = String(params.get("subtitle_trace") || params.get("trace") || "").trim().toLowerCase();
      if (["1", "true", "on", "yes"].includes(queryFlag)) enabled = true;
      if (["0", "false", "off", "no"].includes(queryFlag)) enabled = false;
      const saved = String(localStorage.getItem("subtitle_trace") || "").trim().toLowerCase();
      if (["1", "true", "on", "yes"].includes(saved)) enabled = true;
      if (["0", "false", "off", "no"].includes(saved)) enabled = false;
    } catch (err) {}
    subtitleTraceEnabled = enabled;
  })();

  applyTranslationDirection("zh2en", { silent: true });

  function traceSubtitle(event, payload = {}, force = false){
    if (!subtitleTraceEnabled && !force) return;
    const cap = Math.max(200, Number(SUBTITLE_TRACE_MAX_EVENTS || 1200));
    const row = Object.assign(
      {
        idx: ++subtitleTraceSeq,
        ts: Date.now(),
        event: String(event || ""),
      },
      (payload && typeof payload === "object") ? payload : { value: payload },
    );
    subtitleTraceEvents.push(row);
    if (subtitleTraceEvents.length > cap) {
      subtitleTraceEvents.splice(0, subtitleTraceEvents.length - cap);
    }
    if (subtitleTraceEnabled) {
      try {
        console.debug("[subtitle-trace]", row);
      } catch (err) {}
    }
  }

  function clearTailStabilizeTimer(){
    if (tailStabilizeTimer) {
      clearTimeout(tailStabilizeTimer);
      tailStabilizeTimer = null;
    }
  }

  function clearCommittedTentativeTailNow(){
    clearTailStabilizeTimer();
    if (currentTextTail) {
      traceSubtitle("tail_cleared", { by: "clearCommittedTentativeTailNow", prevLen: currentTextTail.length });
    }
    currentTextTail = "";
  }

  function updateCommittedTentativeTail(tailText){
    const nextTail = String(tailText || "").trim();
    if (nextTail) {
      clearTailStabilizeTimer();
      if (nextTail !== currentTextTail) {
        traceSubtitle("tail_set", { by: "updateCommittedTentativeTail", nextLen: nextTail.length });
      }
      currentTextTail = nextTail;
      return;
    }
    if (!currentTextTail) {
      clearTailStabilizeTimer();
      currentTextTail = "";
      return;
    }
    if (tailStabilizeTimer) return;
    tailStabilizeTimer = setTimeout(() => {
      tailStabilizeTimer = null;
      if (!currentTextTail) return;
      traceSubtitle("tail_stabilize_expired", { prevLen: currentTextTail.length });
      currentTextTail = "";
      renderTranscript();
      renderTranslation();
    }, TAIL_STABILIZE_MS);
  }

  function isLocalhost(){
    return (
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1" ||
      location.hostname === "::1"
    );
  }

  function setStatus(msg, cls){
    const prev = String(statusEl.textContent || "");
    statusEl.textContent = msg;
    statusEl.className = "badge " + (cls || "");
    if (prev !== msg) {
      traceSubtitle("status_changed", { prev, next: String(msg || ""), cls: String(cls || "") });
    }
  }

  function normalizeTranslationDirection(raw){
    const text = String(raw || "").trim().toLowerCase();
    if (text === "en2zh" || text === "en->zh") return "en2zh";
    return "zh2en";
  }

  function selectedTranslationDirection(){
    if (!translationDirectionToggle) return "zh2en";
    return translationDirectionToggle.checked ? "en2zh" : "zh2en";
  }

  function applyTranslationDirection(direction, options = {}){
    const normalized = normalizeTranslationDirection(direction);
    const prev = translationDirection;
    translationDirection = normalized;
    if (translationDirectionToggle) {
      translationDirectionToggle.checked = normalized === "en2zh";
    }
    if (translationDirectionLabel) {
      translationDirectionLabel.textContent = normalized === "en2zh" ? "英文->中文" : "中文->英文";
    }
    const silent = !!(options && options.silent);
    if (!silent && prev !== normalized) {
      traceSubtitle("translation_direction_ui_set", { prev, next: normalized });
    }
    return normalized;
  }

  function sendTranslationDirection(direction){
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const next = normalizeTranslationDirection(direction);
    try {
      ws.send(JSON.stringify({ type: "set_translation_direction", translation_direction: next }));
      traceSubtitle("translation_direction_sent", { next });
    } catch (err) {
      traceSubtitle("translation_direction_send_failed", { next, error: String(err || "") });
    }
  }

  function lockUI(active){
    btnStart.disabled = active;
    btnStop.disabled = !active;
  }

  function setRawAsrText(text, options = {}){
    if (options && options.resetCurrent) {
      rawAsrLastSnapshot = "";
      if (rawTextEl) {
        rawTextEl.textContent = rawAsrText;
        rawTextEl.scrollTop = rawTextEl.scrollHeight;
      }
      return;
    }

    const next = String(text || "").trim();
    if (!next) return;
    const prev = String(rawAsrLastSnapshot || "");

    if (!rawAsrText) {
      rawAsrText = next;
    } else if (!prev) {
      rawAsrText = `${rawAsrText}\n${next}`;
    } else if (next.startsWith(prev)) {
      if (rawAsrText.endsWith(prev)) {
        rawAsrText = rawAsrText.slice(0, rawAsrText.length - prev.length) + next;
      } else {
        rawAsrText = `${rawAsrText}\n${next}`;
      }
    } else if (prev.startsWith(next)) {
      // Ignore temporary shrink rewrite from unstable partials.
      if (rawTextEl) {
        rawTextEl.textContent = rawAsrText;
        rawTextEl.scrollTop = rawTextEl.scrollHeight;
      }
      return;
    } else {
      rawAsrText = `${rawAsrText}\n${next}`;
    }

    rawAsrLastSnapshot = next;
    if (rawTextEl) {
      rawTextEl.textContent = rawAsrText;
      rawTextEl.scrollTop = rawTextEl.scrollHeight;
    }
  }

  function setCurrentSegmentText(nextText){
    currentSegmentText = String(nextText || "");
  }

  function combineSegments(segments){
    const parts = [];
    for (const seg of segments) {
      const text = String(seg || "").trim();
      if (!text) continue;
      parts.push(text);
    }
    return parts.join(" ").trim();
  }

  function pruneSubtitleWindow(nowMs){
    const cutoff = nowMs - SUBTITLE_KEEP_MS;
    let drop = 0;
    while (drop < subtitleSentencePairs.length - 1 && subtitleSentencePairs[drop].ts < cutoff) {
      drop += 1;
    }
    if (drop > 0) {
      const droppedIds = subtitleSentencePairs.slice(0, drop).map((item) => String(item.sid || "")).slice(0, 8);
      subtitleSentencePairs = subtitleSentencePairs.slice(drop);
      traceSubtitle("prune_subtitle_window", {
        drop,
        remaining: subtitleSentencePairs.length,
        droppedIds,
      });
    }
  }

  function upsertCommittedSentence(sentenceId, text, tsMs, options = {}){
    const zhText = String(text || "").trim();
    if (!zhText) return false;
    const allowOverwrite = options.allowOverwrite !== false;
    const sliceCommit = !!options.sliceCommit;
    const sid = String(sentenceId || "").trim();
    const now = Number(tsMs || Date.now());
    if (sid) {
      const foundIndex = subtitleSentencePairs.findIndex((item) => item.sid === sid);
      const found = foundIndex >= 0 ? subtitleSentencePairs[foundIndex] : null;
      if (found) {
        if (!allowOverwrite) {
          traceSubtitle("sentence_skip_overwrite", { sid, nextLen: zhText.length });
          return false;
        }
        if (found.zh === zhText) {
          if (sliceCommit && !found.sliceCommit) found.sliceCommit = true;
          traceSubtitle("sentence_noop", { sid, len: zhText.length, sliceCommit: !!sliceCommit });
          return false;
        }
        const prevLen = String(found.zh || "").length;
        found.zh = zhText;
        found.ts = Math.max(Number(found.ts || now), now);
        if (sliceCommit && !found.sliceCommit) found.sliceCommit = true;
        traceSubtitle("sentence_updated_local", { sid, prevLen, nextLen: zhText.length, sliceCommit: !!sliceCommit });
        return true;
      }
    }
    subtitleSentencePairs.push({
      sid: sid || `local-${now}-${subtitleSentencePairs.length + 1}`,
      zh: zhText,
      en: "",
      ts: now,
      sliceCommit,
    });
    traceSubtitle("sentence_insert_local", {
      sid: sid || `local-${now}-${subtitleSentencePairs.length}`,
      len: zhText.length,
      count: subtitleSentencePairs.length,
      sliceCommit: !!sliceCommit,
    });
    return true;
  }

  function updateCommittedSentenceTranslation(sentenceId, text){
    const sid = String(sentenceId || "").trim();
    const enText = String(text || "").trim();
    if (!sid) return;
    if (!enText) {
      traceSubtitle("translation_skip_empty", { sid });
      return;
    }
    const found = subtitleSentencePairs.find((item) => item.sid === sid);
    if (!found) {
      traceSubtitle("translation_skip_missing_sentence", { sid, len: enText.length });
      return;
    }
    const cur = String(found.en || "").trim();
    if (cur === enText) {
      traceSubtitle("translation_noop", { sid, len: enText.length });
      return;
    }
    found.en = enText;
    if (cur) {
      traceSubtitle("translation_updated_local", {
        sid,
        prevLen: cur.length,
        len: enText.length,
      });
      return;
    }
    traceSubtitle("translation_set_local", { sid, len: enText.length });
  }

  function renderTranscript(){
    const rows = buildSubtitleRows();
    zhLineNodes = patchSubtitleContainer(
      textEl,
      rows,
      (row) => row.zh,
      zhLineNodes
    );
  }

  function renderTranslation(){
    const rows = buildSubtitleRows();
    enLineNodes = patchSubtitleContainer(
      translationEl,
      rows,
      (row) => row.en || " ",
      enLineNodes
    );
  }

  function buildSubtitleRows(){
    const committedRows = [];
    for (const item of subtitleSentencePairs) {
      const sid = String(item.sid || `row-${committedRows.length + 1}`);
      const zh = String(item.zh || "").trim();
      const en = String(item.en || "").trim();
      if (!zh && !en) continue;
      committedRows.push({ sid, zh, en });
    }
    const tail = String(currentTextTail || "").trim();

    if (USE_COMMITTED_SENTENCE_EVENTS) {
      const reserveTailSlot = running && committedRows.length > 0;
      const committedCap = reserveTailSlot ? Math.max(1, MAX_VISIBLE_ROWS - 1) : MAX_VISIBLE_ROWS;
      const head = committedRows.length > committedCap
        ? committedRows.slice(committedRows.length - committedCap)
        : committedRows.slice();
      if (tail || reserveTailSlot) {
        head.push({ sid: "__tail__", zh: tail, en: "" });
      }
      return clipVisibleRows(head);
    }

    const rows = committedRows.slice();
    if (tail) {
      rows.push({ sid: "__tail__", zh: tail, en: "" });
    }
    return clipVisibleRows(rows);
  }

  function clearSubtitleDom(){
    if (textEl) textEl.replaceChildren();
    if (translationEl) translationEl.replaceChildren();
  }

  function clipVisibleRows(rows){
    if (!rows || rows.length <= MAX_VISIBLE_ROWS) return rows || [];
    return rows.slice(rows.length - MAX_VISIBLE_ROWS);
  }

  function subtitleChars(rows, pickText){
    let total = 0;
    for (const row of rows) {
      const text = String((pickText(row) || "")).trim();
      if (!text) continue;
      total += text.length;
    }
    return total;
  }

  function patchSubtitleContainer(container, rows, pickText, prevNodes){
    const keep = new Set(rows.map((row) => String(row.sid || "")));
    let removed = 0;
    const removedIds = [];
    for (const [sid, node] of prevNodes.entries()) {
      if (!keep.has(sid)) {
        node.remove();
        removed += 1;
        if (removedIds.length < 8) removedIds.push(String(sid || ""));
      }
    }

    const nextNodes = new Map();
    const orderedNodes = [];
    let created = 0;
    let changedText = 0;
    for (const row of rows) {
      const sid = String(row.sid || "");
      const text = String((pickText(row) || "")).trim() || " ";
      let node = prevNodes.get(sid);
      if (!node) {
        node = document.createElement("div");
        node.className = "subtitle-line line-enter";
        node.addEventListener("animationend", () => {
          node.classList.remove("line-enter");
        }, { once: true });
        created += 1;
      }
      if (node.textContent !== text) {
        node.textContent = text;
        changedText += 1;
      }
      node.dataset.sid = sid;
      nextNodes.set(sid, node);
      orderedNodes.push(node);
    }

    for (let i = 0; i < orderedNodes.length; i++) {
      const node = orderedNodes[i];
      const refNode = container.children[i] || null;
      if (refNode !== node) {
        container.insertBefore(node, refNode);
      }
    }
    // Always pin to latest lines when content exceeds lane height.
    container.scrollTop = container.scrollHeight;
    const lane = container === textEl ? "zh" : (container === translationEl ? "en" : "unknown");
    if (removed > 0 || created > 0 || changedText > 0) {
      traceSubtitle("patch_container", {
        lane,
        rows: rows.length,
        prevRows: prevNodes.size,
        removed,
        created,
        changedText,
        removedIds,
        keepTail: !!currentTextTail,
      });
    }
    return nextNodes;
  }

  function resetFinalWait(){
    if (finalTimer) {
      clearTimeout(finalTimer);
      finalTimer = null;
    }
    pendingFinalResolve = null;
    pendingFinalReject = null;
  }

  function rejectPendingFinal(err){
    if (!pendingFinalReject) return;
    const reject = pendingFinalReject;
    resetFinalWait();
    reject(err);
  }

  async function sendFinishAndAwaitFinal(mode, timeoutMs, reason = ""){
    if (!ws || ws.readyState !== WebSocket.OPEN) return null;
    if (pendingFinalResolve) {
      throw new Error("finish already pending");
    }
    return new Promise((resolve, reject) => {
      pendingFinalResolve = resolve;
      pendingFinalReject = reject;
      traceSubtitle("finish_sent", {
        mode: String(mode || ""),
        timeoutMs: Number(timeoutMs || 0),
        queuedBytes,
        sendQueueLen: sendQueue.length,
      });
      finalTimer = setTimeout(() => {
        rejectPendingFinal(new Error("final timeout"));
      }, timeoutMs);
      try {
        const payload = {type: "finish", mode};
        if (reason) payload.reason = String(reason);
        ws.send(JSON.stringify(payload));
      } catch (err) {
        rejectPendingFinal(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  function resetSessionFlags(keepSubtitles = true){
    traceSubtitle("reset_session_flags", {
      keepSubtitles: !!keepSubtitles,
      committedCount: subtitleSentencePairs.length,
      tailLen: String(currentTextTail || "").length,
    });
    clearTailStabilizeTimer();
    running = false;
    awaitingFinal = false;
    resetFinalWait();
    sendQueue = [];
    queuedBytes = 0;
    pending = new Float32Array(0);
    if (!keepSubtitles) {
      subtitleSentencePairs = [];
      clearSubtitleDom();
      zhLineNodes = new Map();
      enLineNodes = new Map();
      clearCommittedTentativeTailNow();
      currentTranslationTail = "";
      setCurrentSegmentText("");
    }
    sessionStartedAt = 0;
    lastCaptureAt = 0;
    lastChunkSentAt = 0;
    lastPartialAt = 0;
    if (watchdogTimer) {
      clearInterval(watchdogTimer);
      watchdogTimer = null;
    }
    lockUI(false);
  }

  function startWatchdog(){
    if (watchdogTimer) clearInterval(watchdogTimer);
    watchdogTimer = setInterval(() => {
      if (!running) return;
      const now = Date.now();
      if (sessionStartedAt && now - sessionStartedAt > 8000 && lastCaptureAt === 0) {
        setStatus("No audio input / 未检测到音频输入", "warn");
        return;
      }
      if (
        sessionStartedAt &&
        now - sessionStartedAt > 8000 &&
        lastCaptureAt > 0 &&
        lastChunkSentAt === 0
      ) {
        setStatus("Upstream blocked / 上行拥塞", "warn");
        return;
      }
      if (
        ws &&
        ws.readyState === WebSocket.OPEN &&
        ws.bufferedAmount > MAX_WS_BUFFERED_BYTES &&
        lastPartialAt > 0 &&
        now - lastPartialAt > 10000
      ) {
        setStatus("Server busy / 识别延迟", "warn");
      }
    }, 1000);
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  function float32ToPcm16(samples){
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
    }
    return out.buffer;
  }

  function describeStartError(err){
    const name = (err && err.name) ? err.name : "Error";
    const msg = (err && err.message) ? err.message : String(err || "unknown");
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "麦克风权限被拒绝，请在浏览器地址栏允许麦克风访问。";
    }
    if (name === "NotFoundError") {
      return "未检测到可用麦克风设备，请检查系统输入设备。";
    }
    if (name === "NotReadableError") {
      return "麦克风不可读，可能被其他应用占用。";
    }
    if (name === "OverconstrainedError") {
      return "麦克风参数不兼容，已建议改用默认音频配置。";
    }
    if (name === "AbortError") {
      return "麦克风初始化被中断，请重试。";
    }
    return `${name}: ${msg}`;
  }

  function selectedLanguage(){
    if (!languageSelect) return "";
    return String(languageSelect.value || "").trim();
  }

  function buildAudioConstraints(){
    return {
      channelCount: { ideal: 1 },
      echoCancellation: !!(toggleEchoCancellation && toggleEchoCancellation.checked),
      noiseSuppression: !!(toggleNoiseSuppression && toggleNoiseSuppression.checked),
      autoGainControl: !!(toggleAutoGainControl && toggleAutoGainControl.checked)
    };
  }

  async function openMicrophone(){
    if (!window.isSecureContext && !isLocalhost()) {
      throw new Error("远程访问麦克风需要 HTTPS。请确认通过 Caddy 的 https 地址访问。");
    }

    const modernGetUserMedia =
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function"
        ? navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
        : null;
    const legacyGetUserMedia =
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia;

    if (!modernGetUserMedia && !legacyGetUserMedia) {
      throw new Error(
        "当前页面环境不支持麦克风采集，请使用最新版 Chrome/Edge/Safari，并避免在受限内嵌 WebView 中打开。"
      );
    }

    const getUserMediaCompat = (constraints) => {
      if (modernGetUserMedia) {
        return modernGetUserMedia(constraints);
      }
      return new Promise((resolve, reject) => {
        legacyGetUserMedia.call(navigator, constraints, resolve, reject);
      });
    };

    const preferredConstraints = {
      audio: buildAudioConstraints(),
      video: false
    };
    try {
      return await getUserMediaCompat(preferredConstraints);
    } catch (err) {
      // Fallback for devices/browsers that reject advanced constraints.
      if (err && err.name === "OverconstrainedError") {
        return await getUserMediaCompat({ audio: true, video: false });
      }
      throw err;
    }
  }

  async function stopPipeline(resetPending = true){
    try {
      if (processor) {
        processor.disconnect();
        processor.onaudioprocess = null;
      }
      if (workletNode) {
        workletNode.port.onmessage = null;
        workletNode.disconnect();
      }
      if (sinkGain) sinkGain.disconnect();
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    } catch (err) {
      console.error(err);
    }
    processor = null;
    workletNode = null;
    sinkGain = null;
    source = null;
    audioCtx = null;
    mediaStream = null;
    if (workletModuleUrl) {
      try { URL.revokeObjectURL(workletModuleUrl); } catch (err) {}
      workletModuleUrl = null;
    }
    if (resetPending) {
      pending = new Float32Array(0);
      sendQueue = [];
      queuedBytes = 0;
    }
  }

  function enqueueSendBuffer(frame){
    if (!frame || !(frame instanceof ArrayBuffer)) return;
    if (frame.byteLength <= 0) return;
    sendQueue.push(frame);
    queuedBytes += frame.byteLength;
    while (queuedBytes > MAX_SEND_QUEUE_BYTES && sendQueue.length > 1) {
      const dropped = sendQueue.shift();
      queuedBytes -= dropped.byteLength;
    }
  }

  function flushPendingToQueue(){
    while (pending.length >= CHUNK_SAMPLES) {
      const chunk = pending.slice(0, CHUNK_SAMPLES);
      pending = pending.slice(CHUNK_SAMPLES);
      enqueueSendBuffer(float32ToPcm16(chunk));
    }
  }

  function onCapturedSamples(samples, srcSr){
    if (!running) return;
    if (!samples || samples.length === 0) return;
    lastCaptureAt = Date.now();
    const rs = resampleLinear(samples, srcSr, TARGET_SR);
    // Frontend no longer owns slice decisions; keep a single stream state per websocket.
    pending = concatFloat32(pending, rs);
    flushPendingToQueue();
    pump();
  }

  async function drainSendQueue(timeoutMs){
    const deadline = Date.now() + timeoutMs;
    while (ws && ws.readyState === WebSocket.OPEN && Date.now() < deadline) {
      pump();
      if (sendQueue.length === 0 && ws.bufferedAmount < 16384) {
        return true;
      }
      await sleep(20);
    }
    return sendQueue.length === 0;
  }

  async function buildCaptureGraph(){
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
    source = audioCtx.createMediaStreamSource(mediaStream);

    if (audioCtx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
      const moduleCode = `
        class MicCaptureProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];
            if (input && input[0] && input[0].length > 0) {
              this.port.postMessage(input[0].slice(0));
            }
            return true;
          }
        }
        registerProcessor("mic-capture-processor", MicCaptureProcessor);
      `;
      workletModuleUrl = URL.createObjectURL(
        new Blob([moduleCode], { type: "application/javascript" })
      );
      await audioCtx.audioWorklet.addModule(workletModuleUrl);
      workletNode = new AudioWorkletNode(audioCtx, "mic-capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit"
      });
      workletNode.port.onmessage = (evt) => {
        const frame = evt.data instanceof Float32Array ? evt.data : new Float32Array(evt.data || []);
        onCapturedSamples(frame, audioCtx.sampleRate);
      };
      sinkGain = audioCtx.createGain();
      sinkGain.gain.value = 0.0;
      source.connect(workletNode);
      workletNode.connect(sinkGain);
      sinkGain.connect(audioCtx.destination);
      return;
    }

    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (evt) => {
      const in0 = evt.inputBuffer.getChannelData(0);
      onCapturedSamples(in0, audioCtx.sampleRate);
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);
    setStatus("Listening (fallback) / 识别中(兼容模式)", "warn");
  }

  function handleServerMessage(evt){
    let msg = {};
    try {
      msg = JSON.parse(evt.data);
    } catch (err) {
      console.error("invalid json", err);
      return;
    }
    if (msg.type === "ready") {
      const localDirectionBeforeStart = selectedTranslationDirection();
      if (msg.translation_direction) {
        const serverDirection = normalizeTranslationDirection(msg.translation_direction);
        if (serverDirection !== localDirectionBeforeStart) {
          traceSubtitle("ws_ready_direction_ignored", {
            serverDirection,
            localDirection: localDirectionBeforeStart,
          });
        }
      }
      traceSubtitle("ws_ready", {
        translationDirection: localDirectionBeforeStart,
      });
      setStatus("Connected / 已连接", "ok");
      return;
    }
    if (msg.type === "started") {
      if (msg.translation_direction) {
        applyTranslationDirection(msg.translation_direction);
      }
      traceSubtitle("ws_started", {
        language: String(msg.language || ""),
        translationDirection: String(msg.translation_direction || selectedTranslationDirection()),
      });
      if (msg.language) {
        if (langEl) langEl.textContent = msg.language;
      } else {
        if (langEl) langEl.textContent = "-";
      }
      setCurrentSegmentText("");
      setRawAsrText("", { resetCurrent: true });
      clearCommittedTentativeTailNow();
      currentTranslationTail = "";
      renderTranscript();
      renderTranslation();
      return;
    }
    if (msg.type === "translation_direction") {
      const direction = applyTranslationDirection(msg.translation_direction);
      traceSubtitle("ws_translation_direction", { direction });
      return;
    }
    if (msg.type === "sentence_committed") {
      if (!USE_COMMITTED_SENTENCE_EVENTS) return;
      lastPartialAt = Date.now();
      traceSubtitle("ws_sentence_committed", {
        sid: String(msg.sentence_id || ""),
        len: String(msg.text || "").trim().length,
        sliceCommit: !!msg.slice_commit,
        beforeCount: subtitleSentencePairs.length,
      });
      const changed = upsertCommittedSentence(
        msg.sentence_id,
        msg.text || "",
        msg.ts_ms || Date.now(),
        { sliceCommit: !!msg.slice_commit }
      );
      if (!changed) return;
      pruneSubtitleWindow(Date.now());
      renderTranscript();
      renderTranslation();
      if (running) {
        setStatus("Listening / 识别中", "ok");
      }
      return;
    }
    if (msg.type === "sentence_updated") {
      if (!USE_COMMITTED_SENTENCE_EVENTS) return;
      lastPartialAt = Date.now();
      const sid = String(msg.sentence_id || "").trim();
      const nextText = String(msg.text || "").trim();
      const current = sid ? subtitleSentencePairs.find((item) => String(item.sid || "") === sid) : null;
      const prevText = current ? String(current.zh || "").trim() : "";
      const allowOverwrite = true;
      traceSubtitle("ws_sentence_updated", {
        sid,
        len: nextText.length,
        prevLen: prevText.length,
        allowOverwrite,
      });
      const changed = upsertCommittedSentence(
        sid,
        nextText,
        msg.ts_ms || Date.now(),
        { allowOverwrite, sliceCommit: !!msg.slice_commit }
      );
      if (!changed) return;
      pruneSubtitleWindow(Date.now());
      renderTranscript();
      renderTranslation();
      return;
    }
    if (msg.type === "sentence_translation") {
      if (!USE_COMMITTED_SENTENCE_EVENTS) return;
      lastPartialAt = Date.now();
      traceSubtitle("ws_sentence_translation", {
        sid: String(msg.sentence_id || ""),
        len: String(msg.translation || "").trim().length,
      });
      updateCommittedSentenceTranslation(msg.sentence_id, msg.translation || "");
      renderTranslation();
      return;
    }
    if (msg.type === "sentence_reset") {
      if (!USE_COMMITTED_SENTENCE_EVENTS) return;
      lastPartialAt = Date.now();
      traceSubtitle("ws_sentence_reset", {
        reason: String(msg.reason || ""),
        beforeCount: subtitleSentencePairs.length,
      });
      subtitleSentencePairs = [];
      setCurrentSegmentText("");
      setRawAsrText("", { resetCurrent: true });
      clearSubtitleDom();
      zhLineNodes = new Map();
      enLineNodes = new Map();
      clearCommittedTentativeTailNow();
      currentTranslationTail = "";
      renderTranscript();
      renderTranslation();
      return;
    }
    if (msg.type === "partial") {
      lastPartialAt = Date.now();
      lastPartialSeq = Number(msg.seq || lastPartialSeq || 0);
      if (langEl) langEl.textContent = msg.language || "-";
      const nextText = String(msg.text || "");
      setRawAsrText(nextText);
      if (USE_COMMITTED_SENTENCE_EVENTS) {
        setCurrentSegmentText(nextText);
        const tentativeTail = String(msg.tentative_text || "").trim();
        if (lastPartialSeq <= 5 || (lastPartialSeq % 20 === 0 && lastPartialTraceSeq !== lastPartialSeq)) {
          traceSubtitle("ws_partial", {
            seq: lastPartialSeq,
            textLen: nextText.trim().length,
            tailLen: tentativeTail.length,
            committedCount: subtitleSentencePairs.length,
          });
          lastPartialTraceSeq = lastPartialSeq;
        }
        updateCommittedTentativeTail(tentativeTail);
        currentTranslationTail = "";
        renderTranscript();
        renderTranslation();
        if (running) {
          setStatus("Listening / 识别中", "ok");
          pump();
        }
        return;
      }
      return;
    }
    if (msg.type === "final") {
      lastPartialAt = Date.now();
      if (langEl) langEl.textContent = msg.language || "-";
      const finalText = msg.text || "";
      setRawAsrText(finalText);
      const mode = "stop";
      traceSubtitle("ws_final", {
        mode,
        finalLen: String(finalText || "").trim().length,
        tentativeLen: String(msg.tentative_text || "").trim().length,
        committedCount: subtitleSentencePairs.length,
      });
      const resolve = pendingFinalResolve;
      resetFinalWait();

      clearTailStabilizeTimer();
      setCurrentSegmentText("");
      currentTextTail = String(msg.tentative_text || "").trim();
      currentTranslationTail = "";
      renderTranscript();
      renderTranslation();
      awaitingFinal = false;
      setStatus("Stopped / 已停止", "");
      if (resolve) resolve(msg);
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (err) {}
      }
      return;
    }
    if (msg.type === "processing") {
      setStatus("Processing / 服务器处理中", "warn");
      return;
    }
    if (msg.type === "error") {
      rejectPendingFinal(new Error(msg.message || "websocket server error"));
      resetSessionFlags();
      stopPipeline();
      setStatus("Error / 错误: " + (msg.message || "unknown"), "err");
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (err) {}
      }
      return;
    }
  }

  async function openSocket(timeoutMs = 8000){
    return new Promise((resolve, reject) => {
      let timer = null;
      let done = false;
      const finish = (fn, value) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        fn(value);
      };
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${scheme}://${location.host}/ws`);
      ws = sock;
      sock.binaryType = "arraybuffer";

      sock.onmessage = (evt) => {
        if (sock !== ws) return;
        handleServerMessage(evt);
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "ready") {
            finish(resolve);
            return;
          }
          if (msg.type === "error" && !running && !awaitingFinal) {
            finish(reject, new Error(msg.message || "websocket server error"));
          }
        } catch (err) {}
      };
      sock.onerror = () => {
        if (sock !== ws) return;
        finish(reject, new Error("websocket failed"));
      };
      sock.onclose = (evt) => {
        if (sock !== ws) return;
        if (!done) {
          finish(reject, new Error(`websocket closed (${evt.code})`));
        }
        rejectPendingFinal(new Error(`websocket closed (${evt.code})`));
        if (running) {
          resetSessionFlags();
          stopPipeline();
          setStatus("Disconnected / 连接断开", "warn");
        } else if (awaitingFinal) {
          resetSessionFlags();
          stopPipeline();
          setStatus("Disconnected before final / 收尾前连接断开", "err");
        }
      };
      timer = setTimeout(() => {
        finish(reject, new Error("websocket ready timeout"));
      }, timeoutMs);
    });
  }

  function pump(){
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    while (sendQueue.length > 0) {
      if (ws.bufferedAmount > MAX_WS_BUFFERED_BYTES) break;
      const frame = sendQueue.shift();
      queuedBytes -= frame.byteLength;
      try {
        ws.send(frame);
        lastChunkSentAt = Date.now();
      } catch (err) {
        console.error(err);
        resetSessionFlags();
        stopPipeline();
        setStatus("Send failed / 音频发送失败", "err");
        return;
      }
    }
  }

  if (translationDirectionToggle) {
    translationDirectionToggle.addEventListener("change", () => {
      const next = selectedTranslationDirection();
      applyTranslationDirection(next);
      sendTranslationDirection(next);
    });
  }

  btnStart.onclick = async () => {
    if (running) return;
    subtitleSentencePairs = [];
    setCurrentSegmentText("");
    clearSubtitleDom();
    zhLineNodes = new Map();
    enLineNodes = new Map();
    clearCommittedTentativeTailNow();
    currentTranslationTail = "";
    renderTranscript();
    renderTranslation();
    if (langEl) langEl.textContent = "-";
    pending = new Float32Array(0);
    sendQueue = [];
    queuedBytes = 0;
    resetFinalWait();
    sessionStartedAt = 0;
    lastCaptureAt = 0;
    lastChunkSentAt = 0;
    lastPartialAt = 0;
    lockUI(true);
    setStatus("Starting / 启动中", "warn");

    try {
      mediaStream = await openMicrophone();
      await openSocket();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "start",
            language: selectedLanguage(),
            translation_direction: selectedTranslationDirection(),
          })
        );
      }

      await buildCaptureGraph();

      running = true;
      sessionStartedAt = Date.now();
      startWatchdog();
      if (!processor) {
        setStatus("Listening / 识别中", "ok");
      }
    } catch (err) {
      console.error(err);
      await stopPipeline();
      if (ws) {
        try { ws.close(); } catch (closeErr) {}
      }
      ws = null;
      running = false;
      lockUI(false);
      setStatus("Start failed / 启动失败: " + describeStartError(err), "err");
    }
  };

  btnStop.onclick = async () => {
    if (!running) return;
    // Stop microphone first, then flush queued PCM before sending finish.
    running = false;
    awaitingFinal = true;
    lockUI(false);
    setStatus("Finishing / 收尾中", "warn");
    await stopPipeline(false);

    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        flushPendingToQueue();
        if (pending.length > 0) {
          enqueueSendBuffer(float32ToPcm16(pending));
          pending = new Float32Array(0);
        }
        const drained = await drainSendQueue(WEBSOCKET_DRAIN_TIMEOUT_MS);
        if (!drained) {
          setStatus("Finishing (network backlog) / 收尾中(网络积压)", "warn");
        }
        await sendFinishAndAwaitFinal("stop", 45000);
      } else {
        awaitingFinal = false;
        setStatus("Stopped / 已停止", "");
      }
    } catch (err) {
      console.error(err);
      rejectPendingFinal(err instanceof Error ? err : new Error(String(err)));
      awaitingFinal = false;
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (closeErr) {}
      }
      setStatus("Stop failed / 停止失败", "err");
    }
  };

  if (typeof window !== "undefined") {
    const _resetDebugSubtitleState = () => {
      subtitleSentencePairs = [];
      setCurrentSegmentText("");
      setRawAsrText("", { resetCurrent: true });
      clearSubtitleDom();
      zhLineNodes = new Map();
      enLineNodes = new Map();
      clearCommittedTentativeTailNow();
      currentTranslationTail = "";
      resetFinalWait();
      renderTranscript();
      renderTranslation();
      if (langEl) langEl.textContent = "-";
    };

    const _base64ToUint8 = (value) => {
      let src = String(value || "").trim();
      if (!src) return new Uint8Array(0);
      const marker = "base64,";
      const idx = src.indexOf(marker);
      if (idx >= 0) src = src.slice(idx + marker.length);
      src = src.replace(/\\s+/g, "");
      const bin = atob(src);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    };

    const _waitUntil = async (predicate, timeoutMs) => {
      const timeout = Math.max(1000, Number(timeoutMs) || 1000);
      const begin = Date.now();
      while (Date.now() - begin < timeout) {
        if (predicate()) return true;
        await sleep(20);
      }
      return false;
    };

    window.__subtitleDebug = {
      feed(msg){
        handleServerMessage({ data: JSON.stringify(msg || {}) });
        return this.getState();
      },
      setState(next){
        const state = (next && typeof next === "object") ? next : {};
        const hasOwn = (name) => Object.prototype.hasOwnProperty.call(state, name);
        if (hasOwn("running")) running = !!state.running;
        if (hasOwn("currentTextTail")) currentTextTail = String(state.currentTextTail || "");
        if (hasOwn("currentSegmentText")) setCurrentSegmentText(String(state.currentSegmentText || ""));
        if (hasOwn("currentTranslationTail")) currentTranslationTail = String(state.currentTranslationTail || "");
        if (state.render !== false) {
          renderTranscript();
          renderTranslation();
        }
        return this.getState();
      },
      async wait(ms){
        await sleep(Math.max(0, Number(ms) || 0));
        return this.getState();
      },
      getTrace(limit){
        const rows = subtitleTraceEvents.slice();
        const n = Number(limit || 0);
        if (n > 0 && n < rows.length) {
          return rows.slice(rows.length - n);
        }
        return rows;
      },
      clearTrace(){
        subtitleTraceEvents = [];
        subtitleTraceSeq = 0;
        return true;
      },
      setTraceEnabled(enabled){
        subtitleTraceEnabled = !!enabled;
        try {
          localStorage.setItem("subtitle_trace", subtitleTraceEnabled ? "1" : "0");
        } catch (err) {}
        traceSubtitle("trace_toggle", { enabled: subtitleTraceEnabled }, true);
        return subtitleTraceEnabled;
      },
      async streamPcmFromUrl(url, options){
        const src = String(url || "").trim();
        if (!src) throw new Error("streamPcmFromUrl requires url");
        const resp = await fetch(src, { cache: "no-store" });
        if (!resp.ok) {
          throw new Error(`fetch pcm failed: ${resp.status}`);
        }
        const bytes = new Uint8Array(await resp.arrayBuffer());
        if (bytes.length === 0) {
          throw new Error("fetched empty pcm bytes");
        }
        let bin = "";
        const step = 0x8000;
        for (let i = 0; i < bytes.length; i += step) {
          const chunk = bytes.subarray(i, Math.min(bytes.length, i + step));
          bin += String.fromCharCode.apply(null, chunk);
        }
        const cfg = (options && typeof options === "object") ? { ...options } : {};
        cfg.base64 = btoa(bin);
        return await this.streamPcm16Base64(cfg);
      },
      async streamPcm16Base64(options){
        const cfg = (options && typeof options === "object") ? options : {};
        const pcmB64 = String(cfg.base64 || "").trim();
        if (!pcmB64) {
          throw new Error("streamPcm16Base64 requires {base64}");
        }
        const bytes = _base64ToUint8(pcmB64);
        if (bytes.length === 0) {
          throw new Error("empty pcm payload");
        }

        const language = String(cfg.language || "auto");
        const timeoutMs = Math.max(5000, Number(cfg.timeoutMs || 120000));
        const paceMs = Math.max(0, Number(cfg.paceMs || 0));
        let chunkBytes = Number(cfg.chunkBytes || 0);
        if (!(chunkBytes > 0)) {
          const chunkMs = Math.max(20, Number(cfg.chunkMs || 200));
          chunkBytes = Math.round(16000 * 2 * (chunkMs / 1000.0));
        }
        chunkBytes = Math.max(320, Math.floor(chunkBytes));
        if (chunkBytes % 2 === 1) chunkBytes += 1;

        _resetDebugSubtitleState();
        running = true;
        awaitingFinal = false;
        setStatus("Debug stream / 调试流式", "warn");

        const scheme = location.protocol === "https:" ? "wss" : "ws";
        const sock = new WebSocket(`${scheme}://${location.host}/ws`);
        ws = sock;
        sock.binaryType = "arraybuffer";

        const events = [];
        let ready = false;
        let finished = false;
        let errorMessage = "";

        sock.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            events.push(msg);
            if (msg.type === "ready") ready = true;
            if (msg.type === "error") {
              errorMessage = String(msg.message || "websocket server error");
              finished = true;
            } else if (msg.type === "final") {
              finished = true;
            }
          } catch (err) {}
          handleServerMessage(evt);
        };
        sock.onerror = () => {
          if (!errorMessage) errorMessage = "websocket failed";
          finished = true;
        };
        sock.onclose = (evt) => {
          if (!finished && !errorMessage && evt.code !== 1000) {
            errorMessage = `websocket closed (${evt.code})`;
            finished = true;
          }
        };

        const readyOk = await _waitUntil(() => ready || !!errorMessage, timeoutMs);
        if (!readyOk || errorMessage) {
          if (sock.readyState === WebSocket.OPEN || sock.readyState === WebSocket.CONNECTING) {
            try { sock.close(); } catch (closeErr) {}
          }
          throw new Error(errorMessage || "websocket ready timeout");
        }

        sock.send(
          JSON.stringify({
            type: "start",
            language,
            translation_direction: selectedTranslationDirection(),
          })
        );
        for (let i = 0; i < bytes.length; i += chunkBytes) {
          if (sock.readyState !== WebSocket.OPEN) {
            errorMessage = errorMessage || "websocket closed during stream";
            break;
          }
          const chunk = bytes.subarray(i, Math.min(bytes.length, i + chunkBytes));
          if (chunk.length > 0) {
            sock.send(chunk);
          }
          if (paceMs > 0) await sleep(paceMs);
        }

        if (!errorMessage && sock.readyState === WebSocket.OPEN) {
          sock.send(JSON.stringify({type: "finish", mode: "stop"}));
        }
        if (!errorMessage) {
          const doneOk = await _waitUntil(() => finished, timeoutMs);
          if (!doneOk) errorMessage = "finish timeout";
        }
        if (sock.readyState === WebSocket.OPEN) {
          try { sock.close(); } catch (closeErr) {}
        }

        const committedById = new Map();
        let finalText = "";
        for (const msg of events) {
          if (!msg || typeof msg !== "object") continue;
          const t = String(msg.type || "");
          if (t === "sentence_reset") {
            committedById.clear();
          } else if (t === "sentence_committed") {
            const sid = String(msg.sentence_id || `local-${committedById.size + 1}`);
            committedById.set(sid, String(msg.text || "").trim());
          } else if (t === "sentence_updated") {
            const sid = String(msg.sentence_id || "");
            if (sid && committedById.has(sid)) {
              committedById.set(sid, String(msg.text || "").trim());
            }
          } else if (t === "final") {
            finalText = String(msg.text || "").trim();
          }
        }

        const committedTexts = [];
        for (const text of committedById.values()) {
          const s = String(text || "").trim();
          if (s) committedTexts.push(s);
        }
        const eventCounts = {};
        for (const msg of events) {
          const key = String((msg && msg.type) || "");
          if (!key) continue;
          eventCounts[key] = Number(eventCounts[key] || 0) + 1;
        }

        return {
          ok: !errorMessage,
          error: String(errorMessage || ""),
          eventCounts,
          committedTexts,
          committedJoined: committedTexts.join(" ").trim(),
          finalText,
          state: this.getState(),
        };
      },
      getState(){
        const toRows = (container) => container ? Array.from(container.children).map((node) => String(node.textContent || "")) : [];
        return {
          running,
          subtitleTraceEnabled,
          subtitleTraceCount: subtitleTraceEvents.length,
          currentTextTail: String(currentTextTail || ""),
          currentSegmentText: String(currentSegmentText || ""),
          zhRows: toRows(textEl),
          enRows: toRows(translationEl),
        };
      },
    };
  }

})();
</script>
</body>
</html>
"""


def _decode_pcm16le(raw: bytes) -> np.ndarray:
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("binary frame is required")
    if len(raw) % 2 != 0:
        raise ValueError("pcm16le bytes length must be even")
    if not raw:
        return np.zeros((0,), dtype=np.float32)

    pcm16 = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    wav = pcm16 / 32768.0
    return np.clip(wav, -1.0, 1.0)


def _parse_json_message(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("json message must be an object")
    return payload


def _create_app(args: argparse.Namespace, asr: Any, translator: Optional[LocalTranslator] = None) -> FastAPI:
    app = FastAPI(title="VoxBridge Streaming WebSocket Demo")
    infer_lock = asyncio.Lock()
    runtime = SimpleNamespace(active_connections=0)
    debug_roots = [Path.cwd().resolve(), Path("/tmp").resolve()]

    def _normalize_force_language(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text.lower() in {"auto", "none", "null", "default"}:
            return None
        return text

    def _new_vllm_state(force_language: Optional[str]):
        kwargs = dict(
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_sec,
        )
        if force_language is not None:
            kwargs["language"] = force_language
        return asr.init_streaming_state(**kwargs)

    def _new_transformers_state(force_language: Optional[str]):
        return SimpleNamespace(
            audio_accum=np.zeros((0,), dtype=np.float32),
            language="",
            text="",
            force_language=force_language,
            min_decode_samples=max(1, int(round(float(args.min_audio_sec) * SAMPLE_RATE))),
            decode_interval_samples=max(1, int(round(float(args.decode_interval_sec) * SAMPLE_RATE))),
            last_decoded_samples=0,
        )

    @app.get("/")
    async def index() -> HTMLResponse:
        subtitle_trace = bool(getattr(args, "subtitle_trace", False))
        subtitle_trace_max_events = max(200, int(getattr(args, "subtitle_trace_max_events", 1200)))
        html = INDEX_HTML_TEMPLATE.replace("__CHUNK_MS__", str(int(args.client_chunk_ms)))
        html = html.replace("__SUBTITLE_TRACE__", "true" if subtitle_trace else "false")
        html = html.replace("__SUBTITLE_TRACE_MAX_EVENTS__", str(subtitle_trace_max_events))
        return HTMLResponse(html)

    def _resolve_debug_file(path_text: str) -> Path:
        raw = str(path_text or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="missing path")
        p = Path(raw).expanduser()
        resolved = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
        for root in debug_roots:
            try:
                resolved.relative_to(root)
                if resolved.is_file():
                    return resolved
                raise HTTPException(status_code=404, detail="file not found")
            except ValueError:
                continue
        raise HTTPException(status_code=403, detail="path out of debug roots")

    @app.get("/__debug/file")
    async def debug_file(path: str) -> FileResponse:
        fp = _resolve_debug_file(path)
        return FileResponse(str(fp), media_type="application/octet-stream", filename=fp.name)

    @app.websocket("/ws")
    async def ws_stream(websocket: WebSocket) -> None:
        if runtime.active_connections >= args.max_connections:
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "too many active connections"})
            await websocket.close(code=1013)
            return

        await websocket.accept()
        runtime.active_connections += 1
        peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"

        use_vllm_streaming = getattr(args, "backend", "vllm") == "vllm"
        session_force_language = _normalize_force_language(getattr(args, "force_language", None))
        state = None
        state_generation = 0
        seq = 0
        finished = False
        finish_requested = False
        finish_mode = "stop"
        finish_reason = "stop"
        audio_queue_size = max(1, int(getattr(args, "audio_queue_size", 32)))
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=audio_queue_size)
        consumer_max_batch_samples = max(
            1,
            int(max(0.1, float(getattr(args, "consumer_batch_sec", 1.0))) * SAMPLE_RATE),
        )
        consumer_high_batch_samples = max(consumer_max_batch_samples, int(3.0 * SAMPLE_RATE))
        final_redecode_on_stop = bool(getattr(args, "final_redecode_on_stop", True))
        final_redecode_max_samples = int(max(0.0, float(getattr(args, "final_redecode_max_sec", 180.0))) * SAMPLE_RATE)
        rollover_sec = max(0.0, float(getattr(args, "state_rollover_sec", 30.0)))
        segment_hard_cut_sec = max(1.0, float(getattr(args, "segment_hard_cut_sec", max(rollover_sec, 30.0) or 30.0)))
        segment_overlap_sec = max(0.0, float(getattr(args, "segment_overlap_sec", 0.8)))
        segment_overlap_samples = int(segment_overlap_sec * SAMPLE_RATE)
        queue_target_sec = max(0.2, float(getattr(args, "backpressure_target_queue_sec", 3.0)))
        queue_max_sec = max(queue_target_sec, float(getattr(args, "backpressure_max_queue_sec", 5.0)))
        total_consumed_samples = 0
        last_partial_emit_at = time.monotonic()
        queue_samples = 0
        full_audio_parts: List[np.ndarray] = []
        full_audio_samples = 0
        full_audio_overflow = False
        send_lock = asyncio.Lock()
        state_lock = asyncio.Lock()
        stop_consumer = asyncio.Event()
        consumer_task: Optional[asyncio.Task] = None
        idle_commit_sec = max(3.0, float(getattr(args, "vad_force_cut_sec", 1.8)) + 2.7)
        last_text_snapshot = ""
        last_text_advance_at = time.monotonic()
        last_idle_commit_at = 0.0
        vad_silence_trigger_ms = max(120.0, float(getattr(args, "vad_silence_sec", 0.9)) * 1000.0)
        vad_force_silence_ms = max(vad_silence_trigger_ms + 300.0, float(getattr(args, "vad_force_cut_sec", 1.8)) * 1000.0)
        vad_min_slice_ms = max(500.0, float(getattr(args, "vad_min_slice_sec", 4.0)) * 1000.0)
        vad_min_active_ms = max(200.0, float(getattr(args, "vad_min_active_sec", 1.2)) * 1000.0)
        vad_enter_snr_db = max(1.0, float(getattr(args, "backend_vad_enter_snr_db", 8.0)))
        vad_exit_snr_db = float(getattr(args, "backend_vad_exit_snr_db", 4.0))
        vad_exit_snr_db = max(0.2, min(vad_enter_snr_db - 0.5, vad_exit_snr_db))
        vad_frame_samples = max(80, int(SAMPLE_RATE * 0.02))
        text_stable_cut_ms = max(180.0, float(getattr(args, "backend_cut_stable_sec", 0.45)) * 1000.0)
        segment_policy = SegmentPolicy(
            vad_silence_ms=vad_silence_trigger_ms,
            hard_cut_ms=(segment_hard_cut_sec * 1000.0),
            min_segment_ms=vad_min_slice_ms,
            min_active_ms=vad_min_active_ms,
        )
        backpressure = QueueBackpressureController(
            sample_rate=SAMPLE_RATE,
            target_queue_sec=queue_target_sec,
            max_queue_sec=queue_max_sec,
        )
        segment_runtime = SimpleNamespace(
            id=1,
            started_at=time.monotonic(),
            last_cut_reason="open",
        )
        backpressure_runtime = SimpleNamespace(
            under_pressure=False,
            reason="normal",
        )
        backend_vad = SimpleNamespace(
            noise_db=-55.0,
            in_speech=False,
            speech_confirm_ms=0.0,
            silence_ms=0.0,
            segment_active_ms=0.0,
            segment_elapsed_ms=0.0,
            last_cut_at=time.monotonic(),
        )
        stats = SimpleNamespace(
            raw_frames=0,
            raw_samples=0,
            text_msgs=0,
            start_msgs=0,
            finish_msgs=0,
            partial_msgs=0,
            final_msgs=0,
            queue_dropped=0,
            queue_depth_peak=0,
            last_error="",
        )
        subtitle_state = SimpleNamespace(
            stream_uid=f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}",
            next_sentence_id=1,
            committed_sentences=[],
            sentence_items=[],
            commit_base=0,
            prev_completed_sentences=[],
            tentative_tail="",
            pending_prefix_text="",
            pending_prefix_segment_id=0,
            boundary_anchor_text="",
            boundary_anchor_segment_id=0,
            boundary_overlap_cap_chars=max(4, min(24, int(round(segment_overlap_sec * 14.0)))),
        )
        stream_text_state = SimpleNamespace(last_text="")
        translation_source_default = str(getattr(args, "translation_source_language", "Chinese") or "Chinese")
        translation_target_default = str(getattr(args, "translation_target_language", "English") or "English")

        zh_label = (
            translation_source_default
            if _is_chinese_label(translation_source_default)
            else (
                translation_target_default
                if _is_chinese_label(translation_target_default)
                else "Chinese"
            )
        )
        en_label = (
            translation_source_default
            if _is_english_label(translation_source_default)
            else (
                translation_target_default
                if _is_english_label(translation_target_default)
                else "English"
            )
        )

        def _normalize_translation_direction(raw: Any) -> str:
            text = str(raw or "").strip().lower()
            if text in {"en2zh", "en->zh", "english->chinese", "英文->中文"}:
                return "en2zh"
            return "zh2en"

        def _resolve_direction_languages(direction: str) -> Tuple[str, str]:
            normalized = _normalize_translation_direction(direction)
            if normalized == "en2zh":
                return en_label, zh_label
            return zh_label, en_label

        initial_translation_direction = _normalize_translation_direction("zh2en")
        initial_translation_source, initial_translation_target = _resolve_direction_languages(initial_translation_direction)
        translation_runtime = SimpleNamespace(
            task=None,
            parallelism=max(1, int(getattr(args, "translation_workers", 3))),
            queue=asyncio.Queue(maxsize=256),
            direction=initial_translation_direction,
            source_language=initial_translation_source,
            target_language=initial_translation_target,
        )
        subtitle_trace_log = bool(getattr(args, "subtitle_trace_log", False))
        subtitle_trace_log_partial_every = max(1, int(getattr(args, "subtitle_trace_log_partial_every", 20)))
        trace_seq = 0
        trace_t0 = time.monotonic()
        logger.info(
            "ws open peer=%s active=%d backend=%s force_language=%s translation_direction=%s",
            peer,
            runtime.active_connections,
            "vllm" if use_vllm_streaming else "transformers",
            session_force_language or "",
            translation_runtime.direction,
        )

        def _trace_event(event: str, **payload: Any) -> None:
            nonlocal trace_seq
            if not subtitle_trace_log:
                return
            trace_seq += 1
            row: Dict[str, Any] = {
                "topic": "subtitle_state",
                "trace_seq": int(trace_seq),
                "ts_ms": int(time.time() * 1000),
                "elapsed_ms": int((time.monotonic() - trace_t0) * 1000),
                "peer": peer,
                "event": str(event or ""),
                "state_generation": int(state_generation),
                "finish_mode": str(finish_mode or ""),
                "finish_reason": str(finish_reason or ""),
                "finish_requested": bool(finish_requested),
                "finished": bool(finished),
            }
            if payload:
                row.update(payload)
            try:
                logger.info("subtitle_trace %s", json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logger.info("subtitle_trace %s", row)

        def _hash8(text: str) -> str:
            src = str(text or "").encode("utf-8", errors="ignore")
            if not src:
                return "00000000"
            return hashlib.md5(src).hexdigest()[:8]

        def _trace_text_pool(
            event: str,
            *,
            phase: str,
            text: str,
            reason: str,
            seq_hint: int = 0,
            sentence_id: str = "",
            delta_chars: int = 0,
            **payload: Any,
        ) -> None:
            if not subtitle_trace_log:
                return
            snapshot = str(text or "").strip()
            row: Dict[str, Any] = {
                "topic": "text_pool",
                "event": str(event or ""),
                "phase": str(phase or ""),
                "ws_id": peer,
                "segment_id": int(getattr(segment_runtime, "id", 0) or 0),
                "seq": int(seq_hint or 0),
                "text_chars": len(snapshot),
                "text_hash8": _hash8(snapshot),
                "delta_chars": int(delta_chars or 0),
                "reason": str(reason or ""),
                "sentence_id": str(sentence_id or ""),
                "state_generation": int(state_generation),
            }
            if payload:
                row.update(payload)
            try:
                logger.info("text_pool %s", json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logger.info("text_pool %s", row)

        _trace_event(
            "ws_open",
            active_connections=int(runtime.active_connections),
            backend="vllm" if use_vllm_streaming else "transformers",
            force_language=session_force_language or "",
            audio_queue_size=int(audio_queue_size),
            translation_direction=str(translation_runtime.direction or ""),
            translation_source_language=str(translation_runtime.source_language or ""),
            translation_target_language=str(translation_runtime.target_language or ""),
        )
        _trace_text_pool(
            "segment_open",
            phase="generating",
            text="",
            reason="ws_open",
            seq_hint=int(seq),
        )

        async def _clear_translation_queue() -> int:
            dropped = 0
            while True:
                try:
                    translation_runtime.queue.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    break
            return dropped

        async def _set_translation_direction(
            direction_raw: Any,
            *,
            clear_pending: bool,
            emit: bool,
        ) -> str:
            requested = _normalize_translation_direction(direction_raw)
            source_language, target_language = _resolve_direction_languages(requested)
            changed = (
                requested != str(getattr(translation_runtime, "direction", "") or "")
                or source_language != str(getattr(translation_runtime, "source_language", "") or "")
                or target_language != str(getattr(translation_runtime, "target_language", "") or "")
            )
            translation_runtime.direction = requested
            translation_runtime.source_language = source_language
            translation_runtime.target_language = target_language

            dropped = 0
            if clear_pending:
                if translation_runtime.task is not None and not translation_runtime.task.done():
                    translation_runtime.task.cancel()
                if translation_runtime.task is not None:
                    with suppress(asyncio.CancelledError, Exception):
                        await translation_runtime.task
                translation_runtime.task = None
                dropped = await _clear_translation_queue()

            _trace_event(
                "translation_direction_set",
                direction=requested,
                source_language=source_language,
                target_language=target_language,
                changed=bool(changed),
                queue_dropped=int(dropped),
            )

            if emit:
                await _send_json(
                    {
                        "type": "translation_direction",
                        "translation_direction": requested,
                        "translation_source_language": source_language,
                        "translation_target_language": target_language,
                    }
                )
            return requested

        def _new_sentence_id() -> str:
            sid = f"{subtitle_state.stream_uid}-{subtitle_state.next_sentence_id}"
            subtitle_state.next_sentence_id += 1
            return sid

        def _committed_translation_text() -> str:
            return _join_segments([str(item.get("en", "") or "") for item in subtitle_state.sentence_items])

        async def _translate_sentence_once(
            sentence: str,
            language: str,
            seq_hint: int,
            source_language: str,
            target_language: str,
            direction: str,
        ) -> str:
            if translator is None:
                return ""
            src = str(sentence or "").strip()
            if not src:
                return ""
            effective_source_language = str(source_language or "")
            if not _text_matches_source_language(src, effective_source_language):
                if _has_cjk(src):
                    effective_source_language = zh_label
                elif _has_latin(src):
                    effective_source_language = en_label
                else:
                    effective_source_language = ""
                _trace_event(
                    "translation_source_autofallback",
                    seq=int(seq_hint or 0),
                    language=str(language or ""),
                    source_language=str(source_language or ""),
                    effective_source_language=str(effective_source_language or ""),
                    target_language=str(target_language or ""),
                    direction=str(direction or ""),
                    src_chars=len(src),
                )

            t0 = time.monotonic()
            try:
                try:
                    out = await asyncio.to_thread(
                        translator.translate,
                        src,
                        source_language=effective_source_language,
                        target_language=target_language,
                    )
                except TypeError:
                    out = await asyncio.to_thread(translator.translate, src)
            except Exception as e:
                stats.last_error = f"translate failed: {e}"
                logger.warning("translation failed peer=%s err=%s", peer, e)
                _trace_event(
                    "translation_failed",
                    seq=int(seq_hint or 0),
                    language=str(language or ""),
                    source_language=str(source_language or ""),
                    target_language=str(target_language or ""),
                    direction=str(direction or ""),
                    src_chars=len(src),
                    error=str(e),
                )
                return ""
            latency = time.monotonic() - t0
            _trace_event(
                "translation_done",
                seq=int(seq_hint or 0),
                language=str(language or ""),
                source_language=str(source_language or ""),
                effective_source_language=str(effective_source_language or ""),
                target_language=str(target_language or ""),
                direction=str(direction or ""),
                src_chars=len(src),
                out_chars=len(out or ""),
                latency_ms=int(latency * 1000),
            )
            if latency >= 1.0:
                logger.info(
                    "translation latency peer=%s sec=%.2f seq=%d src_chars=%d out_chars=%d",
                    peer,
                    latency,
                    int(seq_hint or 0),
                    len(src),
                    len(out or ""),
                )
            return str(out or "").strip()

        async def _translation_worker() -> None:
            async def _translate_one(
                sentence_id: str,
                sentence_text: str,
                language: str,
                seq_hint: int,
                state_gen_hint: int,
                source_language: str,
                target_language: str,
                direction: str,
            ) -> None:
                if int(state_gen_hint or 0) != int(state_generation):
                    return
                translated = await _translate_sentence_once(
                    sentence_text,
                    language,
                    int(seq_hint or 0),
                    str(source_language or ""),
                    str(target_language or ""),
                    str(direction or ""),
                )
                if not translated:
                    return

                for item in subtitle_state.sentence_items:
                    if item.get("id") == sentence_id:
                        item["en"] = translated
                        break
                _trace_text_pool(
                    "pool_translation_done",
                    phase="solidified",
                    text=translated,
                    reason="translation",
                    seq_hint=int(seq_hint or 0),
                    sentence_id=str(sentence_id),
                    delta_chars=len(translated),
                )

                await _send_json(
                    {
                        "type": "sentence_translation",
                        "sentence_id": sentence_id,
                        "translation": translated,
                        "seq": int(seq_hint or 0),
                    }
                )

            active_tasks = set()
            try:
                while True:
                    try:
                        (
                            sentence_id,
                            sentence_text,
                            language,
                            seq_hint,
                            state_gen_hint,
                            source_language,
                            target_language,
                            direction,
                        ) = await asyncio.wait_for(
                            translation_runtime.queue.get(),
                            timeout=0.2,
                        )
                    except asyncio.TimeoutError:
                        if translation_runtime.queue.empty() and not active_tasks:
                            break
                        if active_tasks:
                            done, pending = await asyncio.wait(
                                active_tasks,
                                timeout=0.05,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for task in done:
                                with suppress(Exception):
                                    task.result()
                            active_tasks = set(pending)
                        continue

                    while len(active_tasks) >= int(translation_runtime.parallelism):
                        done, pending = await asyncio.wait(
                            active_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            with suppress(Exception):
                                task.result()
                        active_tasks = set(pending)

                    task = asyncio.create_task(
                        _translate_one(
                            str(sentence_id),
                            str(sentence_text),
                            str(language),
                            int(seq_hint or 0),
                            int(state_gen_hint or 0),
                            str(source_language or ""),
                            str(target_language or ""),
                            str(direction or ""),
                        )
                    )
                    active_tasks.add(task)

                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
            finally:
                if active_tasks:
                    for task in active_tasks:
                        task.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                translation_runtime.task = None

        def _request_sentence_translation(sentence_id: str, sentence_text: str, language: str, seq_hint: int) -> None:
            if translator is None:
                return
            src = str(sentence_text or "").strip()
            if not src:
                return
            direction = str(getattr(translation_runtime, "direction", "zh2en") or "zh2en")
            source_language = str(getattr(translation_runtime, "source_language", "") or "")
            target_language = str(getattr(translation_runtime, "target_language", "") or "")

            item = (
                str(sentence_id),
                src,
                str(language or ""),
                int(seq_hint or 0),
                int(state_generation),
                source_language,
                target_language,
                direction,
            )
            try:
                translation_runtime.queue.put_nowait(item)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    translation_runtime.queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    translation_runtime.queue.put_nowait(item)
            _trace_event(
                "translation_queued",
                sentence_id=str(sentence_id),
                seq=int(seq_hint or 0),
                src_chars=len(src),
                direction=direction,
                source_language=source_language,
                target_language=target_language,
                queue_depth=int(translation_runtime.queue.qsize()),
            )

            if translation_runtime.task is None or translation_runtime.task.done():
                translation_runtime.task = asyncio.create_task(_translation_worker())

        def _track_text_progress(text: str) -> None:
            nonlocal last_text_snapshot, last_text_advance_at
            snapshot = str(text or "").strip()
            if not snapshot:
                return
            if snapshot != last_text_snapshot:
                prev_len = len(last_text_snapshot)
                last_text_snapshot = snapshot
                last_text_advance_at = time.monotonic()
                _trace_text_pool(
                    "pool_generating_set",
                    phase="generating",
                    text=snapshot,
                    reason="partial",
                    seq_hint=int(seq or 0),
                    delta_chars=max(0, len(snapshot) - prev_len),
                )

        def _rms_db(samples: np.ndarray) -> float:
            if samples is None or int(samples.size) <= 0:
                return -120.0
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))
            return float(20.0 * np.log10(max(rms, 1e-12)))

        def _reset_backend_vad_segment(reset_cut_clock: bool = False) -> None:
            backend_vad.in_speech = False
            backend_vad.speech_confirm_ms = 0.0
            backend_vad.silence_ms = 0.0
            backend_vad.segment_active_ms = 0.0
            backend_vad.segment_elapsed_ms = 0.0
            if reset_cut_clock:
                backend_vad.last_cut_at = time.monotonic()

        def _update_backend_vad(wav: np.ndarray) -> Dict[str, Any]:
            if wav is None or int(wav.size) <= 0:
                return {"candidate": False, "force": False, "silence_ms": float(backend_vad.silence_ms)}

            frame_size = int(max(80, vad_frame_samples))
            snr_db_last = 0.0
            for off in range(0, int(wav.size), frame_size):
                frame = wav[off : off + frame_size]
                if frame.size <= 0:
                    continue
                frame_ms = (float(frame.size) / float(SAMPLE_RATE)) * 1000.0
                db = _rms_db(frame)
                if (not backend_vad.in_speech) or db < float(backend_vad.noise_db) + 6.0:
                    backend_vad.noise_db = (0.985 * float(backend_vad.noise_db)) + (0.015 * float(db))
                snr_db = float(db - float(backend_vad.noise_db))
                snr_db_last = snr_db

                if snr_db >= float(vad_enter_snr_db):
                    backend_vad.speech_confirm_ms = min(3000.0, float(backend_vad.speech_confirm_ms) + frame_ms)
                else:
                    backend_vad.speech_confirm_ms = max(0.0, float(backend_vad.speech_confirm_ms) - frame_ms * 0.5)

                if (not backend_vad.in_speech) and float(backend_vad.speech_confirm_ms) >= 120.0:
                    backend_vad.in_speech = True
                    backend_vad.silence_ms = 0.0

                backend_vad.segment_elapsed_ms = float(backend_vad.segment_elapsed_ms) + frame_ms
                if backend_vad.in_speech:
                    if snr_db <= float(vad_exit_snr_db):
                        backend_vad.silence_ms = float(backend_vad.silence_ms) + frame_ms
                    else:
                        backend_vad.segment_active_ms = float(backend_vad.segment_active_ms) + frame_ms
                        backend_vad.silence_ms = max(0.0, float(backend_vad.silence_ms) - frame_ms * 0.5)

            since_last_cut_ms = (time.monotonic() - float(backend_vad.last_cut_at)) * 1000.0
            can_cut = bool(
                backend_vad.in_speech
                and float(backend_vad.segment_elapsed_ms) >= float(vad_min_slice_ms)
                and float(backend_vad.segment_active_ms) >= float(vad_min_active_ms)
                and float(backend_vad.silence_ms) >= float(vad_silence_trigger_ms)
                and since_last_cut_ms >= min(float(vad_min_slice_ms), 1500.0)
            )
            force_cut = bool(can_cut and float(backend_vad.silence_ms) >= float(vad_force_silence_ms))
            return {
                "candidate": can_cut,
                "force": force_cut,
                "silence_ms": float(backend_vad.silence_ms),
                "snr_db": float(snr_db_last),
                "segment_active_ms": float(backend_vad.segment_active_ms),
                "segment_elapsed_ms": float(backend_vad.segment_elapsed_ms),
            }

        def _text_ready_for_vad_cut(text: str, force: bool = False) -> bool:
            snapshot = str(text or "").strip()
            if not snapshot:
                return False
            if force:
                return True
            if bool(re.search(r"[。！？!?…]+[\"'”’)\]）】》]*$", snapshot)):
                return True
            idle_ms = (time.monotonic() - float(last_text_advance_at)) * 1000.0
            min_idle_ms = max(float(text_stable_cut_ms), 1200.0)
            if idle_ms < min_idle_ms:
                return False
            if _has_cjk(snapshot):
                return len(snapshot) >= MIN_CJK_SENTENCE_CHARS
            return len(snapshot) >= 20

        def _should_commit_tail_on_segment_finalize(reason: str, final_text: str, force_finalize: bool = False) -> bool:
            cut_reason = str(reason or "").strip()
            snapshot = str(final_text or "").strip()
            if not snapshot:
                return False
            if cut_reason == "vad_silence":
                if force_finalize:
                    return True
                # Require stronger confirmation for VAD silence cuts:
                # non-forced boundaries should keep tail pending to avoid
                # punctuation hallucination causing premature sentence commit.
                return False
            if cut_reason == "hard_cut":
                return False
            return True

        def _compose_effective_text_for_commit(full_text: str, seq_no: int) -> str:
            raw = str(full_text or "").strip()
            pending_prefix = str(getattr(subtitle_state, "pending_prefix_text", "") or "").strip()
            merged = raw
            if pending_prefix:
                merged = dedup_segment_join(pending_prefix, raw, min_overlap=2).strip()

            boundary_anchor = str(getattr(subtitle_state, "boundary_anchor_text", "") or "").strip()
            overlap_cap = int(getattr(subtitle_state, "boundary_overlap_cap_chars", 12) or 12)
            if overlap_cap > 0 and raw and pending_prefix:
                trimmed, overlap = trim_prefix_overlap(
                    pending_prefix,
                    raw,
                    min_overlap=2,
                    max_overlap=overlap_cap,
                )
                if overlap > 0:
                    merged = dedup_segment_join(pending_prefix, trimmed, min_overlap=2).strip()
                    _trace_event(
                        "pending_prefix_overlap_trimmed",
                        seq=int(seq_no or 0),
                        overlap_chars=int(overlap),
                        cap_chars=int(overlap_cap),
                        pending_chars=len(pending_prefix),
                        raw_chars=len(raw),
                        merged_chars=len(merged),
                        pending_segment_id=int(getattr(subtitle_state, "pending_prefix_segment_id", 0) or 0),
                    )
            elif overlap_cap > 0 and boundary_anchor and merged:
                trimmed, overlap = trim_prefix_overlap(
                    boundary_anchor,
                    merged,
                    min_overlap=2,
                    max_overlap=overlap_cap,
                )
                if overlap > 0:
                    merged = str(trimmed or "").strip()
                    _trace_event(
                        "boundary_anchor_overlap_trimmed",
                        seq=int(seq_no or 0),
                        overlap_chars=int(overlap),
                        cap_chars=int(overlap_cap),
                        anchor_chars=len(boundary_anchor),
                        merged_chars=len(merged),
                        anchor_segment_id=int(getattr(subtitle_state, "boundary_anchor_segment_id", 0) or 0),
                    )
            return merged

        def _compute_text_delta(prev_text: str, next_text: str) -> Tuple[str, bool]:
            prev = str(prev_text or "").strip()
            nxt = str(next_text or "").strip()
            if not nxt:
                return "", False
            if not prev:
                return nxt, False
            if nxt.startswith(prev):
                return nxt[len(prev):], False
            if prev.startswith(nxt):
                return "", True

            n = min(len(prev), len(nxt))
            i = 0
            while i < n and prev[i] == nxt[i]:
                i += 1

            if i >= max(8, int(len(prev) * 0.65)):
                return nxt[i:], False
            return nxt, True

        def _apply_incremental_text_fields(payload: Dict[str, Any]) -> None:
            full_text = str(payload.get("text", "") or "").strip()
            prev_text = str(stream_text_state.last_text or "")
            delta_text, text_reset = _compute_text_delta(prev_text, full_text)
            payload["state_text"] = full_text
            payload["delta_text"] = delta_text
            payload["text_reset"] = bool(text_reset)
            stream_text_state.last_text = full_text

        async def _maybe_idle_tail_commit() -> None:
            nonlocal last_idle_commit_at, last_text_advance_at
            if finish_requested or stop_consumer.is_set():
                return
            snapshot = str(last_text_snapshot or "").strip()
            if not snapshot:
                return
            idle_elapsed = time.monotonic() - float(last_text_advance_at)
            if idle_elapsed < float(idle_commit_sec):
                return
            now = time.monotonic()
            if (now - float(last_idle_commit_at)) < 1.0:
                return
            if state is None:
                return

            seq_hint = int(seq or 0)
            language = str(getattr(state, "language", "") or "")
            preview_completed, preview_tail = _split_sentences_and_tail(snapshot)
            tail_preview = str(preview_tail or "").strip()
            tail_looks_complete = bool(re.search(r"[。！？!?…]+[\"'”’)\]）】》]*$", tail_preview))
            tail_meets_min_len = bool(
                tail_preview
                and (
                    (_has_cjk(tail_preview) and len(tail_preview) >= 4)
                    or ((not _has_cjk(tail_preview)) and len(tail_preview) >= 12)
                )
            )
            allow_tail_commit = bool(tail_looks_complete or tail_meets_min_len)
            committed_before = len(subtitle_state.committed_sentences)
            tentative_before = str(subtitle_state.tentative_tail or "")
            tentative_after = await _update_sentence_commits(
                snapshot,
                language,
                seq_hint,
                force_tail=False,
                holdback_newest=False,
                commit_tail_if_no_completed=True,
                commit_tail_always=allow_tail_commit,
                commit_all_completed=True,
                slice_commit=True,
                translate_now=False,
            )
            committed_after = len(subtitle_state.committed_sentences)
            last_idle_commit_at = now
            last_text_advance_at = now
            if committed_after > committed_before:
                _trace_event(
                    "idle_tail_commit",
                    seq=int(seq_hint),
                    idle_ms=int(idle_elapsed * 1000),
                    committed_added=int(committed_after - committed_before),
                    preview_completed=int(len(preview_completed)),
                    preview_tail_chars=len(tail_preview),
                    allow_tail_commit=bool(allow_tail_commit),
                    tentative_before_chars=len(tentative_before.strip()),
                    tentative_after_chars=len(str(tentative_after or "").strip()),
                )

        async def _finalize_segment_and_rotate(
            *,
            reason: str,
            seq_hint: int,
            snapshot_text: str,
            snapshot_language: str,
            force_finalize: bool,
        ) -> bool:
            nonlocal state, seq, last_text_snapshot, last_text_advance_at, last_idle_commit_at, last_partial_emit_at
            if finish_requested or stop_consumer.is_set():
                return False
            if not use_vllm_streaming:
                return False

            async with state_lock:
                local_state = state
            if local_state is None:
                return False

            _trace_event(
                "segment_finalize_start",
                reason=str(reason or ""),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                seq=int(seq_hint or 0),
                snapshot_chars=len(str(snapshot_text or "").strip()),
                force_finalize=bool(force_finalize),
            )
            _trace_text_pool(
                "segment_finalize_start",
                phase="generating",
                text=str(snapshot_text or ""),
                reason=str(reason or ""),
                seq_hint=int(seq_hint or 0),
                delta_chars=0,
            )

            try:
                async with infer_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, local_state)
            except Exception as e:
                _trace_event(
                    "segment_finalize_failed",
                    reason=str(reason or ""),
                    error=str(e),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                )
                return False

            async with state_lock:
                if state is not local_state:
                    return False

            seq = max(int(seq), int(seq_hint or 0)) + 1
            final_text = str(getattr(local_state, "text", "") or snapshot_text or "").strip()
            final_language = str(getattr(local_state, "language", "") or snapshot_language or "")
            commit_tail_on_finalize = _should_commit_tail_on_segment_finalize(
                str(reason or ""),
                final_text,
                force_finalize=bool(force_finalize),
            )

            _track_text_progress(final_text)
            tentative_after_finalize = await _update_sentence_commits(
                final_text,
                final_language,
                seq,
                force_tail=bool(commit_tail_on_finalize),
                holdback_newest=False,
                commit_tail_if_no_completed=bool(commit_tail_on_finalize),
                commit_tail_always=bool(commit_tail_on_finalize),
                commit_all_completed=True,
                slice_commit=True,
                translate_now=False,
            )
            pending_prefix = ""
            if not commit_tail_on_finalize:
                pending_prefix = str(tentative_after_finalize or "").strip()
                if not pending_prefix:
                    _, pending_tail = _split_sentences_and_tail(final_text)
                    pending_prefix = str(pending_tail or "").strip()
            overlap_cap_chars = int(getattr(subtitle_state, "boundary_overlap_cap_chars", 12) or 12)
            boundary_anchor_chars = max(4, overlap_cap_chars * 2)
            boundary_anchor = str(final_text[-boundary_anchor_chars:] if final_text else "").strip()
            subtitle_state.pending_prefix_text = str(pending_prefix or "")
            subtitle_state.pending_prefix_segment_id = int(getattr(segment_runtime, "id", 0) or 0)
            subtitle_state.boundary_anchor_text = str(boundary_anchor or "")
            subtitle_state.boundary_anchor_segment_id = int(getattr(segment_runtime, "id", 0) or 0)
            _trace_text_pool(
                "pending_prefix_set",
                phase="generating",
                text=str(subtitle_state.pending_prefix_text or ""),
                reason=str(reason or ""),
                seq_hint=int(seq),
                delta_chars=len(str(subtitle_state.pending_prefix_text or "").strip()),
                commit_tail=bool(commit_tail_on_finalize),
                boundary_anchor_chars=len(str(subtitle_state.boundary_anchor_text or "").strip()),
            )
            subtitle_state.commit_base = int(len(subtitle_state.committed_sentences))
            subtitle_state.prev_completed_sentences = []
            subtitle_state.tentative_tail = ""

            overlap_audio = np.zeros((0,), dtype=np.float32)
            local_audio_accum = getattr(local_state, "audio_accum", None)
            if isinstance(local_audio_accum, np.ndarray) and int(local_audio_accum.size) > 0 and segment_overlap_samples > 0:
                take = min(int(local_audio_accum.size), int(segment_overlap_samples))
                overlap_audio = np.asarray(local_audio_accum[-take:], dtype=np.float32).copy()

            new_state = await asyncio.to_thread(_new_vllm_state, session_force_language)
            if int(overlap_audio.size) > 0:
                new_state.audio_accum = overlap_audio

            async with state_lock:
                if state is local_state:
                    state = new_state

            old_segment_id = int(getattr(segment_runtime, "id", 0) or 0)
            segment_runtime.id = old_segment_id + 1
            segment_runtime.started_at = time.monotonic()
            segment_runtime.last_cut_reason = str(reason or "")
            backend_vad.last_cut_at = time.monotonic()
            _reset_backend_vad_segment(reset_cut_clock=True)

            stream_text_state.last_text = ""
            last_text_snapshot = ""
            last_text_advance_at = time.monotonic()
            last_idle_commit_at = 0.0
            last_partial_emit_at = time.monotonic()

            _trace_text_pool(
                "pool_generating_reset",
                phase="generating",
                text="",
                reason=str(reason or ""),
                seq_hint=int(seq),
                delta_chars=0,
            )
            _trace_event(
                "segment_finalize_done",
                reason=str(reason or ""),
                old_segment_id=int(old_segment_id),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                seq=int(seq),
                final_text_chars=len(final_text),
                overlap_samples=int(overlap_audio.size),
                commit_tail=bool(commit_tail_on_finalize),
                pending_prefix_chars=len(str(subtitle_state.pending_prefix_text or "").strip()),
            )
            _trace_text_pool(
                "segment_open",
                phase="generating",
                text="",
                reason=str(reason or ""),
                seq_hint=int(seq),
                delta_chars=0,
                prev_segment_id=int(old_segment_id),
            )
            return True

        async def _maybe_vad_silence_cut(
            vad_signal: Dict[str, Any],
            full_text: str,
            language: str,
            seq_hint: int,
        ) -> None:
            if finish_requested or stop_consumer.is_set():
                return
            if not use_vllm_streaming:
                return
            signal = vad_signal if isinstance(vad_signal, dict) else {}
            snapshot = str(full_text or "").strip()
            if not snapshot:
                snapshot = str(last_text_snapshot or "").strip()

            segment_age_ms = (time.monotonic() - float(getattr(segment_runtime, "started_at", time.monotonic()))) * 1000.0
            decision = segment_policy.evaluate(
                silence_ms=float(signal.get("silence_ms", 0.0) or 0.0),
                segment_age_ms=float(segment_age_ms),
                segment_active_ms=float(signal.get("segment_active_ms", 0.0) or 0.0),
                has_pending_text=bool(snapshot),
                vad_candidate=bool(signal.get("candidate", False)),
                vad_force=bool(signal.get("force", False)),
            )
            if not decision.should_cut:
                return
            if decision.reason == "vad_silence" and not _text_ready_for_vad_cut(snapshot, force=bool(decision.force_finalize)):
                _trace_event(
                    "segment_cut_deferred",
                    reason=str(decision.reason),
                    seq=int(seq_hint or 0),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    silence_ms=int(decision.silence_ms),
                    segment_age_ms=int(decision.segment_age_ms),
                )
                return
            _trace_event(
                "segment_cut_decision",
                reason=str(decision.reason),
                seq=int(seq_hint or 0),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                silence_ms=int(decision.silence_ms),
                segment_age_ms=int(decision.segment_age_ms),
                force_finalize=bool(decision.force_finalize),
                text_chars=len(snapshot),
            )
            await _finalize_segment_and_rotate(
                reason=str(decision.reason),
                seq_hint=int(seq_hint or 0),
                snapshot_text=snapshot,
                snapshot_language=str(language or ""),
                force_finalize=bool(decision.force_finalize),
            )

        def _is_committed_sentence_upgrade(old_text: str, new_text: str) -> bool:
            old = str(old_text or "").strip()
            new = str(new_text or "").strip()
            if not old or not new or old == new:
                return False
            if len(new) < len(old) + 8:
                return False
            if new.startswith(old):
                return True
            if old in new and len(new) >= len(old) + 12:
                return True
            old_base = re.sub(r"[。！？!?…]+[\"'”’)\]）】》]*$", "", old).strip()
            new_base = re.sub(r"[。！？!?…]+[\"'”’)\]）】》]*$", "", new).strip()
            if old_base and new_base and len(new_base) >= len(old_base) + 6:
                if new_base.startswith(old_base):
                    return True
                if old_base in new_base and len(new_base) >= len(old_base) + 10:
                    return True
            return False

        async def _update_sentence_commits(
            full_text: str,
            language: str,
            seq_hint: int,
            force_tail: bool = False,
            holdback_newest: bool = True,
            commit_tail_if_no_completed: bool = False,
            commit_tail_always: bool = False,
            commit_all_completed: bool = False,
            slice_commit: bool = False,
            translate_now: bool = False,
        ) -> str:
            seq_no = int(seq_hint or 0)
            total_committed_count = len(subtitle_state.committed_sentences)
            commit_base = int(getattr(subtitle_state, "commit_base", 0) or 0)
            commit_base = max(0, min(commit_base, total_committed_count))
            prev_committed_count = int(total_committed_count - commit_base)
            prev_tail_text = str(subtitle_state.tentative_tail or "")
            prev_completed_count = len(subtitle_state.prev_completed_sentences)
            pending_prefix_before = str(getattr(subtitle_state, "pending_prefix_text", "") or "").strip()
            boundary_anchor_before = str(getattr(subtitle_state, "boundary_anchor_text", "") or "").strip()
            effective_full_text = _compose_effective_text_for_commit(full_text, seq_no)
            completed, tail = _split_sentences_and_tail(effective_full_text)
            if commit_tail_always and not force_tail and tail:
                tail_text = str(tail or "").strip()
                if tail_text:
                    completed.append(tail_text)
                    tail = ""
            elif commit_tail_if_no_completed and not force_tail and not completed and tail:
                tail_text = str(tail or "").strip()
                if tail_text:
                    if (_has_cjk(tail_text) and len(tail_text) >= MIN_CJK_SENTENCE_CHARS) or (
                        (not _has_cjk(tail_text)) and len(tail_text) >= 20
                    ):
                        completed.append(tail_text)
                        tail = ""
            if force_tail and tail:
                completed.append(tail)
                tail = ""

            committed_count = int(len(subtitle_state.committed_sentences) - commit_base)
            ready_end = committed_count
            if force_tail or commit_all_completed:
                ready_end = len(completed)
            else:
                upper = min(len(completed), len(subtitle_state.prev_completed_sentences))
                i = committed_count
                while i < upper and completed[i] == subtitle_state.prev_completed_sentences[i]:
                    i += 1
                ready_end = i
            ready_end = max(ready_end, committed_count)
            if not force_tail and holdback_newest:
                # Keep the newest completed sentence as tentative text so it can continue
                # growing without being frozen as a committed row too early.
                newest_holdback = max(committed_count, len(completed) - 1)
                ready_end = min(ready_end, newest_holdback)

            should_sample = (
                seq_no <= 3
                or seq_no % subtitle_trace_log_partial_every == 0
                or force_tail
                or commit_tail_always
                or commit_all_completed
                or bool(slice_commit)
                or ready_end != committed_count
            )
            if should_sample:
                _trace_event(
                    "commit_eval",
                    seq=seq_no,
                    full_chars=len(str(full_text or "").strip()),
                    effective_full_chars=len(str(effective_full_text or "").strip()),
                    completed_count=len(completed),
                    tail_chars=len(str(tail or "").strip()),
                    commit_base=int(commit_base),
                    total_committed_count=int(total_committed_count),
                    prev_committed_count=int(prev_committed_count),
                    prev_completed_count=int(prev_completed_count),
                    ready_end=int(ready_end),
                    holdback_newest=bool(holdback_newest),
                    force_tail=bool(force_tail),
                    commit_tail_if_no_completed=bool(commit_tail_if_no_completed),
                    commit_tail_always=bool(commit_tail_always),
                    commit_all_completed=bool(commit_all_completed),
                    slice_commit=bool(slice_commit),
                    translate_now=bool(translate_now),
                    pending_prefix_chars=len(pending_prefix_before),
                    boundary_anchor_chars=len(boundary_anchor_before),
                )

            active_sentence_items = max(0, int(len(subtitle_state.sentence_items) - commit_base))
            update_upper = min(committed_count, len(completed), active_sentence_items)
            committed_added = 0
            for i in range(update_upper):
                upgraded = str(completed[i] or "").strip()
                global_idx = int(commit_base + i)
                current = str(subtitle_state.committed_sentences[global_idx] or "").strip()
                if not _is_committed_sentence_upgrade(current, upgraded):
                    continue
                subtitle_state.committed_sentences[global_idx] = upgraded
                sentence_item = subtitle_state.sentence_items[global_idx]
                sentence_id = str(sentence_item.get("id") or "")
                sentence_item["zh"] = upgraded
                sentence_item["en"] = ""
                _trace_event(
                    "sentence_upgrade_commit",
                    seq=seq_no,
                    idx=int(i),
                    global_idx=int(global_idx),
                    sentence_id=sentence_id,
                    old_chars=len(current),
                    new_chars=len(upgraded),
                    slice_commit=bool(slice_commit),
                )
                _trace_text_pool(
                    "pool_solidified_update",
                    phase="solidified",
                    text=upgraded,
                    reason="sentence_updated",
                    seq_hint=int(seq_hint or 0),
                    sentence_id=sentence_id,
                    delta_chars=max(0, len(upgraded) - len(current)),
                    slice_commit=bool(slice_commit),
                )
                await _send_json(
                    {
                        "type": "sentence_updated",
                        "sentence_id": sentence_id,
                        "text": upgraded,
                        "language": str(language or ""),
                        "seq": int(seq_hint or 0),
                        "ts_ms": int(time.time() * 1000),
                        "slice_commit": bool(slice_commit),
                    }
                )
                await _send_json(
                    {
                        "type": "sentence_translation",
                        "sentence_id": sentence_id,
                        "translation": "",
                        "seq": int(seq_hint or 0),
                    }
                )
                if translate_now:
                    source_language = str(getattr(translation_runtime, "source_language", "") or "")
                    target_language = str(getattr(translation_runtime, "target_language", "") or "")
                    direction = str(getattr(translation_runtime, "direction", "zh2en") or "zh2en")
                    translated = await _translate_sentence_once(
                        upgraded,
                        language,
                        seq_hint,
                        source_language,
                        target_language,
                        direction,
                    )
                    if translated:
                        sentence_item["en"] = translated
                        await _send_json(
                            {
                                "type": "sentence_translation",
                                "sentence_id": sentence_id,
                                "translation": translated,
                                "seq": int(seq_hint or 0),
                            }
                        )
                else:
                    _request_sentence_translation(sentence_id, upgraded, language, seq_hint)

            for i in range(committed_count, ready_end):
                sentence = str(completed[i] or "").strip()
                if not sentence:
                    continue
                sentence_id = _new_sentence_id()
                subtitle_state.committed_sentences.append(sentence)
                subtitle_state.sentence_items.append({"id": sentence_id, "zh": sentence, "en": ""})
                committed_added += 1
                _trace_event(
                    "sentence_new_commit",
                    seq=seq_no,
                    idx=int(i),
                    global_idx=int(len(subtitle_state.committed_sentences) - 1),
                    sentence_id=sentence_id,
                    chars=len(sentence),
                    slice_commit=bool(slice_commit),
                )
                _trace_text_pool(
                    "pool_solidified_append",
                    phase="solidified",
                    text=sentence,
                    reason="sentence_committed",
                    seq_hint=int(seq_hint or 0),
                    sentence_id=sentence_id,
                    delta_chars=len(sentence),
                    slice_commit=bool(slice_commit),
                )
                await _send_json(
                    {
                        "type": "sentence_committed",
                        "sentence_id": sentence_id,
                        "text": sentence,
                        "language": str(language or ""),
                        "seq": int(seq_hint or 0),
                        "ts_ms": int(time.time() * 1000),
                        "slice_commit": bool(slice_commit),
                    }
                )
                if translate_now:
                    source_language = str(getattr(translation_runtime, "source_language", "") or "")
                    target_language = str(getattr(translation_runtime, "target_language", "") or "")
                    direction = str(getattr(translation_runtime, "direction", "zh2en") or "zh2en")
                    translated = await _translate_sentence_once(
                        sentence,
                        language,
                        seq_hint,
                        source_language,
                        target_language,
                        direction,
                    )
                    if translated:
                        subtitle_state.sentence_items[-1]["en"] = translated
                        await _send_json(
                            {
                                "type": "sentence_translation",
                                "sentence_id": sentence_id,
                                "translation": translated,
                                "seq": int(seq_hint or 0),
                            }
                        )
                else:
                    _request_sentence_translation(sentence_id, sentence, language, seq_hint)

            if pending_prefix_before and (committed_added > 0 or bool(force_tail)):
                subtitle_state.pending_prefix_text = ""
                subtitle_state.pending_prefix_segment_id = 0
                _trace_text_pool(
                    "pending_prefix_cleared",
                    phase="generating",
                    text="",
                    reason="sentence_committed",
                    seq_hint=int(seq_hint or 0),
                    delta_chars=-len(pending_prefix_before),
                    committed_added=int(committed_added),
                    force_tail=bool(force_tail),
                )
            if boundary_anchor_before and (committed_added > 0 or bool(force_tail)):
                subtitle_state.boundary_anchor_text = ""
                subtitle_state.boundary_anchor_segment_id = 0
                _trace_text_pool(
                    "boundary_anchor_cleared",
                    phase="generating",
                    text="",
                    reason="sentence_committed",
                    seq_hint=int(seq_hint or 0),
                    delta_chars=-len(boundary_anchor_before),
                    committed_added=int(committed_added),
                    force_tail=bool(force_tail),
                )

            subtitle_state.prev_completed_sentences = completed
            if force_tail:
                subtitle_state.tentative_tail = ""
            else:
                pending_segments = [str(seg or "").strip() for seg in completed[ready_end:]]
                pending_segments = [seg for seg in pending_segments if seg]
                if tail:
                    pending_segments.append(str(tail).strip())
                subtitle_state.tentative_tail = _join_segments(pending_segments)
            next_tail_text = str(subtitle_state.tentative_tail or "")
            if should_sample or next_tail_text != prev_tail_text:
                _trace_event(
                    "tentative_tail_update",
                    seq=seq_no,
                    prev_tail_chars=len(prev_tail_text.strip()),
                    next_tail_chars=len(next_tail_text.strip()),
                    commit_base=int(commit_base),
                    committed_count=int(len(subtitle_state.committed_sentences) - commit_base),
                    total_committed_count=len(subtitle_state.committed_sentences),
                )
            return subtitle_state.tentative_tail

        async def _send_json(payload: Dict[str, Any]) -> None:
            if subtitle_trace_log:
                p = payload if isinstance(payload, dict) else {}
                msg_type = str(p.get("type", "")).strip()
                if msg_type:
                    if msg_type in {"partial", "final"}:
                        _trace_event(
                            "ws_send",
                            type=msg_type,
                            seq=int(p.get("seq", 0) or 0),
                            text_chars=len(str(p.get("text", "") or "").strip()),
                            delta_chars=len(str(p.get("delta_text", "") or "").strip()),
                            text_reset=bool(p.get("text_reset", False)),
                            tentative_chars=len(str(p.get("tentative_text", "") or "").strip()),
                            committed_chars=len(str(p.get("committed_text", "") or "").strip()),
                            translation_chars=len(str(p.get("translation", "") or "").strip()),
                        )
                    elif msg_type in {
                        "started",
                        "sentence_committed",
                        "sentence_updated",
                        "sentence_translation",
                        "sentence_reset",
                        "processing",
                        "error",
                    }:
                        _trace_event(
                            "ws_send",
                            type=msg_type,
                            seq=int(p.get("seq", 0) or 0),
                            sentence_id=str(p.get("sentence_id", "") or ""),
                            text_chars=len(str(p.get("text", "") or "").strip()),
                            translation_chars=len(str(p.get("translation", "") or "").strip()),
                            reason=str(p.get("reason", "") or ""),
                            message=str(p.get("message", "") or ""),
                        )
            async with send_lock:
                await websocket.send_json(payload)

        def _drop_oldest_audio() -> bool:
            nonlocal queue_samples
            try:
                _, dropped_wav = audio_queue.get_nowait()
                queue_samples = max(0, int(queue_samples - int(getattr(dropped_wav, "size", 0))))
                stats.queue_dropped += 1
                return True
            except asyncio.QueueEmpty:
                return False

        def _clear_audio_queue() -> int:
            dropped = 0
            while _drop_oldest_audio():
                dropped += 1
            return dropped

        def _enqueue_audio(gen: int, wav: np.ndarray) -> None:
            nonlocal queue_samples
            dropped_now = 0
            try:
                audio_queue.put_nowait((gen, wav))
                queue_samples += int(wav.size)
            except asyncio.QueueFull:
                if _drop_oldest_audio():
                    dropped_now += 1
                try:
                    audio_queue.put_nowait((gen, wav))
                    queue_samples += int(wav.size)
                except asyncio.QueueFull:
                    stats.queue_dropped += 1
                    dropped_now += 1
            pressure = backpressure.evaluate(int(queue_samples))
            if pressure.drop_oldest:
                while backpressure.evaluate(int(queue_samples)).drop_oldest:
                    if not _drop_oldest_audio():
                        break
                    dropped_now += 1
            if bool(pressure.under_pressure) != bool(backpressure_runtime.under_pressure):
                if pressure.under_pressure:
                    _trace_event(
                        "audio_backpressure_enter",
                        reason=str(pressure.reason),
                        queue_sec=round(float(pressure.queue_sec), 3),
                        queue_samples=int(queue_samples),
                        queue_depth=int(audio_queue.qsize()),
                    )
                else:
                    _trace_event(
                        "audio_backpressure_recover",
                        reason=str(backpressure_runtime.reason),
                        queue_sec=round(float(pressure.queue_sec), 3),
                        queue_samples=int(queue_samples),
                        queue_depth=int(audio_queue.qsize()),
                    )
            backpressure_runtime.under_pressure = bool(pressure.under_pressure)
            backpressure_runtime.reason = str(pressure.reason)
            stats.queue_depth_peak = max(stats.queue_depth_peak, int(audio_queue.qsize()))
            if dropped_now > 0:
                _trace_event(
                    "audio_backpressure_drop",
                    reason=str(pressure.reason),
                    dropped_now=int(dropped_now),
                    dropped_total=int(stats.queue_dropped),
                    queue_sec=round(float(pressure.queue_sec), 3),
                    queue_samples=int(queue_samples),
                    queue_depth=int(audio_queue.qsize()),
                    frame_samples=int(wav.size),
                )

        def _coalesce_audio_frames(gen: int, wav: np.ndarray) -> np.ndarray:
            nonlocal queue_samples
            depth_before = int(audio_queue.qsize())
            pressure = backpressure.evaluate(int(queue_samples))
            target_samples = int(max(1, int(consumer_max_batch_samples * float(pressure.suggested_batch_scale))))
            if depth_before >= max(4, audio_queue_size // 2):
                target_samples = max(target_samples, int(consumer_high_batch_samples))
            if wav.size >= target_samples:
                return wav
            chunks = [wav]
            total_samples = int(wav.size)
            merged = 1
            while total_samples < target_samples:
                try:
                    next_gen, next_wav = audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                queue_samples = max(0, int(queue_samples - int(next_wav.size)))
                if int(next_gen) != int(gen):
                    continue
                chunks.append(next_wav)
                total_samples += int(next_wav.size)
                merged += 1
            if merged <= 1:
                return wav
            _trace_event(
                "audio_batch_merge",
                merged=int(merged),
                samples=int(total_samples),
                queue_depth=int(audio_queue.qsize()),
                queue_depth_before=int(depth_before),
                target_samples=int(target_samples),
                queue_sec=round(float(pressure.queue_sec), 3),
            )
            return np.concatenate(chunks, axis=0)

        async def _audio_consumer() -> None:
            nonlocal seq, finished, state
            nonlocal total_consumed_samples, queue_samples
            nonlocal last_text_snapshot, last_text_advance_at, last_idle_commit_at, last_partial_emit_at

            while not stop_consumer.is_set():
                if finish_requested and audio_queue.empty():
                    break
                try:
                    gen, wav = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    await _maybe_idle_tail_commit()
                    await _maybe_vad_silence_cut(
                        {
                            "candidate": False,
                            "force": False,
                            "silence_ms": float(getattr(backend_vad, "silence_ms", 0.0) or 0.0),
                            "segment_active_ms": float(getattr(backend_vad, "segment_active_ms", 0.0) or 0.0),
                        },
                        str(last_text_snapshot or ""),
                        str(getattr(state, "language", "") or "") if state is not None else "",
                        int(seq or 0),
                    )
                    continue
                queue_samples = max(0, int(queue_samples - int(wav.size)))

                async with state_lock:
                    local_state = state
                    local_gen = state_generation
                if local_state is None or gen != local_gen:
                    continue
                wav = _coalesce_audio_frames(gen, wav)
                total_consumed_samples += int(wav.size)
                vad_signal = _update_backend_vad(wav)

                if use_vllm_streaming:
                    async with infer_lock:
                        await asyncio.to_thread(asr.streaming_transcribe, wav, local_state)
                    async with state_lock:
                        if local_state is not state or gen != state_generation:
                            continue
                        seq += 1
                        payload = {
                            "type": "partial",
                            "language": getattr(local_state, "language", "") or "",
                            "text": getattr(local_state, "text", "") or "",
                            "seq": seq,
                        }
                    payload_text = str(payload.get("text", "") or "").strip()
                    if not payload_text:
                        await _maybe_vad_silence_cut(
                            vad_signal,
                            "",
                            payload.get("language", ""),
                            payload.get("seq", 0),
                        )
                        await _maybe_idle_tail_commit()
                        continue
                    if seq <= 3 or seq % subtitle_trace_log_partial_every == 0:
                        _trace_event(
                            "partial_model_output",
                            seq=int(seq),
                            language=str(payload.get("language", "") or ""),
                            text_chars=len(str(payload.get("text", "") or "").strip()),
                            queue_depth=int(audio_queue.qsize()),
                        )
                    _track_text_progress(payload.get("text", ""))
                    payload["tentative_text"] = await _update_sentence_commits(
                        payload.get("text", ""),
                        payload.get("language", ""),
                        payload.get("seq", 0),
                        force_tail=False,
                        holdback_newest=True,
                        commit_tail_if_no_completed=False,
                        commit_all_completed=False,
                        slice_commit=False,
                    )
                    _apply_incremental_text_fields(payload)
                    payload["committed_text"] = _join_segments(subtitle_state.committed_sentences)
                    payload["translation"] = _committed_translation_text()
                    stats.partial_msgs += 1
                    await _send_json(payload)
                    last_partial_emit_at = time.monotonic()
                    await _maybe_vad_silence_cut(
                        vad_signal,
                        payload.get("text", ""),
                        payload.get("language", ""),
                        payload.get("seq", 0),
                    )
                    await _maybe_idle_tail_commit()
                    continue

                if local_state.audio_accum.size == 0:
                    local_state.audio_accum = wav
                else:
                    local_state.audio_accum = np.concatenate([local_state.audio_accum, wav], axis=0)
                enough_audio = local_state.audio_accum.size >= local_state.min_decode_samples
                enough_delta = (
                    local_state.audio_accum.size - local_state.last_decoded_samples
                ) >= local_state.decode_interval_samples
                if not (enough_audio and enough_delta):
                    await _maybe_idle_tail_commit()
                    continue

                async with infer_lock:
                    out = await asyncio.to_thread(
                        lambda: asr.transcribe(
                            audio=[(local_state.audio_accum, SAMPLE_RATE)],
                            context="",
                            language=local_state.force_language,
                        )[0]
                    )
                local_state.language = getattr(out, "language", "") or ""
                local_state.text = getattr(out, "text", "") or ""
                local_state.last_decoded_samples = local_state.audio_accum.size
                async with state_lock:
                    if local_state is not state or gen != state_generation:
                        continue
                    seq += 1
                    payload = {
                        "type": "partial",
                        "language": getattr(local_state, "language", "") or "",
                        "text": getattr(local_state, "text", "") or "",
                        "seq": seq,
                    }
                payload_text = str(payload.get("text", "") or "").strip()
                if not payload_text:
                    await _maybe_vad_silence_cut(
                        vad_signal,
                        "",
                        payload.get("language", ""),
                        payload.get("seq", 0),
                    )
                    await _maybe_idle_tail_commit()
                    continue
                if seq <= 3 or seq % subtitle_trace_log_partial_every == 0:
                    _trace_event(
                        "partial_model_output",
                        seq=int(seq),
                        language=str(payload.get("language", "") or ""),
                        text_chars=len(str(payload.get("text", "") or "").strip()),
                        queue_depth=int(audio_queue.qsize()),
                    )
                _track_text_progress(payload.get("text", ""))
                payload["tentative_text"] = await _update_sentence_commits(
                    payload.get("text", ""),
                    payload.get("language", ""),
                    payload.get("seq", 0),
                    force_tail=False,
                    holdback_newest=True,
                    commit_tail_if_no_completed=False,
                    commit_all_completed=False,
                    slice_commit=False,
                )
                _apply_incremental_text_fields(payload)
                payload["committed_text"] = _join_segments(subtitle_state.committed_sentences)
                payload["translation"] = _committed_translation_text()
                stats.partial_msgs += 1
                await _send_json(payload)
                last_partial_emit_at = time.monotonic()
                await _maybe_vad_silence_cut(
                    vad_signal,
                    payload.get("text", ""),
                    payload.get("language", ""),
                    payload.get("seq", 0),
                )
                await _maybe_idle_tail_commit()

            if not finish_requested or stop_consumer.is_set():
                return

            async with state_lock:
                local_state = state
                local_gen = state_generation
            if local_state is None:
                return

            canonical_redecode_applied = False
            if use_vllm_streaming:
                async with infer_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, local_state)
                if finish_mode == "stop" and final_redecode_on_stop:
                    if full_audio_overflow:
                        _trace_event(
                            "final_redecode_skipped",
                            reason="audio_too_long",
                            full_audio_samples=int(full_audio_samples),
                            cap_samples=int(final_redecode_max_samples),
                        )
                        logger.info(
                            "skip final re-decode peer=%s reason=audio-too-long samples=%d cap=%d",
                            peer,
                            int(full_audio_samples),
                            int(final_redecode_max_samples),
                        )
                    else:
                        full_wav = np.concatenate(full_audio_parts, axis=0) if full_audio_parts else np.zeros((0,), dtype=np.float32)
                        if full_wav.size > 0:
                            await _send_json({"type": "processing"})
                            try:
                                saved_max_tokens = None
                                override_max_tokens = int(max(1, int(getattr(args, "final_redecode_max_new_tokens", 512))))
                                sampling_params = getattr(asr, "sampling_params", None)
                                if sampling_params is not None and hasattr(sampling_params, "max_tokens"):
                                    try:
                                        saved_max_tokens = int(getattr(sampling_params, "max_tokens"))
                                    except Exception:
                                        saved_max_tokens = None
                                    if saved_max_tokens is None or saved_max_tokens < override_max_tokens:
                                        setattr(sampling_params, "max_tokens", override_max_tokens)
                                async with infer_lock:
                                    full_out = await asyncio.to_thread(
                                        lambda: asr.transcribe(
                                            audio=[(full_wav, SAMPLE_RATE)],
                                            context="",
                                            language=(getattr(local_state, "force_language", None) or session_force_language),
                                        )[0]
                                    )
                                local_state.language = getattr(full_out, "language", "") or getattr(local_state, "language", "") or ""
                                local_state.text = getattr(full_out, "text", "") or getattr(local_state, "text", "") or ""
                                canonical_redecode_applied = True
                                _trace_event(
                                    "final_redecode_done",
                                    full_audio_samples=int(full_wav.size),
                                    text_chars=len(str(local_state.text or "").strip()),
                                )
                            except Exception as e:
                                logger.warning("final re-decode failed peer=%s err=%s", peer, e)
                                _trace_event("final_redecode_failed", error=str(e))
                            finally:
                                if saved_max_tokens is not None:
                                    with suppress(Exception):
                                        setattr(asr.sampling_params, "max_tokens", saved_max_tokens)
            else:
                if local_state.audio_accum.size > 0:
                    await _send_json({"type": "processing"})
                    async with infer_lock:
                        out = await asyncio.to_thread(
                            lambda: asr.transcribe(
                                audio=[(local_state.audio_accum, SAMPLE_RATE)],
                                context="",
                                language=local_state.force_language,
                            )[0]
                        )
                    local_state.language = getattr(out, "language", "") or ""
                    local_state.text = getattr(out, "text", "") or ""

            if canonical_redecode_applied:
                _trace_event(
                    "final_redecode_applied",
                    full_audio_samples=int(full_audio_samples),
                    finish_mode=str(finish_mode),
                )
                subtitle_state.stream_uid = f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}"
                subtitle_state.next_sentence_id = 1
                subtitle_state.committed_sentences = []
                subtitle_state.sentence_items = []
                subtitle_state.commit_base = 0
                subtitle_state.prev_completed_sentences = []
                subtitle_state.tentative_tail = ""
                subtitle_state.pending_prefix_text = ""
                subtitle_state.pending_prefix_segment_id = 0
                subtitle_state.boundary_anchor_text = ""
                subtitle_state.boundary_anchor_segment_id = 0
                await _send_json({"type": "sentence_reset", "reason": "final_redecode"})

            async with state_lock:
                if local_state is not state or local_gen != state_generation:
                    return
                seq += 1
                payload = {
                    "type": "final",
                    "language": getattr(local_state, "language", "") or "",
                    "text": getattr(local_state, "text", "") or "",
                    "seq": seq,
                }
            _trace_event(
                "final_model_output",
                seq=int(seq),
                language=str(payload.get("language", "") or ""),
                text_chars=len(str(payload.get("text", "") or "").strip()),
                finish_mode=str(finish_mode),
                finish_reason=str(finish_reason),
            )
            payload["tentative_text"] = await _update_sentence_commits(
                payload.get("text", ""),
                payload.get("language", ""),
                payload.get("seq", 0),
                force_tail=True,
                holdback_newest=False,
                commit_tail_if_no_completed=False,
                commit_tail_always=False,
                commit_all_completed=False,
                slice_commit=False,
                translate_now=True,
            )
            _apply_incremental_text_fields(payload)
            payload["committed_text"] = _join_segments(subtitle_state.committed_sentences)
            if translation_runtime.task is not None and not translation_runtime.task.done():
                with suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(translation_runtime.task), timeout=1.2)
            payload["translation"] = _committed_translation_text()
            finished = True
            stats.final_msgs += 1
            await _send_json(payload)

        try:
            if use_vllm_streaming:
                state = await asyncio.to_thread(_new_vllm_state, session_force_language)
            else:
                state = _new_transformers_state(session_force_language)

            await _send_json(
                {
                    "type": "ready",
                    "sample_rate": SAMPLE_RATE,
                    "translation_direction": str(translation_runtime.direction or "zh2en"),
                    "translation_source_language": str(translation_runtime.source_language or ""),
                    "translation_target_language": str(translation_runtime.target_language or ""),
                }
            )
            consumer_task = asyncio.create_task(_audio_consumer())

            while True:
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=float(args.idle_timeout_sec),
                    )
                except asyncio.TimeoutError:
                    await _send_json({"type": "error", "message": "idle timeout"})
                    break

                if msg.get("type") == "websocket.disconnect":
                    break

                raw = msg.get("bytes")
                text = msg.get("text")

                if raw is not None:
                    try:
                        wav = _decode_pcm16le(raw)
                    except ValueError as e:
                        stats.last_error = str(e)
                        await _send_json({"type": "error", "message": str(e)})
                        continue

                    if wav.size == 0:
                        continue
                    if wav.size > args.max_frame_samples:
                        stats.last_error = "audio frame too large"
                        await _send_json({"type": "error", "message": "audio frame too large"})
                        continue

                    stats.raw_frames += 1
                    stats.raw_samples += int(wav.size)
                    if stats.raw_frames <= 3 or stats.raw_frames % 40 == 0:
                        _trace_event(
                            "audio_frame_recv",
                            raw_frames=int(stats.raw_frames),
                            raw_samples=int(stats.raw_samples),
                            frame_samples=int(wav.size),
                        )
                    if final_redecode_on_stop:
                        if final_redecode_max_samples <= 0:
                            full_audio_parts.append(np.asarray(wav, dtype=np.float32).copy())
                            full_audio_samples += int(wav.size)
                        else:
                            remaining = max(0, int(final_redecode_max_samples - full_audio_samples))
                            if remaining <= 0:
                                full_audio_overflow = True
                            else:
                                take = min(int(wav.size), remaining)
                                if take > 0:
                                    full_audio_parts.append(np.asarray(wav[:take], dtype=np.float32).copy())
                                    full_audio_samples += int(take)
                                if take < int(wav.size):
                                    full_audio_overflow = True
                    if stats.raw_frames == 1 or stats.raw_frames % 20 == 0:
                        logger.info(
                            "ws recv peer=%s frames=%d samples=%d seq=%d",
                            peer,
                            stats.raw_frames,
                            stats.raw_samples,
                            seq,
                        )
                    async with state_lock:
                        gen = state_generation
                    _enqueue_audio(gen, wav)
                    continue

                if text is not None:
                    stats.text_msgs += 1
                    try:
                        payload = _parse_json_message(text)
                    except ValueError as e:
                        stats.last_error = str(e)
                        await _send_json({"type": "error", "message": str(e)})
                        continue

                    msg_type = str(payload.get("type", "")).lower()
                    _trace_event("ws_text_recv", type=msg_type)
                    if msg_type == "start":
                        stats.start_msgs += 1
                        finish_mode = "stop"
                        finish_reason = "stop"
                        finish_requested = False
                        finished = False
                        stream_text_state.last_text = ""
                        last_text_snapshot = ""
                        last_text_advance_at = time.monotonic()
                        last_idle_commit_at = 0.0
                        total_consumed_samples = 0
                        queue_samples = 0
                        last_partial_emit_at = time.monotonic()
                        segment_runtime.id = 1
                        segment_runtime.started_at = time.monotonic()
                        segment_runtime.last_cut_reason = "start"
                        _reset_backend_vad_segment(reset_cut_clock=True)
                        full_audio_parts = []
                        full_audio_samples = 0
                        full_audio_overflow = False
                        _trace_event(
                            "start_received",
                            requested_language=str(payload.get("language", "") or ""),
                            requested_translation_direction=str(payload.get("translation_direction", "") or ""),
                        )
                        if "language" in payload:
                            session_force_language = _normalize_force_language(payload.get("language"))
                        requested_translation_direction = _normalize_translation_direction(
                            payload.get("translation_direction", translation_runtime.direction)
                        )
                        try:
                            if use_vllm_streaming:
                                new_state = await asyncio.to_thread(_new_vllm_state, session_force_language)
                            else:
                                new_state = _new_transformers_state(session_force_language)
                            async with state_lock:
                                state_generation += 1
                                state = new_state
                                seq = 0
                            if translation_runtime.task is not None and not translation_runtime.task.done():
                                translation_runtime.task.cancel()
                            if translation_runtime.task is not None:
                                with suppress(asyncio.CancelledError, Exception):
                                    await translation_runtime.task
                            translation_runtime.task = None
                            translation_runtime.queue = asyncio.Queue(maxsize=256)
                            translation_runtime.direction = requested_translation_direction
                            (
                                translation_runtime.source_language,
                                translation_runtime.target_language,
                            ) = _resolve_direction_languages(requested_translation_direction)
                            subtitle_state.stream_uid = f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}"
                            subtitle_state.next_sentence_id = 1
                            subtitle_state.committed_sentences = []
                            subtitle_state.sentence_items = []
                            subtitle_state.commit_base = 0
                            subtitle_state.prev_completed_sentences = []
                            subtitle_state.tentative_tail = ""
                            subtitle_state.pending_prefix_text = ""
                            subtitle_state.pending_prefix_segment_id = 0
                            subtitle_state.boundary_anchor_text = ""
                            subtitle_state.boundary_anchor_segment_id = 0
                            _trace_text_pool(
                                "pool_generating_reset",
                                phase="generating",
                                text="",
                                reason="start",
                                seq_hint=0,
                            )
                            _trace_text_pool(
                                "segment_open",
                                phase="generating",
                                text="",
                                reason="start",
                                seq_hint=0,
                            )
                            dropped = _clear_audio_queue()
                            if dropped:
                                logger.info(
                                    "ws start resets queue peer=%s dropped=%d",
                                    peer,
                                    dropped,
                                )
                                _trace_event("start_queue_cleared", dropped=int(dropped))
                            _trace_event(
                                "start_applied",
                                force_language=session_force_language or "",
                                state_generation=int(state_generation),
                                translation_direction=str(translation_runtime.direction or ""),
                                translation_source_language=str(translation_runtime.source_language or ""),
                                translation_target_language=str(translation_runtime.target_language or ""),
                            )
                            if consumer_task is None or consumer_task.done():
                                consumer_task = asyncio.create_task(_audio_consumer())
                            await _send_json(
                                {
                                    "type": "started",
                                    "language": session_force_language or "",
                                    "translation_direction": str(translation_runtime.direction or "zh2en"),
                                    "translation_source_language": str(translation_runtime.source_language or ""),
                                    "translation_target_language": str(translation_runtime.target_language or ""),
                                }
                            )
                        except Exception as e:
                            stats.last_error = f"start failed: {e}"
                            _trace_event("start_failed", error=str(e))
                            await _send_json({"type": "error", "message": f"start failed: {e}"})
                        continue

                    if msg_type == "set_translation_direction":
                        requested_direction = _normalize_translation_direction(payload.get("translation_direction"))
                        await _set_translation_direction(
                            requested_direction,
                            clear_pending=True,
                            emit=True,
                        )
                        continue

                    if msg_type == "finish":
                        stats.finish_msgs += 1
                        requested_mode = str(payload.get("mode", "")).strip().lower()
                        requested_reason = str(payload.get("reason", "") or "").strip().lower()
                        finish_mode = "stop"
                        finish_reason = "stop"
                        if requested_mode == "slice":
                            _trace_event("finish_slice_ignored", requested_reason=requested_reason)
                        finish_requested = True
                        _trace_event(
                            "finish_received",
                            requested_mode=requested_mode,
                            applied_mode=finish_mode,
                            requested_reason=requested_reason,
                            applied_reason=finish_reason,
                            queue_depth=int(audio_queue.qsize()),
                        )
                        dropped = _clear_audio_queue()
                        if dropped:
                            logger.info(
                                "ws finish drops pending queue peer=%s dropped=%d",
                                peer,
                                dropped,
                            )
                            _trace_event("finish_queue_cleared", dropped=int(dropped))
                        if consumer_task is not None:
                            await consumer_task
                        break

                    if msg_type == "ping":
                        _trace_event("ping_received")
                        await _send_json({"type": "pong"})
                        continue

                    stats.last_error = "unknown message type"
                    _trace_event("unknown_message_type", type=msg_type)
                    await _send_json({"type": "error", "message": "unknown message type"})
                    continue

        except WebSocketDisconnect:
            _trace_event("ws_disconnect")
            pass
        except Exception as e:
            stats.last_error = str(e)
            _trace_event("ws_exception", error=str(e))
            try:
                await _send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            stop_consumer.set()
            if consumer_task is not None and not consumer_task.done():
                consumer_task.cancel()
            if consumer_task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await consumer_task
            if translation_runtime.task is not None and not translation_runtime.task.done():
                translation_runtime.task.cancel()
            if translation_runtime.task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await translation_runtime.task

            runtime.active_connections = max(0, runtime.active_connections - 1)
            should_finalize_on_disconnect = bool(getattr(args, "finalize_on_disconnect", False))
            if state is not None and not finished and should_finalize_on_disconnect:
                try:
                    if use_vllm_streaming:
                        async with infer_lock:
                            await asyncio.to_thread(asr.finish_streaming_transcribe, state)
                except Exception:
                    pass
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
            _trace_text_pool(
                "pool_snapshot",
                phase="solidified",
                text=_join_segments(subtitle_state.committed_sentences),
                reason="ws_close",
                seq_hint=int(seq or 0),
                delta_chars=0,
                solidified_count=int(len(subtitle_state.committed_sentences)),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
            )
            _trace_event(
                "ws_close",
                active_connections=int(runtime.active_connections),
                finished=bool(finished),
                raw_frames=int(stats.raw_frames),
                partial_msgs=int(stats.partial_msgs),
                final_msgs=int(stats.final_msgs),
                queue_dropped=int(stats.queue_dropped),
                queue_depth_peak=int(stats.queue_depth_peak),
                last_error=str(stats.last_error or ""),
            )
            logger.info(
                "ws close peer=%s active=%d finished=%s raw_frames=%d raw_samples=%d text_msgs=%d start=%d finish=%d partial=%d final=%d queue_dropped=%d queue_depth_peak=%d last_error=%s",
                peer,
                runtime.active_connections,
                finished,
                stats.raw_frames,
                stats.raw_samples,
                stats.text_msgs,
                stats.start_msgs,
                stats.finish_msgs,
                stats.partial_msgs,
                stats.final_msgs,
                stats.queue_dropped,
                stats.queue_depth_peak,
                stats.last_error,
            )

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoxBridge Streaming Web Demo (HTTPS + WebSocket)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--backend", default="vllm", choices=["transformers", "vllm"], help="Inference backend")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8024, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    p.add_argument("--max-model-len", type=int, default=8192, help="vLLM max_model_len")
    p.add_argument("--max-new-tokens", type=int, default=32, help="vLLM max_new_tokens")
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=1024,
        help="vLLM max_num_batched_tokens (lower can reduce startup profiling cost)",
    )

    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=2.0)
    p.add_argument(
        "--min-audio-sec",
        type=float,
        default=1.0,
        help="Minimum buffered audio seconds before first decode in transformers mode",
    )
    p.add_argument(
        "--decode-interval-sec",
        type=float,
        default=2.0,
        help="Decode every N seconds of new audio in transformers mode",
    )
    p.add_argument(
        "--force-language",
        default=None,
        help="Force language for decoding (e.g. Chinese or English). Empty means auto",
    )
    p.add_argument(
        "--enable-translation",
        action="store_true",
        help="Enable real-time translation for recognized text",
    )
    p.add_argument(
        "--translation-backend",
        default="local",
        choices=["local", "openai_api"],
        help="Translation backend: local model or OpenAI-compatible HTTP API",
    )
    p.add_argument(
        "--translation-model-path",
        default="tencent/HY-MT1.5-1.8B",
        help="Local/HF translation model path (used when --translation-backend=local)",
    )
    p.add_argument(
        "--translation-source-language",
        default="Chinese",
        help="Translation source language name used in prompt",
    )
    p.add_argument(
        "--translation-target-language",
        default="英语",
        help="Translation target language name used in prompt",
    )
    p.add_argument(
        "--translation-max-new-tokens",
        type=int,
        default=96,
        help="Translation max generation tokens",
    )
    p.add_argument(
        "--translation-device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for local translation model (used when --translation-backend=local)",
    )
    p.add_argument(
        "--translation-api-base-url",
        default="http://127.0.0.1:8001",
        help="OpenAI-compatible translation API base URL",
    )
    p.add_argument(
        "--translation-api-model",
        default="tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M",
        help="OpenAI-compatible translation model name",
    )
    p.add_argument(
        "--translation-api-key",
        default="",
        help="Bearer token for OpenAI-compatible translation API (optional)",
    )
    p.add_argument(
        "--translation-api-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout seconds for each translation API request",
    )
    p.add_argument(
        "--translation-min-interval-sec",
        type=float,
        default=0.25,
        help="Minimum interval between translation updates for partial text",
    )
    p.add_argument(
        "--translation-min-delta-chars",
        type=int,
        default=6,
        help="Minimum text length delta to trigger translation before interval timeout",
    )
    p.add_argument(
        "--translation-workers",
        type=int,
        default=3,
        help="Concurrent sentence translation workers for committed subtitles",
    )

    p.add_argument("--client-chunk-ms", type=int, default=320, help="Client capture chunk length in milliseconds")
    p.add_argument(
        "--slice-mode",
        default="time",
        choices=["off", "time", "vad"],
        help="Slice strategy: off disables session slicing, time uses fixed interval, vad uses pause-aware slicing",
    )
    p.add_argument(
        "--auto-slice-sec",
        type=float,
        default=0.0,
        help="For time mode: close and restart ASR session every N seconds; for vad mode: hard max session length",
    )
    p.add_argument(
        "--slice-overlap-sec",
        type=float,
        default=1.0,
        help="Replay this many recent seconds into the next auto-sliced session",
    )
    p.add_argument(
        "--vad-silence-sec",
        type=float,
        default=0.6,
        help="In vad mode, trigger slice after this much silence (seconds)",
    )
    p.add_argument(
        "--vad-min-slice-sec",
        type=float,
        default=8.0,
        help="In vad mode, minimum session length before allowing silence-based slice (seconds)",
    )
    p.add_argument(
        "--vad-min-active-sec",
        type=float,
        default=1.2,
        help="In vad mode, minimum detected speech duration before allowing silence-based slice (seconds)",
    )
    p.add_argument(
        "--vad-force-cut-sec",
        type=float,
        default=1.8,
        help="In vad mode, allow cut without sentence boundary after this long silence (seconds)",
    )
    p.add_argument("--idle-timeout-sec", type=int, default=30, help="Close idle websocket after timeout")
    p.add_argument("--max-connections", type=int, default=1, help="Maximum active websocket connections")
    p.add_argument(
        "--audio-queue-size",
        type=int,
        default=32,
        help="Per-connection queue size for decoded audio frames before inference",
    )
    p.add_argument(
        "--consumer-batch-sec",
        type=float,
        default=1.0,
        help="Target seconds of audio merged per backend decode call (higher improves long-session throughput).",
    )
    p.add_argument(
        "--state-rollover-sec",
        type=float,
        default=30.0,
        help="Rotate internal vLLM streaming state every N seconds to keep long-session latency bounded (0 disables).",
    )
    p.add_argument(
        "--segment-hard-cut-sec",
        type=float,
        default=30.0,
        help="Force segment finalize+rotate after this many seconds even without silence.",
    )
    p.add_argument(
        "--segment-overlap-sec",
        type=float,
        default=0.8,
        help="Audio overlap tail carried into next segment after finalize+rotate.",
    )
    p.add_argument(
        "--backpressure-target-queue-sec",
        type=float,
        default=3.0,
        help="Soft queue target in seconds before decode cadence is reduced.",
    )
    p.add_argument(
        "--backpressure-max-queue-sec",
        type=float,
        default=5.0,
        help="Hard queue cap in seconds before oldest frames are dropped.",
    )
    p.add_argument(
        "--backend-vad-enter-snr-db",
        type=float,
        default=8.0,
        help="Backend VAD enter-speech SNR threshold in dB.",
    )
    p.add_argument(
        "--backend-vad-exit-snr-db",
        type=float,
        default=4.0,
        help="Backend VAD exit-speech SNR threshold in dB.",
    )
    p.add_argument(
        "--backend-cut-stable-sec",
        type=float,
        default=0.45,
        help="Text stability time before backend VAD applies a silence cut.",
    )
    p.add_argument(
        "--max-frame-samples",
        type=int,
        default=SAMPLE_RATE * 2,
        help="Maximum samples accepted in a single websocket binary frame",
    )
    p.add_argument(
        "--finalize-on-disconnect",
        action="store_true",
        help="Run finish_streaming_transcribe on unexpected websocket disconnect",
    )
    p.add_argument(
        "--final-redecode-on-stop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="On stop mode, run a one-shot full-audio decode and use it as final canonical text",
    )
    p.add_argument(
        "--final-redecode-max-sec",
        type=float,
        default=180.0,
        help="Maximum buffered audio seconds used for stop-time one-shot re-decode (<=0 means unlimited)",
    )
    p.add_argument(
        "--final-redecode-max-new-tokens",
        type=int,
        default=512,
        help="Temporary max generation tokens used by stop-time one-shot re-decode",
    )
    p.add_argument(
        "--subtitle-trace",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable frontend subtitle trace collection by default",
    )
    p.add_argument(
        "--subtitle-trace-max-events",
        type=int,
        default=1200,
        help="Maximum in-browser subtitle trace events kept in ring buffer",
    )
    p.add_argument(
        "--subtitle-trace-log",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Emit backend structured subtitle trace logs for state transitions",
    )
    p.add_argument(
        "--subtitle-trace-log-partial-every",
        type=int,
        default=20,
        help="Sample interval for backend partial trace events (1 means every partial)",
    )

    p.add_argument("--ssl-certfile", default=None, help="Path to TLS certificate file (enables HTTPS/WSS)")
    p.add_argument("--ssl-keyfile", default=None, help="Path to TLS private key file")
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return p.parse_args()


def main() -> None:
    from qwen_asr import Qwen3ASRModel
    import torch

    global _INSTANCE_LOCK_HANDLE
    args = parse_args()
    try:
        _INSTANCE_LOCK_HANDLE = _acquire_instance_lock_or_raise(args.port)
        _assert_port_bindable(args.host, args.port)
    except RuntimeError as exc:
        _release_instance_lock(_INSTANCE_LOCK_HANDLE)
        _INSTANCE_LOCK_HANDLE = None
        logger.error("startup guard failed: %s", exc)
        raise SystemExit(2) from exc

    try:
        if args.backend == "vllm":
            stale = _cleanup_orphan_enginecore_processes()
            if stale:
                logger.warning("cleaned orphan EngineCore processes before startup: %s", stale)

        if args.backend == "vllm":
            asr = Qwen3ASRModel.LLM(
                model=args.asr_model_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                enforce_eager=True,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            asr = Qwen3ASRModel.from_pretrained(
                args.asr_model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                max_inference_batch_size=1,
                max_new_tokens=64,
            )
        print("Model loaded.")

        translator: Optional[Any] = None
        if bool(getattr(args, "enable_translation", False)):
            translation_backend = str(getattr(args, "translation_backend", "local") or "local").strip().lower()
            if translation_backend == "openai_api":
                logger.info(
                    "loading openai-compatible translator base_url=%s model=%s",
                    args.translation_api_base_url,
                    args.translation_api_model,
                )
                translator = OpenAIAPITranslator(
                    base_url=args.translation_api_base_url,
                    model=args.translation_api_model,
                    source_language=args.translation_source_language,
                    target_language=args.translation_target_language,
                    max_new_tokens=args.translation_max_new_tokens,
                    timeout_sec=args.translation_api_timeout_sec,
                    api_key=args.translation_api_key,
                )
                logger.info("openai-compatible translator loaded.")
            else:
                logger.info(
                    "loading local translator model=%s device=%s",
                    args.translation_model_path,
                    args.translation_device,
                )
                translator = LocalTranslator(
                    model_path=args.translation_model_path,
                    source_language=args.translation_source_language,
                    target_language=args.translation_target_language,
                    max_new_tokens=args.translation_max_new_tokens,
                    device=args.translation_device,
                )
                logger.info("local translator loaded.")

        app = _create_app(args, asr, translator=translator)

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
        )
    finally:
        _release_instance_lock(_INSTANCE_LOCK_HANDLE)
        _INSTANCE_LOCK_HANDLE = None


if __name__ == "__main__":
    main()
