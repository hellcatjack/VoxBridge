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
import base64
import difflib
import fcntl
import html
import hashlib
import hmac
import inspect
import importlib.util
import json
import logging
import os
import signal
import secrets
import socket
import threading
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from voxbridge.streaming.backpressure import QueueBackpressureController
from voxbridge.streaming.context_schedule import (
    ContextSchedule,
    normalize_session_context_terms,
)
from voxbridge.streaming.segment_policy import SegmentPolicy
from voxbridge.streaming.text_pool import dedup_segment_join, trim_prefix_overlap
from voxbridge.tts.broadcast import (
    TTSBroadcastHub,
    TTSBroadcastNotFound,
    TTSBroadcastQueueFull,
)
from voxbridge.tts.jobs import RevisionStableTTSBuffer, TTSJobNotFound, TTSJobRegistry, TTSQueueFull
from voxbridge.tts.listener_page import TTS_LISTENER_HTML
from voxbridge.tts.kokoro_onnx import (
    KokoroOnnxSynthesizer,
    KokoroTTSConfig,
    TTSConfigurationError,
    TTSSynthesisError,
)

SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"[。！？!?…]+[\"'”’)\]）】》]*|\.+[\"'”’)\]）】》]*(?=\s|$|[\u3400-\u9fff])"
)
SENTENCE_CLOSER_CHARS = "\"'”’)]）】》"
INITIALS_ABBREVIATION_PATTERN = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
MIN_CJK_SENTENCE_CHARS = 10
_INSTANCE_LOCK_HANDLE: Optional[Any] = None
AUTH_COOKIE_NAME = "voxbridge_session"
AUTH_HASH_SCHEME = "pbkdf2_sha256"
AUTH_HASH_ITERATIONS = 260_000
AUTH_SESSION_TOKEN_BYTES = 32
TTS_CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def _vllm_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enforce_eager": True,
        "max_new_tokens": int(args.max_new_tokens),
        "mm_processor_cache_gb": float(args.mm_processor_cache_gb),
    }


def _opaque_identifier_hash8(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:8]


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    value = str(text or "").strip()
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _hash_auth_password(password: str, *, salt: Optional[bytes] = None, iterations: int = AUTH_HASH_ITERATIONS) -> str:
    raw_password = str(password or "")
    if not raw_password:
        raise ValueError("password must not be empty")
    rounds = max(1, int(iterations))
    raw_salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", raw_password.encode("utf-8"), raw_salt, rounds)
    return f"{AUTH_HASH_SCHEME}${rounds}${_b64url_encode(raw_salt)}${_b64url_encode(digest)}"


def _parse_auth_password_hash(encoded_hash: str) -> Tuple[int, bytes, bytes]:
    parts = str(encoded_hash or "").strip().split("$")
    if len(parts) != 4 or parts[0] != AUTH_HASH_SCHEME:
        raise ValueError("unsupported password hash format")
    try:
        iterations = int(parts[1])
    except ValueError as exc:
        raise ValueError("invalid password hash iterations") from exc
    if iterations <= 0:
        raise ValueError("invalid password hash iterations")
    try:
        salt = _b64url_decode(parts[2])
        digest = _b64url_decode(parts[3])
    except Exception as exc:
        raise ValueError("invalid password hash encoding") from exc
    if not salt or not digest:
        raise ValueError("invalid password hash payload")
    return iterations, salt, digest


def _verify_auth_password(password: str, encoded_hash: str) -> bool:
    try:
        iterations, salt, expected = _parse_auth_password_hash(encoded_hash)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", str(password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def _build_auth_config(args: argparse.Namespace) -> SimpleNamespace:
    enabled = bool(getattr(args, "auth_enabled", False))
    username = str(getattr(args, "auth_username", "admin") or "admin").strip() or "admin"
    password_hash = str(
        getattr(args, "auth_password_hash", "") or os.environ.get("VOXBRIDGE_AUTH_PASSWORD_HASH", "")
    ).strip()
    if enabled:
        if not password_hash:
            raise RuntimeError(
                "auth is enabled but no password hash was provided; set --auth-password-hash "
                "or VOXBRIDGE_AUTH_PASSWORD_HASH"
            )
        try:
            _parse_auth_password_hash(password_hash)
        except ValueError as exc:
            raise RuntimeError(f"invalid auth password hash: {exc}") from exc
    return SimpleNamespace(
        enabled=enabled,
        username=username,
        password_hash=password_hash,
        cookie_secure=bool(getattr(args, "auth_cookie_secure", False)),
        session_ttl_sec=max(60, int(getattr(args, "auth_session_ttl_sec", 12 * 60 * 60))),
        disable_debug_file=bool(getattr(args, "disable_debug_file", False)),
    )


def _load_asr_context_schedule(args: argparse.Namespace) -> Optional[ContextSchedule]:
    cached = getattr(args, "_asr_context_schedule_obj", None)
    if isinstance(cached, ContextSchedule):
        return cached
    schedule_path = str(getattr(args, "asr_context_schedule", "") or "").strip()
    if not schedule_path:
        return None
    schedule = ContextSchedule.from_path(Path(schedule_path).expanduser())
    setattr(args, "_asr_context_schedule_obj", schedule)
    return schedule


def _normalize_asr_context_apply_mode(value: Any) -> str:
    mode = str(value or "segment_final").strip().lower().replace("-", "_")
    if mode not in {"segment_final", "streaming"}:
        raise ValueError(f"unsupported ASR context apply mode: {value!r}")
    return mode


def _compact_asr_compare_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE).casefold()


def _consume_unmatched_compact_occurrence(
    existing: str,
    candidate: str,
    used_spans: List[Tuple[int, int]],
) -> bool:
    if not existing or not candidate:
        return False
    search_from = 0
    while search_from < len(existing):
        start = existing.find(candidate, search_from)
        if start < 0:
            return False
        end = start + len(candidate)
        if all(end <= used_start or start >= used_end for used_start, used_end in used_spans):
            used_spans.append((start, end))
            return True
        search_from = start + 1
    return False


def _should_accept_context_sentence_correction(old_text: str, new_text: str) -> bool:
    old_compact = _compact_asr_compare_text(old_text)
    new_compact = _compact_asr_compare_text(new_text)
    if not old_compact or not new_compact or old_compact == new_compact:
        return False
    length_ratio = len(new_compact) / float(len(old_compact))
    if length_ratio < 0.65 or length_ratio > 1.45:
        return False
    return difflib.SequenceMatcher(None, old_compact, new_compact).ratio() >= 0.72


def _looks_like_asr_context_echo(context: str, text: str, previous_text: str = "") -> bool:
    terms = [part for part in str(context or "").split() if part]
    if len(terms) < 3:
        return False
    context_compact = _compact_asr_compare_text(context)
    text_compact = _compact_asr_compare_text(text)
    if len(context_compact) < 6 or not text_compact:
        return False
    length_ratio = len(text_compact) / float(len(context_compact))
    if length_ratio < 0.8 or length_ratio > 1.25:
        return False
    if difflib.SequenceMatcher(None, context_compact, text_compact).ratio() < 0.9:
        return False

    previous_compact = _compact_asr_compare_text(previous_text)
    if previous_compact:
        evidence_ratio = len(previous_compact) / float(len(text_compact))
        evidence_similarity = difflib.SequenceMatcher(
            None,
            previous_compact,
            text_compact,
        ).ratio()
        if 0.6 <= evidence_ratio <= 1.6 and evidence_similarity >= 0.55:
            return False
    return True


def _filter_asr_context_echo_sentences(
    context: str,
    text: str,
    previous_text: str = "",
) -> Tuple[str, int]:
    source = str(text or "").strip()
    if not source or not str(context or "").strip():
        return source, 0

    completed, tail = _split_sentences_and_tail(source)
    candidates = list(completed)
    if tail:
        candidates.append(str(tail or "").strip())
    previous_completed, previous_tail = _split_sentences_and_tail(previous_text)
    previous_candidates = [str(item or "").strip() for item in previous_completed]
    if previous_tail:
        previous_candidates.append(str(previous_tail or "").strip())

    kept: List[str] = []
    removed = 0
    for candidate in candidates:
        current = str(candidate or "").strip()
        if not current:
            continue
        current_compact = _compact_asr_compare_text(current)
        previous_evidence = max(
            previous_candidates,
            key=lambda item: difflib.SequenceMatcher(
                None,
                _compact_asr_compare_text(item),
                current_compact,
            ).ratio(),
            default="",
        )
        if _looks_like_asr_context_echo(
            context,
            current,
            previous_text=previous_evidence,
        ):
            removed += 1
            continue
        kept.append(current)
    return _join_segments(kept), int(removed)


def _safe_exception_trace_fields(exc: BaseException) -> Dict[str, str]:
    error_text = str(exc or "").encode("utf-8", errors="replace")
    return {
        "error_type": type(exc).__name__,
        "error_sha256": hashlib.sha256(error_text).hexdigest(),
    }


async def _await_thread_completion_on_cancel(func: Any, *args: Any, **kwargs: Any) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(worker)
        except Exception:
            pass
        raise


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


_ESV_ZH_TO_EN_POLICY = (
    "忠实原文是最高优先级。涉及基督教、圣经或神学内容时，仅对原文中明确出现的"
    "卷名、人名、地名、称谓、神学术语及大小写，必须采用 English Standard Version "
    "(ESV) 的标准英文用法；只有原文完整且语义明确等同于某段经文时，措辞才应尽量"
    "贴近 ESV；若原文是节选、转述、误引或无法确定对应经文时，必须按演讲者实际说法"
    "忠实翻译；不得补写、扩写、解释、纠正或用记忆中的经文替换原文"
)


def _build_translation_prompt(
    text: str,
    source_language: str,
    target_language: str,
    translation_direction: Optional[str] = None,
) -> str:
    source = str(source_language or "Chinese")
    target = str(target_language or "English")
    direction = str(translation_direction or "").strip().lower()
    if direction in {"zh2en", "zh->en", "chinese->english", "中文->英文"}:
        use_esv_policy = True
    elif direction in {"en2zh", "en->zh", "english->chinese", "英文->中文"}:
        use_esv_policy = False
    else:
        use_esv_policy = _is_chinese_label(source) and _is_english_label(target)

    requirements = ["忠实原文，不增删"]
    if use_esv_policy:
        requirements.append(_ESV_ZH_TO_EN_POLICY)
    else:
        requirements.append("保留专有名词")
    requirements.append("只输出译文本身，不要解释")
    return (
        f"请将以下{source}文本翻译为{target}。\n"
        f"要求：{'；'.join(requirements)}。\n\n"
        f"原文：\n{text}"
    )


def _split_sentences_and_tail(text: str) -> Tuple[List[str], str]:
    src = str(text or "").strip()
    if not src:
        return [], ""

    raw_sentences: List[str] = []
    last = 0
    for _, end, _ in _iter_sentence_boundaries(src, SENTENCE_BOUNDARY_PATTERN):
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


def _terminal_core(token: str) -> str:
    src = str(token or "")
    for ch in src:
        if ch in ".。":
            return "."
        if ch in "?？":
            return "?"
        if ch in "!！":
            return "!"
        if ch == "…":
            return "…"
    return ""


def _trim_leading_boundary_overlap(
    prev_sentence: str,
    candidate_sentence: str,
    *,
    max_cjk_chars: int = 4,
) -> str:
    prev = str(prev_sentence or "").strip()
    candidate = str(candidate_sentence or "").strip()
    if not prev or not candidate or not _has_cjk(prev) or not _has_cjk(candidate):
        return candidate

    terminal_match = re.search(r"([。！？!?…]+[\"'”’)\]）】》]*)$", prev)
    if not terminal_match:
        return candidate
    prev_terminal = _terminal_core(str(terminal_match.group(1) or ""))
    if not prev_terminal:
        return candidate

    prev_body = prev[: terminal_match.start()].strip()
    if not prev_body:
        return candidate

    max_len = min(int(max(1, max_cjk_chars)), len(prev_body), len(candidate))
    for size in range(max_len, 0, -1):
        suffix = prev_body[-size:]
        if not suffix or not all(_has_cjk(ch) for ch in suffix):
            continue
        if not candidate.startswith(suffix):
            continue
        rest = candidate[size:]
        replay_terminal = re.match(r"[。！？!?…]+[\"'”’)\]）】》]*", rest)
        if not replay_terminal:
            continue
        if _terminal_core(str(replay_terminal.group(0) or "")) != prev_terminal:
            continue
        trimmed = rest[int(replay_terminal.end()) :].strip()
        if trimmed:
            return trimmed
    return candidate


def _text_ends_with_sentence_terminator(text: str) -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    src = re.sub(r"[\"'”’)\]）】》\s]+$", "", src).strip()
    return bool(re.search(r"[。！？!?….]$", src))


def _should_hard_cut_fallback_merge(pending_prefix: str, raw_text: str) -> bool:
    pending = str(pending_prefix or "").strip()
    raw = str(raw_text or "").strip()
    if not pending or not raw:
        return False
    # Completed carry text should not be glued to unrelated next-segment text
    # when no reliable overlap exists.
    if _text_ends_with_sentence_terminator(pending):
        return False
    return True


def _is_abbreviation_period_boundary(text: str, start: int, end: int) -> bool:
    src = str(text or "")
    if not src:
        return False
    if start < 0 or end <= start or end > len(src):
        return False
    token = src[start:end]
    if "." not in token:
        return False

    trimmed_end = int(end)
    while trimmed_end > start and src[trimmed_end - 1] in SENTENCE_CLOSER_CHARS:
        trimmed_end -= 1
    if trimmed_end <= start or src[trimmed_end - 1] != ".":
        return False

    suffix = src[end:].lstrip()
    if not suffix:
        return False

    prev_char = src[trimmed_end - 2] if trimmed_end >= 2 else ""
    next_char = suffix[0] if suffix else ""
    if prev_char.isdigit() and next_char.isdigit():
        return True

    left_tail = src[max(0, trimmed_end - 40):trimmed_end]
    if INITIALS_ABBREVIATION_PATTERN.search(left_tail):
        return True

    token_match = re.search(r"([A-Za-z]+)$", src[: max(0, trimmed_end - 1)])
    token = token_match.group(1) if token_match else ""
    if token and len(token) <= 2 and token[:1].isupper():
        return True
    return False


def _iter_sentence_boundaries(text: str, boundary_pattern: Any):
    src = str(text or "")
    if not src:
        return
    for match in boundary_pattern.finditer(src):
        start = int(match.start())
        end = int(match.end())
        token = str(match.group(0) or "")
        if boundary_pattern is SENTENCE_BOUNDARY_PATTERN and _is_abbreviation_period_boundary(src, start, end):
            continue
        yield start, end, token


def _find_first_boundary_after(
    text: str,
    start_chars: int,
    boundary_pattern: Any,
) -> Optional[Tuple[int, str]]:
    src = str(text or "")
    if not src:
        return None
    start = max(0, min(len(src), int(start_chars)))
    latest: Optional[Tuple[int, str]] = None
    for _, end, token in _iter_sentence_boundaries(src, boundary_pattern):
        if end <= start:
            continue
        latest = (int(end), str(token or ""))
    return latest


def _resolve_boundary_for_anchor(
    text: str,
    anchor_end_chars: int,
    boundary_pattern: Any,
) -> Optional[Tuple[int, str]]:
    src = str(text or "")
    if not src:
        return None
    target = max(1, min(len(src), int(anchor_end_chars)))
    before: Optional[Tuple[int, str]] = None
    for _, end, token in _iter_sentence_boundaries(src, boundary_pattern):
        if end >= target:
            return end, token
        before = (end, token)
    return before


def _split_text_at_boundary(text: str, boundary_end_chars: int) -> Tuple[str, str]:
    src = str(text or "")
    if not src:
        return "", ""
    cut = max(0, min(len(src), int(boundary_end_chars)))
    if cut <= 0:
        return "", src.strip()
    left = src[:cut].strip()
    right = src[cut:].strip()
    return left, right


def _normalize_sentence_for_duplicate_compare(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    return re.sub(r"\s+", " ", src)


def _sentence_text_quality_score(text: str) -> int:
    src = str(text or "").strip()
    if not src:
        return 0
    score = 0
    if bool(re.search(r"[。！？!?…]+[\"'”’)\]）】》]*$", src)):
        score += 2
    mid_punct = re.findall(r"(?<=[\u3400-\u9fffA-Za-z0-9])[。！？!?…](?=[\u3400-\u9fffA-Za-z0-9])", src)
    if mid_punct:
        score -= 2 * int(len(mid_punct))
    if bool(re.search(r"[。！？!?…]{2,}", src)):
        score -= 1
    return int(score)


def _canonical_sentence_for_upgrade_compare(text: str) -> str:
    src = _normalize_sentence_for_duplicate_compare(text)
    if not src:
        return ""
    src = re.sub(r"(?<=[\u3400-\u9fffA-Za-z0-9])[。！？!?…]+(?=[\u3400-\u9fffA-Za-z0-9])", "", src)
    src = re.sub(r"[\"'“”‘’`]", "", src)
    src = re.sub(r"\s+", "", src)
    return src.strip()


_SMALL_UPGRADE_REQUIRED_HITS = 3
_SMALL_UPGRADE_STABLE_SEC = 0.6


@dataclass
class _DeferredSentenceUpgrade:
    text: str
    first_seen_at: float
    last_seen_at: float
    hits: int
    last_seq: int


@dataclass(frozen=True)
class _DeferredSentenceUpgradeObservation:
    transition: str
    ready: bool
    hits: int
    stable_ms: int
    text: str
    previous_text: str = ""


def _terminal_growth_base(text: str) -> str:
    src = str(text or "").strip()
    src = re.sub(r"[。！？!?….,，;；:：]+[\"'”’)\]）】》]*$", "", src).strip()
    return _canonical_sentence_for_upgrade_compare(src).casefold()


def _is_monotonic_sentence_extension(old_text: str, new_text: str) -> bool:
    old = str(old_text or "").strip()
    new = str(new_text or "").strip()
    if not old or not new or old == new:
        return False
    old_base = _terminal_growth_base(old)
    new_base = _terminal_growth_base(new)
    return bool(
        old_base
        and new_base
        and len(new_base) > len(old_base)
        and new_base.startswith(old_base)
        and _sentence_text_quality_score(new) >= _sentence_text_quality_score(old)
    )


def _observe_deferred_sentence_upgrade(
    candidates: Dict[str, _DeferredSentenceUpgrade],
    sentence_id: str,
    text: str,
    seq: int,
    now: float,
    required_hits: int = _SMALL_UPGRADE_REQUIRED_HITS,
    required_stable_sec: float = _SMALL_UPGRADE_STABLE_SEC,
) -> _DeferredSentenceUpgradeObservation:
    sid = str(sentence_id or "")
    candidate_text = str(text or "").strip()
    previous = candidates.get(sid)
    transition = "waiting"
    previous_text = ""
    if previous is None or previous.text != candidate_text:
        previous_text = str(previous.text if previous is not None else "")
        transition = "changed" if previous is not None else "started"
        current = _DeferredSentenceUpgrade(candidate_text, now, now, 1, int(seq))
        candidates[sid] = current
    else:
        previous.hits += 1
        previous.last_seen_at = float(now)
        previous.last_seq = int(seq)
        current = previous
    stable_ms = max(0, int(round((float(now) - current.first_seen_at) * 1000.0)))
    ready = bool(
        current.hits >= max(1, int(required_hits))
        and stable_ms >= max(0, int(round(float(required_stable_sec) * 1000.0)))
    )
    if ready:
        transition = "accepted"
    return _DeferredSentenceUpgradeObservation(
        transition=transition,
        ready=ready,
        hits=int(current.hits),
        stable_ms=int(stable_ms),
        text=current.text,
        previous_text=previous_text,
    )


def _should_accept_sentence_upgrade(old_text: str, new_text: str) -> bool:
    old = str(old_text or "").strip()
    new = str(new_text or "").strip()
    if not old or not new or old == new:
        return False

    # Primary path: confidence growth from streaming partial -> more complete sentence.
    if len(new) >= len(old) + 8:
        if new.startswith(old):
            return True
        if old in new and len(new) >= len(old) + 12:
            return True
        # Normalize common terminal punctuation so sentence growth like
        # "...as well." -> "...as well, and ..." is treated as an upgrade.
        old_base = re.sub(r"[。！？!?….,，;；:：]+[\"'”’)\]）】》]*$", "", old).strip()
        new_base = re.sub(r"[。！？!?….,，;；:：]+[\"'”’)\]）】》]*$", "", new).strip()
        if old_base and new_base and len(new_base) >= len(old_base) + 6:
            if new_base.startswith(old_base):
                return True
            if old_base in new_base and len(new_base) >= len(old_base) + 10:
                return True
            old_growth = _canonical_sentence_for_upgrade_compare(old_base).casefold()
            new_growth = _canonical_sentence_for_upgrade_compare(new_base).casefold()
            if old_growth and len(new_growth) >= len(old_growth) + 6:
                common_prefix_chars = 0
                for old_char, new_char in zip(old_growth, new_growth):
                    if old_char != new_char:
                        break
                    common_prefix_chars += 1
                correction_budget = max(2, min(16, (len(old_growth) + 7) // 8))
                required_prefix_chars = max(8, len(old_growth) - correction_budget)
                if common_prefix_chars >= required_prefix_chars:
                    return True

    # Secondary path: allow minor corrections if text quality improves
    # (for example removing hallucinated terminal punctuation inside a word).
    old_quality = _sentence_text_quality_score(old)
    new_quality = _sentence_text_quality_score(new)
    old_canon = _canonical_sentence_for_upgrade_compare(old)
    new_canon = _canonical_sentence_for_upgrade_compare(new)
    if old_canon and new_canon and old_canon == new_canon:
        if new_quality > old_quality:
            return True
        if new_quality == old_quality and len(new) >= len(old) + 3:
            return True
        return False

    length_delta = abs(len(new) - len(old))
    if length_delta <= 12:
        ratio = difflib.SequenceMatcher(None, old, new).ratio()
        if ratio >= 0.88 and new_quality > old_quality:
            return True

    return False


def _english_word_count(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    return len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+", src))


def _is_short_english_sentence_for_early_commit(
    text: str,
    *,
    min_words: int = 6,
    min_chars: int = 32,
) -> bool:
    src = str(text or "").strip()
    if not src or _has_cjk(src) or not _has_latin(src):
        return False
    words = _english_word_count(src)
    chars = len(src)
    return words < int(max(1, min_words)) and chars < int(max(1, min_chars))


def _is_short_english_slice_fragment(
    text: str,
    *,
    min_words: int = 6,
    min_chars: int = 32,
) -> bool:
    src = str(text or "").strip()
    if not src or _has_cjk(src) or not _has_latin(src):
        return False
    # Periods in very short English partials are often ASR boundary guesses
    # rather than reliable sentence endings. Keep questions and exclamations
    # eligible because they are stronger end-of-sentence signals.
    if not re.search(r"\.[\"'”’)\]）】》]*$", src):
        return False
    return _is_short_english_sentence_for_early_commit(
        src,
        min_words=int(min_words),
        min_chars=int(min_chars),
    )


def _strip_short_english_fragment_period(
    text: str,
    *,
    min_words: int = 6,
    min_chars: int = 32,
) -> str:
    src = str(text or "").strip()
    if not _is_short_english_slice_fragment(
        src,
        min_words=int(min_words),
        min_chars=int(min_chars),
    ):
        return src
    return re.sub(r"\.[\"'”’)\]）】》]*$", "", src).strip()


def _trace_preview(text: str, max_chars: int = 72) -> str:
    src = str(text or "").strip()
    if not src or max_chars <= 0:
        return ""
    if len(src) <= max_chars:
        return src
    if max_chars <= 12:
        return src[:max_chars]
    head = max(4, max_chars // 2 - 2)
    tail = max(4, max_chars - head - 3)
    return f"{src[:head]}...{src[-tail:]}"


def _is_probable_pending_prefix_duplicate(
    prev_sentence: str,
    candidate_sentence: str,
    raw_full_text: str,
    pending_prefix_text: str,
) -> bool:
    prev = _normalize_sentence_for_duplicate_compare(prev_sentence)
    candidate = _normalize_sentence_for_duplicate_compare(candidate_sentence)
    raw = _normalize_sentence_for_duplicate_compare(raw_full_text)
    pending = _normalize_sentence_for_duplicate_compare(pending_prefix_text)
    if not prev or not candidate or not pending:
        return False
    if prev != candidate:
        return False
    # Be conservative: prefer keeping possible real repetitions.
    # Only treat as carry-over duplicate when pending has clear extra tail
    # after candidate, and current raw text is only a tiny prefix.
    if len(candidate) < 4:
        return False
    if not pending.startswith(candidate):
        return False
    pending_tail = pending[len(candidate):].strip()
    if len(pending_tail) < 2:
        return False
    raw_limit = max(6, int(len(candidate) * 0.35))
    if len(raw) > raw_limit:
        return False
    if raw and not pending.startswith(raw):
        return False
    return bool(raw)


def _should_apply_carry_overlap_skip(*, overlap_count: int, overlap_chars: int, raw_chars: int) -> bool:
    count = max(0, int(overlap_count))
    chars = max(0, int(overlap_chars))
    raw = max(0, int(raw_chars))
    if count < 2 or chars <= 0:
        return False
    # Require strong evidence to skip commit; single-sentence overlap is often
    # a legitimate repetition in fast speech scenarios.
    raw_limit = max(12, int(float(chars) * 0.35))
    raw_limit = min(raw_limit, 24)
    return raw <= raw_limit


def _trim_pending_prefix_leading_sentence(
    pending_prefix_text: str,
    sentence_text: str,
) -> Tuple[str, bool]:
    pending = str(pending_prefix_text or "").strip()
    sentence = str(sentence_text or "").strip()
    if not pending or not sentence:
        return pending, False
    completed, tail = _split_sentences_and_tail(pending)
    if not completed:
        return pending, False
    first = str(completed[0] or "").strip()
    if _normalize_sentence_for_duplicate_compare(first) != _normalize_sentence_for_duplicate_compare(sentence):
        return pending, False
    rest = [str(seg or "").strip() for seg in completed[1:]]
    if tail:
        rest.append(str(tail or "").strip())
    return _join_segments(rest), True


def _alignment_sentence_key(text: str) -> str:
    return _normalize_sentence_for_duplicate_compare(text)


def _alignment_registry_touch(
    registry: Dict[str, Dict[str, Any]],
    sentence_text: str,
    seq_hint: int,
    source: str,
) -> Tuple[str, bool, Dict[str, Any]]:
    text = str(sentence_text or "").strip()
    key = _alignment_sentence_key(text)
    if not key:
        return "", False, {}

    seq_no = int(seq_hint or 0)
    src = str(source or "")
    entry = registry.get(key)
    if entry is None:
        entry = {
            "text": text,
            "hits": 1,
            "first_seq": int(seq_no),
            "last_seq": int(seq_no),
            "sources": [src] if src else [],
        }
        registry[key] = entry
        return key, True, entry

    entry["hits"] = int(entry.get("hits", 0) or 0) + 1
    entry["last_seq"] = max(int(entry.get("last_seq", seq_no) or seq_no), int(seq_no))
    if len(text) > len(str(entry.get("text", "") or "")):
        entry["text"] = text
    sources = entry.get("sources")
    if not isinstance(sources, list):
        sources = []
        entry["sources"] = sources
    if src and src not in sources and len(sources) < 8:
        sources.append(src)
    return key, False, entry


def _alignment_registry_touch_model(
    registry: Dict[str, Dict[str, Any]],
    sentence_text: str,
    seq_hint: int,
    source: str,
) -> Tuple[str, bool, Dict[str, Any]]:
    def _base(src: str) -> str:
        return re.sub(r"[。！？!?…]+[\"'”’)\]）】》]*$", "", str(src or "").strip()).strip()

    key, created, entry = _alignment_registry_touch(registry, sentence_text, seq_hint, source)
    if not key:
        return "", False, {}
    if not created:
        return key, False, entry

    seq_no = int(seq_hint or 0)
    src = str(source or "")
    key_base = _base(key)
    # Collapse incremental growth into the same logical sentence key.
    best_prefix_key = ""
    for old_key in list(registry.keys()):
        if old_key == key:
            continue
        old_base = _base(old_key)
        if key_base.startswith(old_base) and len(key_base) >= len(old_base) + 4:
            if len(old_key) > len(best_prefix_key):
                best_prefix_key = old_key
    if best_prefix_key:
        old_entry = registry.pop(best_prefix_key, {})
        merged_hits = int(old_entry.get("hits", 0) or 0) + int(entry.get("hits", 0) or 0)
        sources = list(old_entry.get("sources", []) or [])
        if src and src not in sources and len(sources) < 8:
            sources.append(src)
        first_seq = int(old_entry.get("first_seq", seq_no) or seq_no)
        registry[key] = {
            "text": str(sentence_text or "").strip(),
            "hits": int(merged_hits),
            "first_seq": int(min(first_seq, seq_no)),
            "last_seq": int(max(int(old_entry.get("last_seq", seq_no) or seq_no), seq_no)),
            "sources": sources,
        }
        return key, False, registry[key]

    # Short regression variant: attach hit to the longer existing sentence.
    best_longer_key = ""
    for old_key in list(registry.keys()):
        if old_key == key:
            continue
        old_base = _base(old_key)
        if old_base.startswith(key_base) and len(old_base) >= len(key_base) + 4:
            if len(old_key) > len(best_longer_key):
                best_longer_key = old_key
    if best_longer_key:
        old_entry = registry.get(best_longer_key, {})
        old_entry["hits"] = int(old_entry.get("hits", 0) or 0) + int(entry.get("hits", 0) or 0)
        old_entry["last_seq"] = int(max(int(old_entry.get("last_seq", seq_no) or seq_no), seq_no))
        sources = old_entry.get("sources")
        if not isinstance(sources, list):
            sources = []
            old_entry["sources"] = sources
        if src and src not in sources and len(sources) < 8:
            sources.append(src)
        registry.pop(key, None)
        return best_longer_key, False, old_entry

    return key, True, entry


def _summarize_alignment_gap(
    model_seen: Dict[str, Dict[str, Any]],
    committed_seen: Dict[str, Dict[str, Any]],
    *,
    min_model_hits: int = 2,
    max_samples: int = 6,
) -> Dict[str, Any]:
    model_stable_keys: List[str] = []
    model_final_keys: List[str] = []
    for key, row in model_seen.items():
        hits = int(row.get("hits", 0) or 0)
        sources = row.get("sources")
        has_final_source = isinstance(sources, list) and ("final_raw" in sources)
        if bool(has_final_source):
            model_final_keys.append(key)
        if hits >= int(max(1, min_model_hits)) or bool(has_final_source):
            model_stable_keys.append(key)

    committed_keys = set(committed_seen.keys())
    missing_keys = [k for k in model_stable_keys if k not in committed_keys]
    missing_keys.sort(key=lambda k: int(model_seen.get(k, {}).get("first_seq", 0) or 0))
    final_missing_keys = [k for k in model_final_keys if k not in committed_keys]
    final_missing_keys.sort(key=lambda k: int(model_seen.get(k, {}).get("first_seq", 0) or 0))

    samples: List[Dict[str, Any]] = []
    for key in missing_keys[: max(1, int(max_samples))]:
        row = model_seen.get(key, {})
        samples.append(
            {
                "text": str(row.get("text", "") or ""),
                "hits": int(row.get("hits", 0) or 0),
                "first_seq": int(row.get("first_seq", 0) or 0),
                "last_seq": int(row.get("last_seq", 0) or 0),
                "sources": list(row.get("sources", []) or []),
            }
        )

    final_samples: List[Dict[str, Any]] = []
    for key in final_missing_keys[: max(1, int(max_samples))]:
        row = model_seen.get(key, {})
        final_samples.append(
            {
                "text": str(row.get("text", "") or ""),
                "hits": int(row.get("hits", 0) or 0),
                "first_seq": int(row.get("first_seq", 0) or 0),
                "last_seq": int(row.get("last_seq", 0) or 0),
                "sources": list(row.get("sources", []) or []),
            }
        )

    return {
        "model_all_unique": int(len(model_seen)),
        "model_stable_unique": int(len(model_stable_keys)),
        "model_final_unique": int(len(model_final_keys)),
        "committed_unique": int(len(committed_seen)),
        "missing_unique": int(len(missing_keys)),
        "missing_samples": samples,
        "final_missing_unique": int(len(final_missing_keys)),
        "final_missing_samples": final_samples,
    }


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


def _classify_boundary_join_mode(prev_text: str, new_text: str, merged_text: str) -> str:
    prev = str(prev_text or "").strip()
    nxt = str(new_text or "").strip()
    merged = str(merged_text or "").strip()
    if len(merged) < len(prev) + len(nxt):
        return "overlap"
    if merged == f"{prev} {nxt}":
        return "spaced"
    return "direct"


def _stabilize_completed_prefix_with_committed(
    completed: List[str],
    committed_sentences: List[str],
    *,
    commit_base: int,
    committed_count: int,
) -> Tuple[List[str], int]:
    merged = [str(seg or "").strip() for seg in list(completed or [])]
    total_committed = len(committed_sentences)
    base = max(0, min(int(commit_base), total_committed))
    expected = max(0, min(int(committed_count), total_committed - base))
    if expected <= len(merged):
        return merged, 0
    backfilled = 0
    for idx in range(len(merged), expected):
        global_idx = int(base + idx)
        if global_idx < total_committed:
            merged.append(str(committed_sentences[global_idx] or "").strip())
            backfilled += 1
    return merged, int(backfilled)


def _count_leading_completed_committed_overlap(
    completed: List[str],
    committed_sentences: List[str],
    *,
    max_overlap: int = 6,
) -> int:
    if not completed or not committed_sentences:
        return 0
    total_completed = len(completed)
    total_committed = len(committed_sentences)
    limit = max(0, min(int(max_overlap), total_completed, total_committed))
    if limit <= 0:
        return 0
    for width in range(limit, 0, -1):
        ok = True
        for i in range(width):
            left = _normalize_sentence_for_duplicate_compare(str(completed[i] or ""))
            right = _normalize_sentence_for_duplicate_compare(str(committed_sentences[total_committed - width + i] or ""))
            if not left or not right or left != right:
                ok = False
                break
        if ok:
            return int(width)
    return 0


def _should_skip_stream_decode(
    *,
    in_speech: bool,
    silence_ms: float,
    segment_elapsed_ms: float,
    snr_db: float,
    vad_silence_ms: float,
    vad_exit_snr_db: float,
    has_pending_text: bool,
) -> bool:
    _ = has_pending_text  # Intentional: silence gating no longer depends on pending text.
    if float(snr_db) > float(vad_exit_snr_db):
        return False
    # For active segments, skip decode quickly once trailing silence is observed.
    if bool(in_speech):
        return float(silence_ms) >= 80.0
    # For not-yet-active speech, require a short quiet window to avoid overreacting
    # to tiny startup jitter while still saving compute on silent input.
    silence_gate_ms = max(80.0, min(float(vad_silence_ms), 200.0))
    quiet_window_ms = max(float(silence_ms), float(segment_elapsed_ms))
    return quiet_window_ms >= silence_gate_ms


def _should_use_high_batch_merge(*, queue_depth: int, audio_queue_size: int, under_pressure: bool) -> bool:
    if bool(under_pressure):
        return True
    return int(queue_depth) >= max(4, int(audio_queue_size) // 2)


def _should_hold_partial_reset(
    *,
    prev_text: str,
    next_text: str,
    min_prev_chars: int = 20,
    max_next_ratio: float = 0.6,
) -> bool:
    prev = str(prev_text or "").strip()
    nxt = str(next_text or "").strip()
    if not prev or not nxt:
        return False
    if nxt.startswith(prev):
        return False
    if len(prev) < int(max(4, min_prev_chars)):
        return False
    next_cap = max(6, int(float(len(prev)) * float(max_next_ratio)))
    if len(nxt) > next_cap:
        return False
    n = min(len(prev), len(nxt))
    i = 0
    while i < n and prev[i] == nxt[i]:
        i += 1
    # Keep genuinely incremental updates; hold abrupt rewrites only.
    if i >= max(8, int(len(prev) * 0.65)):
        return False
    return True


def _should_release_partial_reset_guard(
    *,
    candidate_hits: int,
    hold_sec: float,
    min_hits: int = 2,
    max_hold_sec: float = 1.2,
) -> bool:
    if int(candidate_hits) >= int(max(1, min_hits)):
        return True
    return float(hold_sec) >= float(max(0.1, max_hold_sec))


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

    def _build_prompt(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        translation_direction: Optional[str] = None,
    ) -> str:
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        return _build_translation_prompt(text, source, target, translation_direction)

    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        translation_direction: Optional[str] = None,
    ) -> str:
        import torch

        src = str(text or "").strip()
        if not src:
            return ""
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        if not _text_matches_source_language(src, source):
            return ""

        messages = [
            {
                "role": "user",
                "content": self._build_prompt(
                    src,
                    source_language=source,
                    target_language=target,
                    translation_direction=translation_direction,
                ),
            }
        ]
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

    def _build_prompt(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        translation_direction: Optional[str] = None,
    ) -> str:
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        return _build_translation_prompt(text, source, target, translation_direction)

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

    def _extract_finish_reason(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return str(choices[0].get("finish_reason") or "").strip().lower()
        return ""

    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        translation_direction: Optional[str] = None,
    ) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        source = str(source_language or self.source_language or "Chinese")
        target = str(target_language or self.target_language or "English")
        if not _text_matches_source_language(src, source):
            return ""

        messages = [
            {
                "role": "user",
                "content": self._build_prompt(
                    src,
                    source_language=source,
                    target_language=target,
                    translation_direction=translation_direction,
                ),
            }
        ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_tokens = int(self.max_new_tokens)
        retry_limit = max(max_tokens, min(512, max(128, max_tokens * 4)))
        last_out = ""
        while True:
            body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
            }
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(self.chat_url, data=data, headers=headers, method="POST")
            with self._lock:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            out = self._extract_content(payload)
            out = out.replace("<target>", "").replace("</target>", "").strip()
            finish_reason = self._extract_finish_reason(payload)
            if out:
                last_out = out
            if finish_reason != "length":
                return out
            next_tokens = min(int(retry_limit), max(max_tokens * 2, 128))
            if next_tokens <= max_tokens:
                return out or last_out
            max_tokens = next_tokens


INDEX_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>语音识别与翻译</title>
  <style>
    :root{
      --bg-a:#edf4ea;
      --bg-b:#f6f0e6;
      --ink:#243126;
      --muted:#657267;
      --line:#ccd8c7;
      --surface:#f8fbf4;
      --surface-soft:rgba(255, 255, 247, 0.76);
      --surface-strong:#eef5ec;
      --ok:#2f7d55;
      --warn:#9b6b23;
      --err:#b5564b;
      --accent:#527e72;
      --accent-strong:#426c61;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body{
      margin:0;
      font-family: "Avenir Next", "Segoe UI", "Noto Sans SC", "PingFang SC", sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at 16% 14%, rgba(178, 204, 164, 0.36) 0%, transparent 34%),
        radial-gradient(circle at 84% 82%, rgba(225, 203, 165, 0.38) 0%, transparent 36%),
        linear-gradient(160deg, var(--bg-a), var(--bg-b));
    }

    .wrap{
      height: 100%;
      height: 100svh;
      padding: 0;
      display: grid;
      place-items: stretch;
      overflow: hidden;
    }

    .card{
      position: relative;
      width: 100vw;
      height: 100vh;
      height: 100svh;
      border:0;
      border-radius: 0;
      background:
        linear-gradient(180deg, rgba(255, 255, 247, 0.84), rgba(240, 247, 235, 0.95)),
        var(--surface);
      padding: 14px 14px 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 18px 46px rgba(75, 93, 70, 0.18);
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
      overflow: hidden;
    }

    @supports (height: 100dvh){
      .wrap{ height: 100dvh; }
      .card{ height: 100dvh; }
    }

    .card.controls-hidden{
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
    }

    h1{
      margin:0;
      font-size: 16px;
      letter-spacing: .8px;
      font-weight: 700;
      color:#344239;
      padding-right: 80px;
    }

    .row{ display:flex; gap:10px; align-items:center; flex-wrap: wrap; }

    .control-bar{
      align-self: start;
      min-height: 0;
    }

    .card.controls-hidden .control-bar{
      display: none;
    }

    button{
      border:1px solid var(--line);
      border-radius: 10px;
      background: #f7faf3;
      color: #29372e;
      font-weight: 700;
      padding: 9px 15px;
      cursor: pointer;
      transition: background .15s ease, transform .04s ease, box-shadow .15s ease;
      box-shadow: 0 1px 0 rgba(255, 255, 255, 0.82);
    }
    button:hover{ background:#ffffff; box-shadow: 0 4px 14px rgba(82, 126, 114, 0.12); }
    button:active{ transform: translateY(1px); }
    button:disabled{ opacity:.55; cursor:not-allowed; }
    button.primary{ border-color:#4f8173; background:var(--accent); color:#fffdf5; }
    button.primary:hover{ background:var(--accent-strong); }
    button.danger{ border-color:#bf806f; background:#a45f4f; color:#fff8f1; }
    button.danger:hover{ background:#8f4f41; }

    .badge{
      border:1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: #556257;
      background:#f3f7ee;
    }
    .ok{ color: var(--ok); border-color: #a8c8ae; background:#edf7ec; }
    .warn{ color: var(--warn); border-color: #dac18c; background:#faf3de; }
    .err{ color: var(--err); border-color: #e2aaa1; background:#fff0ec; }

    .direction-select{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border:1px solid var(--line);
      border-radius: 10px;
      background:var(--surface-soft);
      padding: 6px 10px;
      font-size: 12px;
      color:var(--muted);
    }

    .font-size-control{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border:1px solid var(--line);
      border-radius: 10px;
      background:var(--surface-soft);
      padding: 6px 10px;
      font-size: 12px;
      color:var(--muted);
    }

    .context-control{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex: 1 1 320px;
      min-width: min(320px, 100%);
      border:1px solid var(--line);
      border-radius: 10px;
      background:var(--surface-soft);
      padding: 6px 10px;
      font-size: 12px;
      color:var(--muted);
    }

    .context-control textarea{
      flex: 1 1 auto;
      min-width: 120px;
      height: 30px;
      max-height: 64px;
      resize: vertical;
      border:1px solid #bdcbb7;
      border-radius: 8px;
      background:#fffdf7;
      color:#2f3c33;
      padding: 5px 8px;
      font: inherit;
      line-height: 1.35;
      outline: none;
    }

    .context-control textarea:focus{
      border-color:var(--accent);
      box-shadow: 0 0 0 2px rgba(82, 126, 114, 0.16);
    }

    .source-toggle{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border:1px solid var(--line);
      border-radius: 10px;
      background:var(--surface-soft);
      padding: 6px 10px;
      font-size: 12px;
      color:var(--muted);
    }

    .listener-link{
      display: inline-flex;
      align-items: center;
      border:1px solid var(--line);
      border-radius: 10px;
      background:#f7faf3;
      padding: 8px 12px;
      font-size: 12px;
      color:#29372e;
      font-weight:700;
      text-decoration:none;
    }

    .source-toggle select{
      border:1px solid #bdcbb7;
      border-radius: 8px;
      background:#fffdf7;
      color:#2f3c33;
      padding: 4px 8px;
      font-size: 12px;
      outline: none;
    }

    .direction-select select{
      border:1px solid #bdcbb7;
      border-radius: 8px;
      background:#fffdf7;
      color:#2f3c33;
      padding: 4px 8px;
      font-size: 12px;
      outline: none;
    }

    .font-size-control input{
      width: 58px;
      border:1px solid #bdcbb7;
      border-radius: 8px;
      background:#fffdf7;
      color:#2f3c33;
      padding: 4px 7px;
      font-size: 12px;
      outline: none;
      text-align: center;
    }

    .direction-select select:focus{
      border-color:var(--accent);
      box-shadow: 0 0 0 2px rgba(82, 126, 114, 0.16);
    }

    .source-toggle select:focus{
      border-color:var(--accent);
      box-shadow: 0 0 0 2px rgba(82, 126, 114, 0.16);
    }

    .font-size-control input:focus{
      border-color:var(--accent);
      box-shadow: 0 0 0 2px rgba(82, 126, 114, 0.16);
    }

    .control-reveal{
      position: absolute;
      top: 10px;
      right: 14px;
      z-index: 6;
      display: none;
      border-radius: 999px;
      padding: 6px 11px;
      background: rgba(255, 255, 247, 0.84);
      color: #4f685b;
      border-color: rgba(120, 145, 112, 0.36);
      backdrop-filter: blur(10px);
    }

    .card.controls-hidden .control-reveal{
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .subtitle-stage{
      position: relative;
      border:1px solid rgba(118, 139, 109, 0.24);
      border-radius: 14px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255, 255, 252, 0.54) 0%, rgba(238, 246, 233, 0.8) 62%, rgba(229, 239, 224, 0.92) 100%),
        radial-gradient(circle at 50% -10%, rgba(255, 255, 255, 0.92), transparent 58%),
        linear-gradient(180deg, #f4f8ef, #e8f0e4);
      min-height: 0;
      height: 100%;
      display: grid;
      grid-template-rows: 2fr 1fr;
      align-items: stretch;
      padding: 0;
      gap: 0;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
    }

    .subtitle-lane{
      position: relative;
      min-height: 0;
      overflow: hidden;
    }

    .subtitle-lane + .subtitle-lane{
      border-top: 1px solid rgba(118, 139, 109, 0.2);
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
      scrollbar-width: thin;
      scrollbar-color: rgba(93, 119, 101, 0.28) transparent;
      -ms-overflow-style: none;
      padding: 10px 10px 14px;
    }

    .subtitle-stack::-webkit-scrollbar{
      width: 8px;
      height: 8px;
    }

    .subtitle-stack::-webkit-scrollbar-thumb{
      background: rgba(93, 119, 101, 0.24);
      border-radius: 999px;
    }

    .subtitle-stack::-webkit-scrollbar-track{
      background: transparent;
    }

    .subtitle-line{
      display: block;
      min-height: 1.2em;
    }

    .jump-latest{
      position: absolute;
      right: 12px;
      bottom: 14px;
      z-index: 3;
      border: 1px solid rgba(114, 138, 106, 0.34);
      border-radius: 999px;
      background: rgba(255, 255, 248, 0.88);
      color: #4b6658;
      font-size: 12px;
      font-weight: 700;
      padding: 6px 10px;
      opacity: 0;
      pointer-events: none;
      transform: translateY(6px);
      transition: opacity .16s ease, transform .16s ease, background .16s ease;
      backdrop-filter: blur(8px);
    }

    .jump-latest.is-visible{
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }

    .jump-latest:hover{
      background: rgba(255, 255, 255, 0.96);
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
      font-size: var(--subtitle-top-font-size, clamp(28px, 3.3vw, 42px));
      font-weight: 750;
      color: #1f302b;
      letter-spacing: 0.02em;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.76);
    }

    #text{
      min-height: 34px;
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: var(--subtitle-bottom-font-size, clamp(16px, 2.25vw, 26px));
      font-weight: 560;
      color: #526644;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.78);
    }

    #lang{
      display: none;
    }

    @media (max-width: 720px){
      .card{
        height: 100svh;
        min-height: 100svh;
        padding: 10px;
        gap: 10px;
      }
      h1{
        font-size: 14px;
        letter-spacing: .5px;
        padding-right: 66px;
      }
      .control-bar{
        align-items: stretch;
        gap: 8px;
      }
      .control-bar button,
      .control-bar .listener-link{
        flex: 1 1 calc(50% - 8px);
        justify-content: center;
      }
      .badge{
        flex: 1 0 100%;
      }
      .source-toggle,
      .direction-select,
      .font-size-control,
      .context-control{
        flex: 1 1 100%;
        min-width: 0;
        justify-content: space-between;
      }
      .source-toggle select,
      .direction-select select{
        max-width: 62%;
      }
      .font-size-control input{
        width: 72px;
      }
      .subtitle-stage{
        min-height: 0;
        height: 100%;
      }
      .subtitle-stack{
        padding: 8px 7px 12px;
        line-height: 2.05;
      }
      #translation{
        font-size: var(--subtitle-top-font-size, clamp(22px, 7vw, 32px));
      }
      #text{
        font-size: var(--subtitle-bottom-font-size, clamp(14px, 4.8vw, 20px));
      }
      .control-reveal{
        top: 8px;
        right: 10px;
        padding: 5px 9px;
      }
    }

    @supports (height: 100dvh){
      @media (max-width: 720px){
        .card{
          height: 100dvh;
          min-height: 100dvh;
        }
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div id="appCard" class="card">
      <h1>语音识别与翻译</h1>
      <button id="controlReveal" class="control-reveal" type="button" hidden aria-controls="controlBar" aria-expanded="false">控制</button>

      <div id="controlBar" class="row control-bar">
        <button id="btnStart" class="primary">Start</button>
        <button id="btnStop" class="danger" disabled>Stop</button>
        <span id="status" class="badge warn">Idle</span>
        <label class="source-toggle" for="inputSourceSelect">
          <span id="inputSourceLabel">输入源</span>
          <select id="inputSourceSelect">
            <option value="mic">麦克风</option>
            <option value="system">系统声音</option>
          </select>
        </label>
        <label class="direction-select" for="translationDirectionSelect">
          <span id="translationDirectionLabel">翻译方向</span>
          <select id="translationDirectionSelect">
            <option value="zh2en">中文 -> 英文</option>
            <option value="en2zh">英文 -> 中文</option>
          </select>
        </label>
        <a class="listener-link" href="/listen" target="_blank" rel="noopener">译文朗读</a>
        <label class="font-size-control" for="subtitleTopFontInput">
          <span>上方字号</span>
          <input id="subtitleTopFontInput" aria-label="上方字幕字号" type="number" min="18" max="72" step="1" inputmode="numeric" placeholder="自动" />
        </label>
        <label class="font-size-control" for="subtitleBottomFontInput">
          <span>下方字号</span>
          <input id="subtitleBottomFontInput" aria-label="下方字幕字号" type="number" min="12" max="56" step="1" inputmode="numeric" placeholder="自动" />
        </label>
        <label class="context-control" for="asrContextInput">
          <span>专业术语 Context</span>
          <textarea
            id="asrContextInput"
            rows="1"
            spellcheck="false"
            autocomplete="off"
            placeholder="术语用逗号、空格或换行分隔"
          ></textarea>
        </label>
      </div>

      <div class="subtitle-stage">
        <div class="subtitle-lane">
          <div id="translation" class="subtitle-stack"></div>
          <button id="jumpLatestEn" class="jump-latest" type="button" hidden>最新</button>
        </div>
        <div class="subtitle-lane">
          <div id="text" class="subtitle-stack"></div>
          <button id="jumpLatestZh" class="jump-latest" type="button" hidden>最新</button>
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
  const STOP_FINAL_TIMEOUT_MS = 120000;
  const MAX_SUBTITLE_HISTORY = 100;
  const MAX_VISIBLE_ROWS_ZH = 4;
  const MAX_VISIBLE_ROWS_EN = MAX_VISIBLE_ROWS_ZH + 2;
  const SUBTITLE_SCROLL_BOTTOM_EPSILON_PX = 24;
  const SUBTITLE_TOP_FONT_KEY = "voxbridge_subtitle_top_font_px";
  const SUBTITLE_BOTTOM_FONT_KEY = "voxbridge_subtitle_bottom_font_px";
  const ASR_CONTEXT_MAX_TERMS = __ASR_CONTEXT_MAX_TERMS__;
  const ASR_CONTEXT_MAX_CHARS = __ASR_CONTEXT_MAX_CHARS__;
  const ASR_CONTEXT_STORAGE_KEY = "voxbridge_asr_context_terms";
  const USE_COMMITTED_SENTENCE_EVENTS = true;
  const SUBTITLE_TRACE_DEFAULT = __SUBTITLE_TRACE__;
  const SUBTITLE_TRACE_MAX_EVENTS = __SUBTITLE_TRACE_MAX_EVENTS__;

  const $ = (id) => document.getElementById(id);
  const appCard = $("appCard");
  const controlBar = $("controlBar");
  const controlReveal = $("controlReveal");
  const btnStart = $("btnStart");
  const btnStop = $("btnStop");
  const statusEl = $("status");
  const langEl = $("lang");
  const textEl = $("text");
  const translationEl = $("translation");
  const jumpLatestEn = $("jumpLatestEn");
  const jumpLatestZh = $("jumpLatestZh");
  const inputSourceSelect = $("inputSourceSelect");
  const inputSourceLabel = $("inputSourceLabel");
  const translationDirectionSelect = $("translationDirectionSelect");
  const translationDirectionLabel = $("translationDirectionLabel");
  const subtitleTopFontInput = $("subtitleTopFontInput");
  const subtitleBottomFontInput = $("subtitleBottomFontInput");
  const asrContextInput = $("asrContextInput");
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
  let pendingStartResolve = null;
  let pendingStartReject = null;
  let pendingStartTimer = null;
  let watchdogTimer = null;
  let sessionStartedAt = 0;
  let lastCaptureAt = 0;
  let lastChunkSentAt = 0;
  let lastPartialAt = 0;
  let controlAutoHideTimer = null;
  let subtitleTraceEnabled = false;
  let subtitleTraceSeq = 0;
  let subtitleTraceEvents = [];
  let lastPartialTraceSeq = -1;
  let inputSource = "mic";
  let translationDirection = "zh2en";
  let activeContextMetadata = null;
  const autoScrollRaf = new WeakMap();
  const scrollFollowState = {
    zh: { follow: true, autoScrolling: false },
    en: { follow: true, autoScrolling: false },
  };

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

  (() => {
    let initial = "mic";
    try {
      const saved = String(localStorage.getItem("subtitle_input_source") || "").trim().toLowerCase();
      if (saved) initial = saved;
    } catch (err) {}
    applyInputSource(initial, { silent: true });
  })();

  (() => {
    let initial = "zh2en";
    try {
      const saved = String(localStorage.getItem("subtitle_translation_direction") || "").trim().toLowerCase();
      if (saved) initial = saved;
    } catch (err) {}
    applyTranslationDirection(initial, { silent: true });
  })();

  (() => {
    applySubtitleFontSizes(readSubtitleFontConfig(), { persist: false, silent: true });
  })();

  (() => {
    if (!asrContextInput) return;
    try {
      asrContextInput.value = String(localStorage.getItem(ASR_CONTEXT_STORAGE_KEY) || "");
    } catch (err) {
      asrContextInput.value = "";
    }
  })();

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

  function clearControlAutoHideTimer(){
    if (controlAutoHideTimer) {
      clearTimeout(controlAutoHideTimer);
      controlAutoHideTimer = null;
    }
  }

  function setControlBarHidden(hidden, reason = ""){
    if (!appCard || !controlBar) return;
    const next = !!hidden;
    const prev = appCard.classList.contains("controls-hidden");
    clearControlAutoHideTimer();
    appCard.classList.toggle("controls-hidden", next);
    controlBar.setAttribute("aria-hidden", next ? "true" : "false");
    if (controlReveal) {
      controlReveal.hidden = !next;
      controlReveal.setAttribute("aria-expanded", next ? "false" : "true");
      controlReveal.setAttribute("aria-hidden", next ? "false" : "true");
    }
    if (prev !== next) {
      traceSubtitle("controls_hidden_changed", {
        hidden: next,
        reason: String(reason || ""),
      });
    }
  }

  function scheduleControlBarAutoHide(delayMs = 5200){
    clearControlAutoHideTimer();
    if (!running || awaitingFinal) return;
    controlAutoHideTimer = setTimeout(() => {
      controlAutoHideTimer = null;
      if (!running || awaitingFinal) return;
      if (controlBar && controlBar.contains(document.activeElement)) {
        scheduleControlBarAutoHide(1800);
        return;
      }
      setControlBarHidden(true, "auto_timeout");
    }, Math.max(1200, Number(delayMs || 0)));
  }

  function revealControlBarTemporarily(reason = "manual_reveal"){
    setControlBarHidden(false, reason);
    scheduleControlBarAutoHide(6200);
  }

  function clearCommittedTentativeTailNow(){
    if (currentTextTail) {
      traceSubtitle("tail_cleared", { by: "clearCommittedTentativeTailNow", prevLen: currentTextTail.length });
    }
    currentTextTail = "";
  }

  function readBackendStability(msg){
    const source = msg && typeof msg === "object" ? msg : {};
    const meta = source.stability && typeof source.stability === "object" ? source.stability : {};
    const metaStable = typeof meta.is_stable === "boolean" ? meta.is_stable : null;
    const msgStable = typeof source.is_stable === "boolean" ? source.is_stable : null;
    const hasSignal = metaStable !== null || msgStable !== null;
    return {
      hasSignal,
      isStable: metaStable !== null ? metaStable : (msgStable === true),
      phase: String(meta.phase || ""),
      reason: String(meta.reason || ""),
      unstableChars: Number(meta.unstable_chars || 0),
    };
  }

  function updateCommittedTentativeTailFromBackend(tailText, stability){
    const nextTail = String(tailText || "").trim();
    const signal = stability && typeof stability === "object" ? stability : { hasSignal: false, isStable: false, phase: "" };
    if (nextTail) {
      if (nextTail !== currentTextTail) {
        traceSubtitle("tail_set", {
          by: "updateCommittedTentativeTailFromBackend",
          nextLen: nextTail.length,
          stable: !!signal.isStable,
          phase: String(signal.phase || ""),
        });
      }
      currentTextTail = nextTail;
      return;
    }
    if (!currentTextTail) {
      currentTextTail = "";
      return;
    }

    if (!signal.hasSignal || signal.isStable || signal.phase === "solidified" || signal.phase === "final") {
      traceSubtitle("tail_cleared", {
        by: "backend_stability",
        prevLen: currentTextTail.length,
        stable: !!signal.isStable,
        phase: String(signal.phase || ""),
        reason: String(signal.reason || ""),
      });
      currentTextTail = "";
      return;
    }
    traceSubtitle("tail_kept_unstable", {
      prevLen: currentTextTail.length,
      phase: String(signal.phase || ""),
      reason: String(signal.reason || ""),
      unstableChars: Number(signal.unstableChars || 0),
    });
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

  function listeningStatus(sourceMode, started = activeContextMetadata){
    let base = "Listening / 识别中";
    if (sourceMode === "system") {
      base = "Listening (system audio) / 识别中(系统声音)";
    } else if (processor) {
      base = "Listening (fallback) / 识别中(兼容模式)";
    }
    const contextSuffix = started && started.asr_context_active
      ? ` · Context 已启用 · ${Number(started.asr_context_term_count || 0)} 个术语`
      : "";
    return base + contextSuffix;
  }

  function normalizeTranslationDirection(raw){
    const text = String(raw || "").trim().toLowerCase();
    if (text === "en2zh" || text === "en->zh") return "en2zh";
    return "zh2en";
  }

  function normalizeInputSource(raw){
    const text = String(raw || "").trim().toLowerCase();
    return text === "system" ? "system" : "mic";
  }

  function parseAsrContextTerms(raw){
    const terms = [];
    const seen = new Set();
    for (const part of String(raw || "").split(/[\s,，]+/u)) {
      const term = String(part || "").trim();
      if (!term) continue;
      const key = term.toLocaleLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      terms.push(term);
    }
    return terms;
  }

  function asrContextTermHasSentencePunctuation(term){
    const value = String(term || "");
    if (/[。！？!?；;:：]/u.test(value)) return true;
    if (!/\.["'”’)\]）】》]*$/u.test(value)) return false;
    return !/^(?:[A-Z]\.){2,}$/u.test(value);
  }

  function readAsrContextTerms(){
    const terms = parseAsrContextTerms(asrContextInput ? asrContextInput.value : "");
    const invalidIndex = terms.findIndex(asrContextTermHasSentencePunctuation);
    if (invalidIndex >= 0) {
      throw new Error(`Context 第 ${invalidIndex + 1} 个术语包含句子标点`);
    }
    if (terms.length > ASR_CONTEXT_MAX_TERMS) {
      throw new Error(`Context 最多允许 ${ASR_CONTEXT_MAX_TERMS} 个术语`);
    }
    const chars = terms.join(" ").length;
    if (chars > ASR_CONTEXT_MAX_CHARS) {
      throw new Error(`Context 最多允许 ${ASR_CONTEXT_MAX_CHARS} 个字符`);
    }
    return terms;
  }

  function persistAsrContextInput(){
    if (!asrContextInput) return;
    const raw = String(asrContextInput.value || "");
    try {
      if (raw.trim()) localStorage.setItem(ASR_CONTEXT_STORAGE_KEY, raw);
      else localStorage.removeItem(ASR_CONTEXT_STORAGE_KEY);
    } catch (err) {}
  }

  function selectedInputSource(){
    if (!inputSourceSelect) return inputSource;
    return normalizeInputSource(inputSourceSelect.value);
  }

  function applyInputSource(source, options = {}){
    const normalized = normalizeInputSource(source);
    const prev = inputSource;
    inputSource = normalized;
    if (inputSourceSelect) {
      inputSourceSelect.value = normalized;
    }
    if (inputSourceLabel) {
      inputSourceLabel.textContent = normalized === "system" ? "系统声音" : "麦克风";
    }
    try {
      localStorage.setItem("subtitle_input_source", normalized);
    } catch (err) {}
    const silent = !!(options && options.silent);
    if (!silent && prev !== normalized) {
      traceSubtitle("input_source_ui_set", { prev, next: normalized });
    }
    return normalized;
  }

  function selectedTranslationDirection(){
    if (!translationDirectionSelect) return translationDirection;
    return normalizeTranslationDirection(translationDirectionSelect.value);
  }

  function applyTranslationDirection(direction, options = {}){
    const normalized = normalizeTranslationDirection(direction);
    const prev = translationDirection;
    translationDirection = normalized;
    if (translationDirectionSelect) {
      translationDirectionSelect.value = normalized;
    }
    if (translationDirectionLabel) {
      translationDirectionLabel.textContent = "翻译方向";
    }
    try {
      localStorage.setItem("subtitle_translation_direction", normalized);
    } catch (err) {}
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

  function normalizeSubtitleFontPx(raw, minPx, maxPx){
    const text = String(raw == null ? "" : raw).trim();
    if (!text) return "";
    const value = Number(text);
    if (!Number.isFinite(value)) return "";
    return Math.max(Number(minPx || 12), Math.min(Number(maxPx || 72), Math.round(value)));
  }

  function readSubtitleFontConfig(){
    let top = "";
    let bottom = "";
    try {
      top = normalizeSubtitleFontPx(localStorage.getItem(SUBTITLE_TOP_FONT_KEY), 18, 72);
      bottom = normalizeSubtitleFontPx(localStorage.getItem(SUBTITLE_BOTTOM_FONT_KEY), 12, 56);
    } catch (err) {}
    return { top, bottom };
  }

  function writeSubtitleFontConfig(config){
    const next = config && typeof config === "object" ? config : {};
    const top = normalizeSubtitleFontPx(next.top, 18, 72);
    const bottom = normalizeSubtitleFontPx(next.bottom, 12, 56);
    try {
      if (top) localStorage.setItem(SUBTITLE_TOP_FONT_KEY, String(top));
      else localStorage.removeItem(SUBTITLE_TOP_FONT_KEY);
      if (bottom) localStorage.setItem(SUBTITLE_BOTTOM_FONT_KEY, String(bottom));
      else localStorage.removeItem(SUBTITLE_BOTTOM_FONT_KEY);
    } catch (err) {}
    return { top, bottom };
  }

  function subtitleComputedFontPx(element){
    if (!element || typeof getComputedStyle !== "function") return "";
    const value = Number.parseFloat(getComputedStyle(element).fontSize || "");
    if (!Number.isFinite(value)) return "";
    return String(Math.round(value));
  }

  function syncSubtitleFontInputs(config){
    const next = config && typeof config === "object" ? config : {};
    if (subtitleTopFontInput) {
      subtitleTopFontInput.value = next.top ? String(next.top) : "";
      subtitleTopFontInput.placeholder = subtitleComputedFontPx(translationEl) || "自动";
    }
    if (subtitleBottomFontInput) {
      subtitleBottomFontInput.value = next.bottom ? String(next.bottom) : "";
      subtitleBottomFontInput.placeholder = subtitleComputedFontPx(textEl) || "自动";
    }
  }

  function applySubtitleFontSizes(config, options = {}){
    const next = config && typeof config === "object" ? config : {};
    const top = normalizeSubtitleFontPx(next.top, 18, 72);
    const bottom = normalizeSubtitleFontPx(next.bottom, 12, 56);
    const rootStyle = document.documentElement.style;
    if (top) rootStyle.setProperty("--subtitle-top-font-size", `${top}px`);
    else rootStyle.removeProperty("--subtitle-top-font-size");
    if (bottom) rootStyle.setProperty("--subtitle-bottom-font-size", `${bottom}px`);
    else rootStyle.removeProperty("--subtitle-bottom-font-size");
    const normalized = { top, bottom };
    if (options && options.persist) {
      writeSubtitleFontConfig(normalized);
    }
    syncSubtitleFontInputs(normalized);
    if (!(options && options.silent)) {
      traceSubtitle("subtitle_font_config_set", {
        top: top || "auto",
        bottom: bottom || "auto",
        persist: !!(options && options.persist),
      });
    }
    return normalized;
  }

  function applySubtitleFontInputs(){
    applySubtitleFontSizes(
      {
        top: subtitleTopFontInput ? subtitleTopFontInput.value : "",
        bottom: subtitleBottomFontInput ? subtitleBottomFontInput.value : "",
      },
      { persist: true }
    );
  }

  function lockUI(active){
    btnStart.disabled = active;
    btnStop.disabled = !active;
    if (inputSourceSelect) inputSourceSelect.disabled = active;
    if (translationDirectionSelect) translationDirectionSelect.disabled = active;
    if (asrContextInput) asrContextInput.disabled = active;
  }

  function lockUIFinishing(){
    btnStart.disabled = true;
    btnStop.disabled = true;
    if (inputSourceSelect) inputSourceSelect.disabled = true;
    if (translationDirectionSelect) translationDirectionSelect.disabled = true;
    if (asrContextInput) asrContextInput.disabled = true;
  }

  function laneForContainer(container){
    if (container === textEl) return "zh";
    if (container === translationEl) return "en";
    return "unknown";
  }

  function containerForLane(lane){
    if (lane === "zh") return textEl;
    if (lane === "en") return translationEl;
    return null;
  }

  function buttonForLane(lane){
    if (lane === "zh") return jumpLatestZh;
    if (lane === "en") return jumpLatestEn;
    return null;
  }

  function isNearSubtitleBottom(container){
    if (!container) return true;
    const remaining = Math.max(
      0,
      Number(container.scrollHeight || 0) - Number(container.clientHeight || 0) - Number(container.scrollTop || 0)
    );
    return remaining <= SUBTITLE_SCROLL_BOTTOM_EPSILON_PX;
  }

  function updateJumpLatestButtons(){
    for (const lane of ["en", "zh"]) {
      const button = buttonForLane(lane);
      const container = containerForLane(lane);
      const state = scrollFollowState[lane];
      if (!button || !container || !state) continue;
      const hasOverflow = Number(container.scrollHeight || 0) > (Number(container.clientHeight || 0) + SUBTITLE_SCROLL_BOTTOM_EPSILON_PX);
      const visible = !state.follow && hasOverflow;
      button.hidden = !visible;
      button.classList.toggle("is-visible", visible);
      button.setAttribute("aria-hidden", visible ? "false" : "true");
      button.disabled = !visible;
      button.title = lane === "en" ? "滚动到最新英文字幕" : "滚动到最新中文字幕";
    }
  }

  function pauseSubtitleAutoFollow(lane, options = {}){
    const state = scrollFollowState[lane];
    if (!state || !state.follow) return false;
    state.follow = false;
    traceSubtitle("scroll_follow_paused", {
      lane,
      reason: String(options.reason || "user_scroll"),
      scrollTop: Math.round(Number(options.scrollTop || 0)),
    });
    updateJumpLatestButtons();
    return true;
  }

  function resumeSubtitleAutoFollow(lane, options = {}){
    const state = scrollFollowState[lane];
    if (!state) return false;
    const wasFollowing = !!state.follow;
    state.follow = true;
    if (!wasFollowing) {
      traceSubtitle("scroll_follow_resumed", {
        lane,
        reason: String(options.reason || "bottom_reached"),
      });
    }
    updateJumpLatestButtons();
    if (options.pin !== false) {
      pinScrollToBottom(containerForLane(lane), { force: true });
    }
    return !wasFollowing;
  }

  function bindSubtitleScrollTracking(container){
    if (!container || container.dataset.scrollBound === "1") return;
    const lane = laneForContainer(container);
    if (lane === "unknown") return;
    container.dataset.scrollBound = "1";
    container.addEventListener("scroll", () => {
      const state = scrollFollowState[lane];
      if (!state || state.autoScrolling) return;
      if (isNearSubtitleBottom(container)) {
        resumeSubtitleAutoFollow(lane, { reason: "user_bottom", pin: false });
        return;
      }
      pauseSubtitleAutoFollow(lane, {
        reason: "user_scroll",
        scrollTop: Number(container.scrollTop || 0),
      });
    }, { passive: true });
  }

  function resetSubtitleAutoFollow(){
    scrollFollowState.zh.follow = true;
    scrollFollowState.zh.autoScrolling = false;
    scrollFollowState.en.follow = true;
    scrollFollowState.en.autoScrolling = false;
    updateJumpLatestButtons();
  }

  function pinScrollToBottom(container, options = {}){
    if (!container) return;
    const lane = laneForContainer(container);
    const state = scrollFollowState[lane];
    const force = !!(options && options.force);
    if (state && !state.follow && !force) return;
    if (state) state.autoScrolling = true;
    container.scrollTop = container.scrollHeight;
    if (typeof requestAnimationFrame !== "function") {
      if (state) state.autoScrolling = false;
      return;
    }
    const prevHandle = autoScrollRaf.get(container);
    if (prevHandle) {
      cancelAnimationFrame(prevHandle);
    }
    const handle = requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
      if (state) state.autoScrolling = false;
      autoScrollRaf.delete(container);
    });
    autoScrollRaf.set(container, handle);
  }

  function setRawAsrText(text, options = {}){
    if (options && options.resetCurrent) {
      rawAsrLastSnapshot = "";
      if (rawTextEl) {
        rawTextEl.textContent = rawAsrText;
        pinScrollToBottom(rawTextEl);
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
        pinScrollToBottom(rawTextEl);
      }
      return;
    } else {
      rawAsrText = `${rawAsrText}\n${next}`;
    }

    rawAsrLastSnapshot = next;
    if (rawTextEl) {
      rawTextEl.textContent = rawAsrText;
      pinScrollToBottom(rawTextEl);
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

  function resolveTentativeTail(nextText, committedText, tentativeText){
    const tentative = String(tentativeText || "").trim();
    if (tentative) return tentative;
    const full = String(nextText || "").trim();
    if (!full) return "";
    const committed = String(committedText || "").trim();
    if (!committed) return full;
    if (!full.startsWith(committed)) return "";
    const tail = full.slice(committed.length).trim();
    return tail;
  }

  function trimSubtitleHistory(){
    const maxKeep = Math.max(1, Number(MAX_SUBTITLE_HISTORY || 100));
    const overflow = Math.max(0, subtitleSentencePairs.length - maxKeep);
    if (overflow <= 0) return;
    const droppedIds = subtitleSentencePairs.slice(0, overflow).map((item) => String(item.sid || "")).slice(0, 8);
    subtitleSentencePairs = subtitleSentencePairs.slice(overflow);
    traceSubtitle("history_trimmed", {
      drop: overflow,
      remaining: subtitleSentencePairs.length,
      maxKeep,
      droppedIds,
    });
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
      const rows = committedRows.slice();
      if (tail || (running && committedRows.length > 0)) {
        rows.push({ sid: "__tail__", zh: tail, en: "" });
      }
      return rows;
    }

    const rows = committedRows.slice();
    if (tail) {
      rows.push({ sid: "__tail__", zh: tail, en: "" });
    }
    return rows;
  }

  function clearSubtitleDom(){
    if (textEl) textEl.replaceChildren();
    if (translationEl) translationEl.replaceChildren();
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
    pinScrollToBottom(container);
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
        follow: lane !== "unknown" ? !!scrollFollowState[lane].follow : true,
      });
    }
    updateJumpLatestButtons();
    return nextNodes;
  }

  function clearPendingStartTimer(){
    if (pendingStartTimer !== null) clearTimeout(pendingStartTimer);
    pendingStartTimer = null;
  }

  function resolvePendingStart(msg){
    const resolve = pendingStartResolve;
    pendingStartResolve = null;
    pendingStartReject = null;
    clearPendingStartTimer();
    if (resolve) resolve(msg);
  }

  function rejectPendingStart(err){
    const reject = pendingStartReject;
    pendingStartResolve = null;
    pendingStartReject = null;
    clearPendingStartTimer();
    if (reject) reject(err instanceof Error ? err : new Error(String(err)));
  }

  function waitForStarted(timeoutMs = 10000){
    if (pendingStartResolve) throw new Error("start already pending");
    return new Promise((resolve, reject) => {
      pendingStartResolve = resolve;
      pendingStartReject = reject;
      pendingStartTimer = setTimeout(() => {
        rejectPendingStart(new Error("start acknowledgement timeout"));
      }, Math.max(1000, Number(timeoutMs) || 10000));
    });
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
    rejectPendingStart(new Error("session reset"));
    running = false;
    awaitingFinal = false;
    activeContextMetadata = null;
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
    setControlBarHidden(false, "reset_session");
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
      return "音频采集权限被拒绝，请允许麦克风或屏幕共享音频。";
    }
    if (name === "NotFoundError") {
      return "未检测到可用音频源，请检查麦克风或共享源是否包含音频。";
    }
    if (name === "NotReadableError") {
      return "音频输入不可读，可能被其他应用占用或共享被系统阻止。";
    }
    if (name === "OverconstrainedError") {
      return "音频参数不兼容，已建议改用默认采集配置。";
    }
    if (name === "AbortError") {
      return "音频采集初始化被中断，请重试。";
    }
    if (name === "InvalidStateError") {
      return "请通过用户手势启动采集（点击 Start），然后重新选择输入源。";
    }
    return `${name}: ${msg}`;
  }

  function selectedLanguage(){
    if (!languageSelect) return "";
    return String(languageSelect.value || "").trim();
  }

  function selectedAsrLanguage(){
    const explicitLanguage = selectedLanguage();
    if (explicitLanguage) return explicitLanguage;
    return selectedTranslationDirection() === "en2zh" ? "English" : "Chinese";
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

  async function openSystemAudio(){
    if (!window.isSecureContext && !isLocalhost()) {
      throw new Error("远程访问系统声音需要 HTTPS。请确认通过 Caddy 的 https 地址访问。");
    }

    const getDisplayMedia =
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getDisplayMedia === "function"
        ? navigator.mediaDevices.getDisplayMedia.bind(navigator.mediaDevices)
        : null;
    if (!getDisplayMedia) {
      throw new Error("当前浏览器不支持系统声音采集，请使用最新版 Chrome/Edge。");
    }

    const preferredConstraints = {
      video: {
        displaySurface: "monitor",
      },
      audio: {
        suppressLocalAudioPlayback: false,
      },
      systemAudio: "include",
      preferCurrentTab: false,
      selfBrowserSurface: "exclude",
      surfaceSwitching: "include",
      monitorTypeSurfaces: "include",
    };

    let stream = null;
    try {
      stream = await getDisplayMedia(preferredConstraints);
    } catch (err) {
      if (err && (err.name === "OverconstrainedError" || err.name === "TypeError")) {
        stream = await getDisplayMedia({ video: true, audio: true });
      } else {
        throw err;
      }
    }

    const audioTracks = stream ? stream.getAudioTracks() : [];
    if (!audioTracks || audioTracks.length === 0) {
      if (stream) {
        for (const track of stream.getTracks()) {
          try { track.stop(); } catch (err) {}
        }
      }
      throw new Error("未检测到共享音频。请选择整屏共享并勾选系统音频后重试。");
    }

    const audioTrack = audioTracks[0];
    audioTrack.addEventListener("ended", () => {
      traceSubtitle("system_audio_track_ended", {});
      if (running && btnStop && !btnStop.disabled) {
        btnStop.click();
      }
    }, { once: true });
    return stream;
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
      activeContextMetadata = {
        asr_context_active: !!msg.asr_context_active,
        asr_context_term_count: Number(msg.asr_context_term_count || 0),
        asr_context_chars: Number(msg.asr_context_chars || 0),
      };
      resolvePendingStart(msg);
      if (msg.translation_direction) {
        applyTranslationDirection(msg.translation_direction);
      }
      traceSubtitle("ws_started", {
        language: String(msg.language || ""),
        translationDirection: String(msg.translation_direction || selectedTranslationDirection()),
        contextActive: !!msg.asr_context_active,
        contextTermCount: Number(msg.asr_context_term_count || 0),
        contextChars: Number(msg.asr_context_chars || 0),
      });
      if (msg.language) {
        if (langEl) langEl.textContent = msg.language;
      } else {
        if (langEl) langEl.textContent = "-";
      }
      setCurrentSegmentText("");
      setRawAsrText("", { resetCurrent: true });
      resetSubtitleAutoFollow();
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
      trimSubtitleHistory();
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
      trimSubtitleHistory();
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
      resetSubtitleAutoFollow();
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
        const committedText = String(msg.committed_text || "").trim();
        const tentativeTail = resolveTentativeTail(
          nextText,
          committedText,
          msg.tentative_text || ""
        );
        const stability = readBackendStability(msg);
        if (lastPartialSeq <= 5 || (lastPartialSeq % 20 === 0 && lastPartialTraceSeq !== lastPartialSeq)) {
          traceSubtitle("ws_partial", {
            seq: lastPartialSeq,
            textLen: nextText.trim().length,
            tailLen: tentativeTail.length,
            committedCount: subtitleSentencePairs.length,
            stable: !!stability.isStable,
            stabilityPhase: String(stability.phase || ""),
          });
          lastPartialTraceSeq = lastPartialSeq;
        }
        updateCommittedTentativeTailFromBackend(tentativeTail, stability);
        currentTranslationTail = "";
        renderTranscript();
        renderTranslation();
        if (running) {
          setStatus(listeningStatus(selectedInputSource()), "ok");
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

      setCurrentSegmentText("");
      updateCommittedTentativeTailFromBackend(String(msg.tentative_text || "").trim(), readBackendStability(msg));
      currentTranslationTail = "";
      renderTranscript();
      renderTranslation();
      awaitingFinal = false;
      lockUI(false);
      setControlBarHidden(false, "final");
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
      rejectPendingStart(new Error(msg.message || "websocket server error"));
      rejectPendingFinal(new Error(msg.message || "websocket server error"));
      resetSessionFlags();
      stopPipeline();
      setControlBarHidden(false, "server_error");
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
        const err = new Error("websocket failed");
        rejectPendingStart(err);
        finish(reject, err);
      };
      sock.onclose = (evt) => {
        if (sock !== ws) return;
        const err = new Error(`websocket closed (${evt.code})`);
        rejectPendingStart(err);
        if (!done) {
          finish(reject, err);
        }
        rejectPendingFinal(err);
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

  if (translationDirectionSelect) {
    translationDirectionSelect.addEventListener("change", () => {
      const next = selectedTranslationDirection();
      applyTranslationDirection(next);
      sendTranslationDirection(next);
    });
  }

  if (inputSourceSelect) {
    inputSourceSelect.addEventListener("change", () => {
      const next = selectedInputSource();
      applyInputSource(next);
    });
  }

  if (subtitleTopFontInput) {
    subtitleTopFontInput.addEventListener("input", applySubtitleFontInputs);
    subtitleTopFontInput.addEventListener("change", applySubtitleFontInputs);
  }

  if (subtitleBottomFontInput) {
    subtitleBottomFontInput.addEventListener("input", applySubtitleFontInputs);
    subtitleBottomFontInput.addEventListener("change", applySubtitleFontInputs);
  }

  if (asrContextInput) {
    asrContextInput.addEventListener("input", persistAsrContextInput);
    asrContextInput.addEventListener("change", persistAsrContextInput);
  }

  bindSubtitleScrollTracking(textEl);
  bindSubtitleScrollTracking(translationEl);

  if (jumpLatestEn) {
    jumpLatestEn.addEventListener("click", () => {
      traceSubtitle("jump_latest_clicked", { lane: "en" });
      resumeSubtitleAutoFollow("en", { reason: "button_click", pin: true });
    });
  }

  if (jumpLatestZh) {
    jumpLatestZh.addEventListener("click", () => {
      traceSubtitle("jump_latest_clicked", { lane: "zh" });
      resumeSubtitleAutoFollow("zh", { reason: "button_click", pin: true });
    });
  }

  if (controlReveal) {
    controlReveal.addEventListener("click", () => {
      traceSubtitle("control_reveal_clicked", {});
      revealControlBarTemporarily("reveal_button");
    });
  }

  if (controlBar) {
    controlBar.addEventListener("focusin", () => {
      if (running && !awaitingFinal) clearControlAutoHideTimer();
    });
    controlBar.addEventListener("focusout", () => {
      scheduleControlBarAutoHide(2600);
    });
    controlBar.addEventListener("pointerenter", () => {
      if (running && !awaitingFinal) clearControlAutoHideTimer();
    });
    controlBar.addEventListener("pointerleave", () => {
      scheduleControlBarAutoHide(2600);
    });
  }

  btnStart.onclick = async () => {
    if (running || awaitingFinal) return;
    let asrContextTerms = [];
    try {
      asrContextTerms = readAsrContextTerms();
    } catch (err) {
      setControlBarHidden(false, "start_failed");
      lockUI(false);
      setStatus("Start failed / 启动失败: " + describeStartError(err), "err");
      return;
    }
    subtitleSentencePairs = [];
    setCurrentSegmentText("");
    resetSubtitleAutoFollow();
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
      const sourceMode = selectedInputSource();
      traceSubtitle("capture_source_starting", { source: sourceMode });
      if (sourceMode === "system") {
        setStatus("Share full screen + system audio / 请选择整屏共享并勾选系统音频", "warn");
      }
      // Request media inside the click activation; no samples are read or sent until started.
      mediaStream = sourceMode === "system" ? await openSystemAudio() : await openMicrophone();
      await openSocket();
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new Error("websocket is not open");
      }
      const startedPromise = waitForStarted(10000);
      ws.send(
        JSON.stringify({
          type: "start",
          language: selectedAsrLanguage(),
          translation_direction: selectedTranslationDirection(),
          asr_context_terms: asrContextTerms,
        })
      );
      const started = await startedPromise;

      await buildCaptureGraph();

      running = true;
      sessionStartedAt = Date.now();
      startWatchdog();
      setStatus(listeningStatus(sourceMode, started), "ok");
      setControlBarHidden(true, "start_success");
    } catch (err) {
      console.error(err);
      rejectPendingStart(err);
      activeContextMetadata = null;
      await stopPipeline();
      if (ws) {
        try { ws.close(); } catch (closeErr) {}
      }
      ws = null;
      running = false;
      lockUI(false);
      setControlBarHidden(false, "start_failed");
      setStatus("Start failed / 启动失败: " + describeStartError(err), "err");
    }
  };

  btnStop.onclick = async () => {
    if (!running) return;
    // Stop microphone first, then flush queued PCM before sending finish.
    running = false;
    awaitingFinal = true;
    setControlBarHidden(false, "stop_requested");
    lockUIFinishing();
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
        await sendFinishAndAwaitFinal("stop", STOP_FINAL_TIMEOUT_MS);
      } else {
        awaitingFinal = false;
        lockUI(false);
        setStatus("Stopped / 已停止", "");
      }
    } catch (err) {
      const msg = String((err && err.message) ? err.message : (err || ""));
      if (msg.includes("final timeout")) {
        traceSubtitle("stop_wait_final_timeout", { timeoutMs: STOP_FINAL_TIMEOUT_MS });
        // Backend final flush can be slow after long meetings; keep waiting on the same WS.
        setStatus("Finishing (slow backend) / 收尾中(后端较慢)", "warn");
        return;
      }
      console.error(err);
      rejectPendingFinal(err instanceof Error ? err : new Error(String(err)));
      awaitingFinal = false;
      lockUI(false);
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (closeErr) {}
      }
      setStatus("Stop failed / 停止失败", "err");
    }
  };

  if (typeof window !== "undefined") {
    const _resetDebugSubtitleState = () => {
      subtitleSentencePairs = [];
      resetSubtitleAutoFollow();
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

        const asrContextTerms = readAsrContextTerms();
        const language = String(cfg.language || selectedAsrLanguage() || "auto");
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
        let startAccepted = false;
        let startedMetadata = null;
        let finished = false;
        let errorMessage = "";

        sock.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            events.push(msg);
            if (msg.type === "ready") ready = true;
            if (msg.type === "started") {
              startAccepted = true;
              startedMetadata = msg;
            }
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
            asr_context_terms: asrContextTerms,
          })
        );
        const startOk = await _waitUntil(
          () => startAccepted || !!errorMessage,
          timeoutMs,
        );
        if (!startOk || errorMessage) {
          throw new Error(errorMessage || "start acknowledgement timeout");
        }
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
          started: startedMetadata ? {
            asr_context_active: !!startedMetadata.asr_context_active,
            asr_context_term_count: Number(startedMetadata.asr_context_term_count || 0),
            asr_context_chars: Number(startedMetadata.asr_context_chars || 0),
          } : null,
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
          controlsHidden: !!(appCard && appCard.classList.contains("controls-hidden")),
          subtitleTopFontPx: subtitleComputedFontPx(translationEl),
          subtitleBottomFontPx: subtitleComputedFontPx(textEl),
          subtitleTraceEnabled,
          subtitleTraceCount: subtitleTraceEvents.length,
          currentTextTail: String(currentTextTail || ""),
          currentSegmentText: String(currentSegmentText || ""),
          historyCount: subtitleSentencePairs.length,
          scrollFollowState: {
            zh: { ...scrollFollowState.zh },
            en: { ...scrollFollowState.en },
          },
          zhScrollTop: textEl ? Number(textEl.scrollTop || 0) : 0,
          zhScrollHeight: textEl ? Number(textEl.scrollHeight || 0) : 0,
          zhClientHeight: textEl ? Number(textEl.clientHeight || 0) : 0,
          enScrollTop: translationEl ? Number(translationEl.scrollTop || 0) : 0,
          enScrollHeight: translationEl ? Number(translationEl.scrollHeight || 0) : 0,
          enClientHeight: translationEl ? Number(translationEl.clientHeight || 0) : 0,
          jumpLatestZhVisible: !!(jumpLatestZh && !jumpLatestZh.hidden),
          jumpLatestEnVisible: !!(jumpLatestEn && !jumpLatestEn.hidden),
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


LOGIN_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VoxBridge 登录</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#eef4ec;
      --panel:#fbfdf8;
      --text:#233026;
      --muted:#667263;
      --accent:#476a46;
      --danger:#9f3328;
      --line:#d7dfd3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 20% 15%, rgba(255,255,255,.75), transparent 34rem),
        linear-gradient(135deg, #eef4ec, #f6f0e6);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(92vw, 420px);
      padding: 32px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(251, 253, 248, .9);
      box-shadow: 0 24px 72px rgba(46, 55, 43, .16);
    }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: .02em; }
    p { margin: 0 0 24px; color: var(--muted); line-height: 1.6; }
    label { display: block; margin: 14px 0 6px; font-weight: 700; }
    input {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      color: var(--text);
      font-size: 16px;
    }
    button {
      width: 100%;
      margin-top: 22px;
      padding: 13px 16px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font-size: 16px;
      font-weight: 800;
      cursor: pointer;
    }
    .error {
      margin: 0 0 18px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(159, 51, 40, .1);
      color: var(--danger);
    }
  </style>
</head>
<body>
  <main>
    <h1>VoxBridge</h1>
    <p>请输入访问凭据后继续使用语音识别与翻译。</p>
    __MESSAGE__
    <form method="post" action="/login" autocomplete="on">
      __NEXT_FIELD__
      <label for="username">用户名</label>
      <input id="username" name="username" type="text" autocomplete="username" required autofocus />
      <label for="password">密码</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required />
      <button type="submit">登录</button>
    </form>
  </main>
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


async def _run_ordered_tts_transition(
    transition_lock: asyncio.Lock,
    transition: Any,
    publish: Any,
) -> None:
    async with transition_lock:
        ready = transition()
        if ready:
            await publish(ready)


def _create_app(
    args: argparse.Namespace,
    asr: Any,
    translator: Optional[LocalTranslator] = None,
    tts_synthesizer: Any = None,
) -> FastAPI:
    app = FastAPI(title="VoxBridge Streaming WebSocket Demo")
    infer_lock = asyncio.Lock()
    tts_synthesis_lock = asyncio.Lock()
    app.state.tts_jobs = TTSJobRegistry(
        ttl_sec=max(1.0, float(getattr(args, "tts_job_ttl_sec", 1800.0))),
        max_client_jobs=max(1, int(getattr(args, "tts_max_client_jobs", 4096))),
    )
    app.state.tts_broadcast = TTSBroadcastHub(
        ttl_sec=max(1.0, float(getattr(args, "tts_job_ttl_sec", 1800.0))),
        max_jobs=max(1, int(getattr(args, "tts_max_client_jobs", 4096))),
        listener_queue_size=max(1, int(getattr(args, "tts_listener_queue_size", 128))),
    )
    app.state.tts_synthesizer = tts_synthesizer
    runtime = SimpleNamespace(active_connections=0)
    translator_accepts_direction = False
    if translator is not None:
        with suppress(TypeError, ValueError):
            translate_parameters = inspect.signature(translator.translate).parameters.values()
            translator_accepts_direction = any(
                parameter.name == "translation_direction"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in translate_parameters
            )
    debug_roots = [Path.cwd().resolve(), Path("/tmp").resolve()]
    auth = _build_auth_config(args)
    asr_context_schedule = _load_asr_context_schedule(args)
    asr_context_max_terms = max(0, int(getattr(args, "asr_context_max_terms", 24)))
    asr_context_max_chars = max(0, int(getattr(args, "asr_context_max_chars", 160)))
    asr_context_lookaround_sec = max(0.0, float(getattr(args, "asr_context_lookaround_sec", 30.0)))
    asr_context_apply_mode = _normalize_asr_context_apply_mode(
        getattr(args, "asr_context_apply_mode", "segment_final")
    )
    auth_sessions: Dict[str, float] = {}

    def _safe_login_next(value: Any) -> str:
        target = str(value or "").strip()
        if (
            not target.startswith("/")
            or target.startswith("//")
            or "\\" in target
            or any(ord(char) < 32 for char in target)
        ):
            return "/"
        parsed = urllib.parse.urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return "/"
        return target

    def _render_login_html(error: bool = False, next_target: str = "/") -> str:
        message = '<div class="error">用户名或密码错误。</div>' if error else ""
        safe_target = _safe_login_next(next_target)
        next_field = (
            '<input type="hidden" name="next" value="'
            + html.escape(safe_target, quote=True)
            + '" />'
        )
        return LOGIN_HTML_TEMPLATE.replace("__MESSAGE__", message).replace(
            "__NEXT_FIELD__",
            next_field,
        )

    def _prune_auth_sessions(now: Optional[float] = None) -> None:
        current = time.time() if now is None else float(now)
        expired = [token for token, expires_at in auth_sessions.items() if expires_at <= current]
        for token in expired:
            auth_sessions.pop(token, None)

    def _new_auth_session() -> str:
        _prune_auth_sessions()
        token = secrets.token_urlsafe(AUTH_SESSION_TOKEN_BYTES)
        auth_sessions[token] = time.time() + float(auth.session_ttl_sec)
        return token

    def _drop_auth_session(token: Optional[str]) -> None:
        if token:
            auth_sessions.pop(str(token), None)

    def _is_valid_auth_session(token: Optional[str]) -> bool:
        if not auth.enabled:
            return True
        token_text = str(token or "")
        if not token_text:
            return False
        _prune_auth_sessions()
        expires_at = auth_sessions.get(token_text)
        if expires_at is None or expires_at <= time.time():
            auth_sessions.pop(token_text, None)
            return False
        return True

    def _request_is_authenticated(request: Request) -> bool:
        return _is_valid_auth_session(request.cookies.get(AUTH_COOKIE_NAME))

    def _websocket_is_authenticated(websocket: WebSocket) -> bool:
        return _is_valid_auth_session(websocket.cookies.get(AUTH_COOKIE_NAME))

    def _tts_owner_key_for_token(token: Optional[str]) -> str:
        if not _is_valid_auth_session(token):
            raise HTTPException(status_code=401, detail="unauthorized")
        if not auth.enabled:
            return "anonymous"
        return hashlib.sha256(f"auth:{token}".encode("utf-8")).hexdigest()

    def _request_tts_owner_key(request: Request) -> str:
        return _tts_owner_key_for_token(request.cookies.get(AUTH_COOKIE_NAME))

    def _websocket_tts_owner_key(websocket: WebSocket) -> str:
        return _tts_owner_key_for_token(websocket.cookies.get(AUTH_COOKIE_NAME))

    def _validated_tts_client_id(client_id: str) -> str:
        value = str(client_id or "")
        if not TTS_CLIENT_ID_PATTERN.fullmatch(value):
            raise HTTPException(status_code=400, detail="invalid TTS client ID")
        return value

    def _set_auth_cookie(response: RedirectResponse, token: str) -> None:
        response.set_cookie(
            AUTH_COOKIE_NAME,
            token,
            max_age=int(auth.session_ttl_sec),
            httponly=True,
            secure=bool(auth.cookie_secure),
            samesite="lax",
            path="/",
        )

    def _clear_auth_cookie(response: RedirectResponse) -> None:
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")

    def _normalize_force_language(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text.lower() in {"auto", "none", "null", "default"}:
            return None
        return text

    def _context_terms_for(
        force_language: Optional[str],
        elapsed_sec: float,
        session_context_terms: Optional[Tuple[str, ...]],
    ) -> Tuple[str, ...]:
        if session_context_terms is not None:
            return tuple(session_context_terms)
        if asr_context_schedule is None:
            return ()
        return asr_context_schedule.terms_at(
            elapsed_sec,
            language=force_language,
            lookaround_sec=asr_context_lookaround_sec,
            max_terms=asr_context_max_terms,
            max_chars=asr_context_max_chars,
        )

    def _context_for(
        force_language: Optional[str],
        elapsed_sec: float,
        session_context_terms: Optional[Tuple[str, ...]],
    ) -> str:
        return " ".join(
            _context_terms_for(force_language, elapsed_sec, session_context_terms)
        )

    def _new_vllm_state(
        force_language: Optional[str],
        elapsed_sec: float = 0.0,
        session_context_terms: Optional[Tuple[str, ...]] = None,
    ):
        selected_context = _context_for(
            force_language,
            elapsed_sec,
            session_context_terms,
        )
        streaming_context = selected_context if asr_context_apply_mode == "streaming" else ""
        kwargs = dict(
            context=streaming_context,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_sec,
        )
        if force_language is not None:
            kwargs["language"] = force_language
        streaming_state = asr.init_streaming_state(**kwargs)
        setattr(
            streaming_state,
            "_voxbridge_final_context",
            selected_context if asr_context_apply_mode == "segment_final" else "",
        )
        setattr(streaming_state, "_voxbridge_context_elapsed_sec", float(elapsed_sec))
        return streaming_state

    def _new_transformers_state(
        force_language: Optional[str],
        elapsed_sec: float = 0.0,
        session_context_terms: Optional[Tuple[str, ...]] = None,
    ):
        selected_context = _context_for(
            force_language,
            elapsed_sec,
            session_context_terms,
        )
        return SimpleNamespace(
            audio_accum=np.zeros((0,), dtype=np.float32),
            language="",
            text="",
            force_language=force_language,
            streaming_context=(
                selected_context if asr_context_apply_mode == "streaming" else ""
            ),
            final_context=selected_context,
            min_decode_samples=max(1, int(round(float(args.min_audio_sec) * SAMPLE_RATE))),
            decode_interval_samples=max(1, int(round(float(args.decode_interval_sec) * SAMPLE_RATE))),
            last_decoded_samples=0,
        )

    @app.get("/login")
    async def login_page(request: Request):
        next_target = _safe_login_next(request.query_params.get("next"))
        if not auth.enabled:
            return RedirectResponse(next_target, status_code=303)
        if _request_is_authenticated(request):
            return RedirectResponse(next_target, status_code=303)
        return HTMLResponse(_render_login_html(next_target=next_target))

    @app.post("/login")
    async def login_submit(request: Request):
        if not auth.enabled:
            return RedirectResponse("/", status_code=303)
        body = (await request.body()).decode("utf-8", errors="replace")
        form = urllib.parse.parse_qs(body, keep_blank_values=True)
        username = str((form.get("username") or [""])[0] or "")
        password = str((form.get("password") or [""])[0] or "")
        next_target = _safe_login_next((form.get("next") or [""])[0])
        username_ok = hmac.compare_digest(username, str(auth.username))
        password_ok = _verify_auth_password(password, str(auth.password_hash))
        if not (username_ok and password_ok):
            return HTMLResponse(
                _render_login_html(error=True, next_target=next_target),
                status_code=401,
            )
        response = RedirectResponse(next_target, status_code=303)
        _set_auth_cookie(response, _new_auth_session())
        return response

    @app.post("/logout")
    async def logout(request: Request):
        _drop_auth_session(request.cookies.get(AUTH_COOKIE_NAME))
        response = RedirectResponse("/login", status_code=303)
        _clear_auth_cookie(response)
        return response

    @app.get("/")
    async def index(request: Request):
        if not _request_is_authenticated(request):
            return RedirectResponse("/login", status_code=303)
        subtitle_trace = bool(getattr(args, "subtitle_trace", False))
        subtitle_trace_max_events = max(200, int(getattr(args, "subtitle_trace_max_events", 1200)))
        html = INDEX_HTML_TEMPLATE.replace("__CHUNK_MS__", str(int(args.client_chunk_ms)))
        html = html.replace("__SUBTITLE_TRACE__", "true" if subtitle_trace else "false")
        html = html.replace("__SUBTITLE_TRACE_MAX_EVENTS__", str(subtitle_trace_max_events))
        html = html.replace("__ASR_CONTEXT_MAX_TERMS__", str(asr_context_max_terms))
        html = html.replace("__ASR_CONTEXT_MAX_CHARS__", str(asr_context_max_chars))
        return HTMLResponse(html)

    @app.get("/listen")
    async def tts_listener_page(request: Request):
        if not _request_is_authenticated(request):
            login_target = "/login?" + urllib.parse.urlencode({"next": "/listen"})
            return RedirectResponse(login_target, status_code=303)
        return HTMLResponse(TTS_LISTENER_HTML)

    @app.websocket("/ws/tts")
    async def ws_tts_listener(websocket: WebSocket) -> None:
        if not _websocket_is_authenticated(websocket):
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "unauthorized"})
            await websocket.close(code=1008)
            return

        owner_key = _websocket_tts_owner_key(websocket)
        subscription = app.state.tts_broadcast.register(owner_key)
        listener_hash = _opaque_identifier_hash8(subscription.listener_id)
        logger.info(
            "tts listener connected listener_hash=%s listeners=%d producer_active=%s",
            listener_hash,
            app.state.tts_broadcast.listener_count,
            app.state.tts_broadcast.producer_active,
        )
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "tts_listener_ready",
                "listener_id": subscription.listener_id,
                "tts_available": bool(app.state.tts_synthesizer is not None),
                "producer_active": bool(app.state.tts_broadcast.producer_active),
            }
        )

        async def _send_listener_events() -> None:
            while True:
                if subscription.overflowed.is_set():
                    logger.warning(
                        "tts listener overflow listener_hash=%s listeners=%d",
                        listener_hash,
                        app.state.tts_broadcast.listener_count,
                    )
                    await websocket.send_json(
                        {"type": "error", "message": "listener queue overloaded"}
                    )
                    await websocket.close(code=1013)
                    return
                try:
                    event = await asyncio.wait_for(subscription.queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                await websocket.send_json(event)

        sender = asyncio.create_task(_send_listener_events())
        try:
            while True:
                payload = _parse_json_message(await websocket.receive_text())
                message_type = str(payload.get("type", "") or "")
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                if message_type == "tts_received":
                    job_id = str(payload.get("job_id", "") or "")
                    acknowledged = app.state.tts_broadcast.acknowledge(
                        job_id,
                        subscription.listener_id,
                        owner_key,
                    )
                    logger.info(
                        "tts listener received listener_hash=%s job_hash=%s accepted=%s retained_jobs=%d",
                        listener_hash,
                        _opaque_identifier_hash8(job_id),
                        acknowledged,
                        app.state.tts_broadcast.job_count,
                    )
                    if not acknowledged:
                        await websocket.send_json(
                            {"type": "error", "message": "TTS job not found"}
                        )
                    continue
                await websocket.send_json(
                    {"type": "error", "message": "unsupported listener message"}
                )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await sender
            app.state.tts_broadcast.unregister(subscription.listener_id, owner_key)
            logger.info(
                "tts listener disconnected listener_hash=%s listeners=%d retained_jobs=%d",
                listener_hash,
                app.state.tts_broadcast.listener_count,
                app.state.tts_broadcast.job_count,
            )

    @app.post("/api/tts/broadcast/jobs/{job_id}/audio")
    async def tts_broadcast_audio(job_id: str, request: Request) -> Response:
        owner_key = _request_tts_owner_key(request)
        listener_id = str(request.headers.get("X-TTS-Listener-ID", "") or "")
        try:
            job = app.state.tts_broadcast.claim_audio(job_id, listener_id, owner_key)
        except TTSBroadcastNotFound as exc:
            raise HTTPException(status_code=404, detail="TTS job not found") from exc

        try:
            if app.state.tts_synthesizer is None:
                raise HTTPException(status_code=503, detail="TTS is unavailable")
            async with tts_synthesis_lock:
                try:
                    job = app.state.tts_broadcast.claimed_job(job_id)
                except TTSBroadcastNotFound as exc:
                    raise HTTPException(status_code=404, detail="TTS job not found") from exc
                if job.audio_bytes is None:
                    try:
                        audio = await asyncio.to_thread(
                            app.state.tts_synthesizer.synthesize,
                            job.text,
                            job.target_language,
                        )
                        job = app.state.tts_broadcast.cache_audio(
                            job.job_id,
                            audio.wav_bytes,
                            sample_rate=audio.sample_rate,
                            duration_ms=audio.duration_ms,
                        )
                    except TTSBroadcastNotFound as exc:
                        raise HTTPException(status_code=404, detail="TTS job not found") from exc
                    except Exception as exc:
                        logger.warning("Broadcast TTS synthesis failed: %s", type(exc).__name__)
                        raise HTTPException(status_code=503, detail="TTS synthesis failed") from exc

            headers = {
                "Cache-Control": "no-store",
                "X-TTS-Sample-Rate": str(job.sample_rate or ""),
                "X-TTS-Duration-Ms": str(job.duration_ms or 0),
            }
            return Response(
                content=job.audio_bytes or b"",
                media_type="audio/wav",
                headers=headers,
            )
        finally:
            app.state.tts_broadcast.release_audio(job_id)

    @app.post("/api/tts/jobs/{job_id}/audio")
    async def tts_audio(job_id: str, request: Request) -> Response:
        owner_key = _request_tts_owner_key(request)
        try:
            job = app.state.tts_jobs.get(job_id, owner_key)
        except TTSJobNotFound as exc:
            raise HTTPException(status_code=404, detail="TTS job not found") from exc
        if app.state.tts_synthesizer is None:
            raise HTTPException(status_code=503, detail="TTS is unavailable")

        async with tts_synthesis_lock:
            try:
                job = app.state.tts_jobs.get(job_id, owner_key)
            except TTSJobNotFound as exc:
                raise HTTPException(status_code=404, detail="TTS job not found") from exc
            if job.audio_bytes is None:
                try:
                    audio = await asyncio.to_thread(
                        app.state.tts_synthesizer.synthesize,
                        job.text,
                        job.target_language,
                    )
                    job = app.state.tts_jobs.cache_audio(
                        job.job_id,
                        owner_key,
                        audio.wav_bytes,
                        sample_rate=audio.sample_rate,
                        duration_ms=audio.duration_ms,
                    )
                except TTSJobNotFound as exc:
                    raise HTTPException(status_code=404, detail="TTS job not found") from exc
                except TTSSynthesisError as exc:
                    logger.warning("TTS synthesis failed job_id=%s", job.job_id)
                    raise HTTPException(status_code=502, detail="TTS synthesis failed") from exc
                except Exception as exc:
                    logger.exception("Unexpected TTS synthesis failure job_id=%s", job.job_id)
                    raise HTTPException(status_code=502, detail="TTS synthesis failed") from exc

        headers = {
            "Cache-Control": "no-store",
            "X-TTS-Sample-Rate": str(job.sample_rate or ""),
            "X-TTS-Duration-Ms": str(job.duration_ms or 0),
        }
        return Response(content=job.audio_bytes or b"", media_type="audio/wav", headers=headers)

    @app.delete("/api/tts/jobs/{job_id}")
    async def tts_ack(job_id: str, request: Request) -> JSONResponse:
        owner_key = _request_tts_owner_key(request)
        try:
            app.state.tts_jobs.get(job_id, owner_key)
        except TTSJobNotFound as exc:
            raise HTTPException(status_code=404, detail="TTS job not found") from exc
        removed = app.state.tts_jobs.acknowledge(job_id, owner_key)
        return JSONResponse({"ok": removed})

    @app.delete("/api/tts/clients/{client_id}/jobs")
    async def tts_cancel_client(client_id: str, request: Request) -> JSONResponse:
        owner_key = _request_tts_owner_key(request)
        validated_client_id = _validated_tts_client_id(client_id)
        removed = app.state.tts_jobs.cancel_client(owner_key, validated_client_id)
        return JSONResponse({"ok": True, "removed": removed})

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
    async def debug_file(request: Request, path: str) -> FileResponse:
        if auth.disable_debug_file:
            raise HTTPException(status_code=404, detail="debug file endpoint disabled")
        if not _request_is_authenticated(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        fp = _resolve_debug_file(path)
        return FileResponse(str(fp), media_type="application/octet-stream", filename=fp.name)

    @app.websocket("/ws")
    async def ws_stream(websocket: WebSocket) -> None:
        if not _websocket_is_authenticated(websocket):
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "unauthorized"})
            await websocket.close(code=1008)
            return

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
        session_context_terms: Optional[Tuple[str, ...]] = None
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
        hard_overflow_relief_sec = max(2.0, float(getattr(args, "backpressure_hard_relief_sec", 6.0)))
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
        audio_spill_generation: Optional[int] = None
        audio_spill_parts: List[np.ndarray] = []
        audio_spill_samples = 0
        full_audio_parts: List[np.ndarray] = []
        full_audio_samples = 0
        full_audio_overflow = False
        segment_final_context_applied = False
        send_lock = asyncio.Lock()
        tts_transition_lock = asyncio.Lock()
        state_lock = asyncio.Lock()
        stop_consumer = asyncio.Event()
        consumer_task: Optional[asyncio.Task] = None
        idle_commit_sec = max(3.0, float(getattr(args, "vad_force_cut_sec", 1.8)) + 2.7)
        early_translation_stable_sec = max(0.0, float(getattr(args, "early_translation_stable_sec", 0.8)))
        early_translation_stable_hits = max(1, int(getattr(args, "early_translation_stable_hits", 3)))
        early_translation_short_stable_sec = max(
            0.0,
            float(getattr(args, "early_translation_short_stable_sec", 1.2)),
        )
        early_translation_short_stable_hits = max(
            1,
            int(getattr(args, "early_translation_short_stable_hits", 4)),
        )
        early_translation_min_english_words = max(
            1,
            int(getattr(args, "early_translation_min_english_words", 6)),
        )
        early_translation_min_english_chars = max(
            1,
            int(getattr(args, "early_translation_min_english_chars", 32)),
        )
        last_text_snapshot = ""
        last_text_advance_at = time.monotonic()
        last_idle_commit_at = 0.0
        partial_reset_guard_min_prev_chars = 20
        partial_reset_guard_max_ratio = 0.6
        partial_reset_guard_release_hits = 2
        partial_reset_guard_max_hold_sec = 1.2
        vad_silence_trigger_ms = max(120.0, float(getattr(args, "vad_silence_sec", 0.9)) * 1000.0)
        vad_force_silence_ms = max(vad_silence_trigger_ms + 300.0, float(getattr(args, "vad_force_cut_sec", 1.8)) * 1000.0)
        vad_min_slice_ms = max(500.0, float(getattr(args, "vad_min_slice_sec", 4.0)) * 1000.0)
        vad_min_active_ms = max(200.0, float(getattr(args, "vad_min_active_sec", 1.2)) * 1000.0)
        vad_enter_snr_db = max(1.0, float(getattr(args, "backend_vad_enter_snr_db", 8.0)))
        vad_exit_snr_db = float(getattr(args, "backend_vad_exit_snr_db", 4.0))
        vad_exit_snr_db = max(0.2, min(vad_enter_snr_db - 0.5, vad_exit_snr_db))
        vad_frame_samples = max(80, int(SAMPLE_RATE * 0.02))
        text_stable_cut_ms = max(180.0, float(getattr(args, "backend_cut_stable_sec", 0.45)) * 1000.0)
        punct_cut_start_ms = max(0.0, float(getattr(args, "punct_cut_start_sec", 0.0)) * 1000.0)
        punct_cut_wait_ms = max(0.0, float(getattr(args, "punct_cut_wait_sec", 0.0)) * 1000.0)
        punct_cut_stable_ms = max(80.0, float(getattr(args, "punct_cut_stable_sec", 0.45)) * 1000.0)
        punct_cut_stable_hits = max(1, int(getattr(args, "punct_cut_stable_hits", 2)))
        punct_cut_max_carry_chars = max(0, int(getattr(args, "punct_cut_max_carry_chars", 12)))
        # Deprecated: punctuation-timeout cutting is disabled permanently to avoid
        # sentence loss/regression in streaming mode.
        punct_cut_enabled = False
        punct_cut_pattern = SENTENCE_BOUNDARY_PATTERN
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
        punct_cut_runtime = SimpleNamespace(
            gate_open=False,
            gate_chars=0,
            gate_open_at=0.0,
            candidate_end=0,
            candidate_token="",
            candidate_since=0.0,
            candidate_hits=0,
            anchor_end=0,
            anchor_token="",
            anchor_locked_at=0.0,
            anchor_seq=0,
            triggered=False,
        )
        backpressure_runtime = SimpleNamespace(
            under_pressure=False,
            reason="normal",
            hard_overflow_since=0.0,
            last_relief_at=0.0,
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
            queue_spill_flushes=0,
            queue_spill_samples_peak=0,
            last_error="",
            silent_decode_skipped=0,
        )
        subtitle_state = SimpleNamespace(
            stream_uid=f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}",
            next_sentence_id=1,
            committed_sentences=[],
            sentence_items=[],
            commit_base=0,
            processed_completed_count=0,
            candidate_sentence_ids=[],
            candidate_texts=[],
            deferred_sentence_upgrades={},
            prev_completed_sentences=[],
            tentative_tail="",
            pending_prefix_text="",
            pending_prefix_segment_id=0,
            pending_prefix_reason="",
            pending_prefix_miss_count=0,
            pending_prefix_terminal_text="",
            pending_prefix_is_separate=False,
            boundary_anchor_text="",
            boundary_anchor_segment_id=0,
            boundary_overlap_cap_chars=max(4, min(24, int(round(segment_overlap_sec * 14.0)))),
            duplicate_filter_pause_until=0.0,
            duplicate_filter_pause_reason="",
            early_holdback_text="",
            early_holdback_since=0.0,
            early_holdback_first_seen_ms=0,
            early_holdback_hits=0,
        )
        stream_text_state = SimpleNamespace(
            last_text="",
            accepted_text="",
            reset_candidate_text="",
            reset_candidate_hits=0,
            reset_candidate_since=0.0,
        )

        def _reset_early_translation_holdback_state() -> None:
            subtitle_state.early_holdback_text = ""
            subtitle_state.early_holdback_since = 0.0
            subtitle_state.early_holdback_first_seen_ms = 0
            subtitle_state.early_holdback_hits = 0

        def _clear_pending_prefix_boundary_evidence() -> None:
            subtitle_state.pending_prefix_terminal_text = ""
            subtitle_state.pending_prefix_is_separate = False

        def _reset_completed_candidate_cursor() -> None:
            subtitle_state.processed_completed_count = 0
            subtitle_state.candidate_sentence_ids = []
            subtitle_state.candidate_texts = []
            subtitle_state.deferred_sentence_upgrades.clear()

        def _record_completed_candidate(index: int, text: str, sentence_id: str = "") -> None:
            idx = max(0, int(index))
            while len(subtitle_state.candidate_sentence_ids) <= idx:
                subtitle_state.candidate_sentence_ids.append("")
                subtitle_state.candidate_texts.append("")
            subtitle_state.candidate_sentence_ids[idx] = str(sentence_id or "")
            subtitle_state.candidate_texts[idx] = str(text or "").strip()
            subtitle_state.processed_completed_count = max(
                int(getattr(subtitle_state, "processed_completed_count", 0) or 0),
                idx + 1,
            )

        def _find_sentence_item_index(sentence_id: str) -> Optional[int]:
            sid = str(sentence_id or "")
            if not sid:
                return None
            for item_idx, item in enumerate(subtitle_state.sentence_items):
                if str(item.get("id", "") or "") == sid:
                    return int(item_idx)
            return None

        def _stabilize_completed_prefix_with_cursor(completed: List[str]) -> Tuple[List[str], int]:
            merged = [str(seg or "").strip() for seg in list(completed or [])]
            expected = int(getattr(subtitle_state, "processed_completed_count", 0) or 0)
            candidate_texts = list(getattr(subtitle_state, "candidate_texts", []) or [])
            if expected <= len(merged):
                return merged, 0
            backfilled = 0
            for idx in range(len(merged), min(expected, len(candidate_texts))):
                text = str(candidate_texts[idx] or "").strip()
                if not text:
                    break
                merged.append(text)
                backfilled += 1
            return merged, int(backfilled)

        alignment_runtime = SimpleNamespace(
            model_seen={},
            committed_seen={},
            model_observed_events=0,
            committed_events=0,
        )
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

        def _canonical_tts_language(language: str) -> str:
            if _is_chinese_label(language):
                return "Chinese"
            if _is_english_label(language):
                return "English"
            return str(language or "")

        initial_translation_direction = _normalize_translation_direction("zh2en")
        initial_translation_source, initial_translation_target = _resolve_direction_languages(initial_translation_direction)
        translation_runtime = SimpleNamespace(
            task=None,
            parallelism=max(1, int(getattr(args, "translation_workers", 3))),
            queue=asyncio.Queue(maxsize=256),
            direction=initial_translation_direction,
            source_language=initial_translation_source,
            target_language=initial_translation_target,
            latest_by_sentence={},
        )
        tts_revision_stable_sec = max(
            0.0,
            float(getattr(args, "tts_revision_stable_sec", 3.0)),
        )
        tts_runtime = SimpleNamespace(
            available=bool(tts_synthesizer is not None and translator is not None),
            broadcast_enabled=bool(tts_synthesizer is not None and translator is not None),
            enabled=False,
            client_id="",
            generation=0,
            next_source_order=0,
            sentence_orders={},
            ordered=RevisionStableTTSBuffer(stable_sec=tts_revision_stable_sec),
            stability_task=None,
            stability_wake=asyncio.Event(),
            stability_stopping=False,
            last_wait_key=None,
            session_issued_job_count=0,
            owner_key=(
                "anonymous"
                if not auth.enabled
                else hashlib.sha256(
                    f"auth:{websocket.cookies.get(AUTH_COOKIE_NAME)}".encode("utf-8")
                ).hexdigest()
            ),
        )
        subtitle_trace_log = bool(getattr(args, "subtitle_trace_log", False))
        subtitle_trace_log_partial_every = max(1, int(getattr(args, "subtitle_trace_log_partial_every", 20)))
        subtitle_trace_log_file = str(getattr(args, "subtitle_trace_log_file", "") or "").strip()
        trace_file_handle = None
        if subtitle_trace_log and subtitle_trace_log_file:
            try:
                trace_log_path = Path(subtitle_trace_log_file).expanduser()
                trace_log_path.parent.mkdir(parents=True, exist_ok=True)
                trace_file_handle = trace_log_path.open("a", encoding="utf-8", buffering=1)
                logger.info("subtitle trace file enabled peer=%s path=%s", peer, trace_log_path)
            except Exception as exc:
                logger.warning("subtitle trace file disabled peer=%s path=%s error=%s", peer, subtitle_trace_log_file, exc)
                trace_file_handle = None
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

        def _write_trace_file_row(row: Dict[str, Any]) -> None:
            if trace_file_handle is None:
                return
            try:
                trace_file_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception:
                pass

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
            _write_trace_file_row(row)
            try:
                logger.info("subtitle_trace %s", json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logger.info("subtitle_trace %s", row)

        def _reset_deferred_sentence_upgrade(
            sentence_id: str,
            *,
            seq_no: int,
            reason: str,
            replacement_text: str = "",
        ) -> None:
            sid = str(sentence_id or "")
            previous = subtitle_state.deferred_sentence_upgrades.pop(sid, None)
            if previous is None:
                return
            _trace_event(
                "sentence_upgrade_candidate_reset",
                seq=int(seq_no),
                sentence_id=sid,
                reason=str(reason or ""),
                candidate_hash8=_hash8(previous.text),
                candidate_chars=len(previous.text),
                replacement_hash8=_hash8(replacement_text),
                replacement_chars=len(str(replacement_text or "").strip()),
            )

        def _trace_asr_context(
            force_language: Optional[str],
            elapsed_sec: float,
            context_terms: Optional[Tuple[str, ...]],
        ) -> None:
            selected_terms = _context_terms_for(
                force_language,
                elapsed_sec,
                context_terms,
            )
            context = " ".join(selected_terms)
            _trace_event(
                "asr_context_selected",
                audio_elapsed_ms=int(max(0.0, float(elapsed_sec)) * 1000.0),
                language=str(force_language or "auto"),
                apply_mode=str(asr_context_apply_mode),
                context_source="session" if context_terms is not None else "schedule",
                context_active=bool(selected_terms),
                streaming_context=bool(context and asr_context_apply_mode == "streaming"),
                term_count=len(selected_terms),
                context_chars=len(context),
                context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
            )

        def _guard_streaming_context_output(local_state: Any, *, reason: str) -> int:
            if asr_context_apply_mode != "streaming":
                return 0
            context = str(getattr(local_state, "context", "") or "")
            current_text = str(getattr(local_state, "text", "") or "").strip()
            previous_text = str(
                getattr(local_state, "_voxbridge_context_guard_text", "") or ""
            ).strip()
            filtered_text, removed = _filter_asr_context_echo_sentences(
                context,
                current_text,
                previous_text=previous_text,
            )
            if removed <= 0:
                local_state._voxbridge_context_guard_text = current_text
                return 0
            local_state.text = filtered_text
            local_state._voxbridge_context_guard_text = filtered_text
            if getattr(local_state, "force_language", None) is not None and hasattr(local_state, "_raw_decoded"):
                local_state._raw_decoded = filtered_text
            _trace_event(
                "asr_context_streaming_echo_filtered",
                reason=str(reason or ""),
                removed_sentences=int(removed),
                previous_text_chars=len(current_text),
                filtered_text_chars=len(filtered_text),
                context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
            )
            return int(removed)

        async def _apply_segment_final_context(local_state: Any, *, reason: str) -> bool:
            context = str(getattr(local_state, "_voxbridge_final_context", "") or "")
            if asr_context_apply_mode != "segment_final" or not context:
                return False

            segment_wav = np.asarray(
                getattr(local_state, "audio_accum", np.zeros((0,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if int(segment_wav.size) <= 0:
                _trace_event(
                    "asr_context_final_redecode_skipped",
                    reason="empty_audio",
                    cut_reason=str(reason or ""),
                    context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                )
                return False

            previous_text = str(getattr(local_state, "text", "") or "").strip()
            saved_max_tokens = None
            sampling_params = getattr(asr, "sampling_params", None)
            override_max_tokens = int(max(1, int(getattr(args, "final_redecode_max_new_tokens", 512))))
            try:
                if sampling_params is not None and hasattr(sampling_params, "max_tokens"):
                    try:
                        saved_max_tokens = int(getattr(sampling_params, "max_tokens"))
                    except Exception:
                        saved_max_tokens = None
                    if saved_max_tokens is None or saved_max_tokens < override_max_tokens:
                        setattr(sampling_params, "max_tokens", override_max_tokens)

                segment_out = await _await_thread_completion_on_cancel(
                    lambda: asr.transcribe(
                        audio=[(segment_wav.copy(), SAMPLE_RATE)],
                        context=context,
                        language=(getattr(local_state, "force_language", None) or session_force_language),
                    )[0]
                )
                corrected_text = str(getattr(segment_out, "text", "") or "").strip()
                if not corrected_text:
                    _trace_event(
                        "asr_context_final_redecode_skipped",
                        reason="empty_result",
                        cut_reason=str(reason or ""),
                        audio_samples=int(segment_wav.size),
                        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    )
                    return False
                if _compact_asr_compare_text(corrected_text) == _compact_asr_compare_text(
                    previous_text
                ):
                    _trace_event(
                        "asr_context_final_redecode_skipped",
                        reason="unchanged_result",
                        cut_reason=str(reason or ""),
                        audio_samples=int(segment_wav.size),
                        corrected_text_chars=len(corrected_text),
                        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    )
                    return False
                if _looks_like_asr_context_echo(
                    context,
                    corrected_text,
                    previous_text=previous_text,
                ):
                    _trace_event(
                        "asr_context_final_redecode_skipped",
                        reason="context_echo",
                        cut_reason=str(reason or ""),
                        audio_samples=int(segment_wav.size),
                        corrected_text_chars=len(corrected_text),
                        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    )
                    return False

                local_state.language = (
                    getattr(segment_out, "language", "")
                    or getattr(local_state, "language", "")
                    or ""
                )
                local_state.text = corrected_text
                _trace_event(
                    "asr_context_final_redecode_done",
                    reason=str(reason or ""),
                    audio_samples=int(segment_wav.size),
                    audio_ms=int(round(int(segment_wav.size) * 1000.0 / SAMPLE_RATE)),
                    previous_text_chars=len(previous_text),
                    corrected_text_chars=len(corrected_text),
                    text_changed=bool(corrected_text != previous_text),
                    context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                )
                return True
            except Exception as exc:
                _trace_event(
                    "asr_context_final_redecode_failed",
                    reason=str(reason or ""),
                    audio_samples=int(segment_wav.size),
                    context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    **_safe_exception_trace_fields(exc),
                )
                return False
            finally:
                if saved_max_tokens is not None:
                    with suppress(Exception):
                        setattr(asr.sampling_params, "max_tokens", saved_max_tokens)

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
            _write_trace_file_row(row)
            try:
                logger.info("text_pool %s", json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logger.info("text_pool %s", row)

        def _sentence_signature_rows(
            sentences: List[str],
            *,
            keep_head: int = 2,
            keep_tail: int = 3,
        ) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            cleaned = [str(seg or "").strip() for seg in list(sentences or [])]
            total = len(cleaned)
            if total <= 0:
                return rows
            head_n = max(0, int(keep_head))
            tail_n = max(0, int(keep_tail))
            indexes: List[int] = []
            if total <= (head_n + tail_n):
                indexes = list(range(total))
            else:
                indexes.extend(range(head_n))
                indexes.extend(range(max(head_n, total - tail_n), total))
            seen = set()
            for idx in indexes:
                if idx in seen or idx < 0 or idx >= total:
                    continue
                seen.add(idx)
                text = str(cleaned[idx] or "")
                rows.append(
                    {
                        "idx": int(idx),
                        "chars": len(text),
                        "hash8": _hash8(text),
                        "preview": _trace_preview(text, 64),
                    }
                )
            return rows

        def _count_matching_sentence_prefix(prev: List[str], cur: List[str]) -> int:
            if not prev or not cur:
                return 0
            limit = min(len(prev), len(cur))
            i = 0
            while i < limit:
                if _normalize_sentence_for_duplicate_compare(str(prev[i] or "")) != _normalize_sentence_for_duplicate_compare(str(cur[i] or "")):
                    break
                i += 1
            return int(i)

        def _track_model_observed_sentences(text: str, seq_hint: int, source: str) -> None:
            snapshot = str(text or "").strip()
            if not snapshot:
                return
            completed, _ = _split_sentences_and_tail(snapshot)
            if not completed:
                return
            new_unique = 0
            for sentence in completed:
                _, created, _ = _alignment_registry_touch_model(
                    alignment_runtime.model_seen,
                    sentence,
                    int(seq_hint or 0),
                    str(source or ""),
                )
                if created:
                    new_unique += 1
            alignment_runtime.model_observed_events = int(getattr(alignment_runtime, "model_observed_events", 0) or 0) + 1
            if new_unique > 0:
                _trace_event(
                    "alignment_model_observed",
                    seq=int(seq_hint or 0),
                    source=str(source or ""),
                    completed_count=int(len(completed)),
                    new_unique=int(new_unique),
                    model_unique_total=int(len(alignment_runtime.model_seen)),
                )

        def _track_committed_sentence(sentence: str, seq_hint: int, source: str, sentence_id: str) -> None:
            key, created, committed_entry = _alignment_registry_touch(
                alignment_runtime.committed_seen,
                sentence,
                int(seq_hint or 0),
                str(source or ""),
            )
            if not key:
                return
            alignment_runtime.committed_events = int(getattr(alignment_runtime, "committed_events", 0) or 0) + 1
            model_entry = alignment_runtime.model_seen.get(key)
            if model_entry is None:
                _trace_event(
                    "alignment_commit_without_model_observation",
                    seq=int(seq_hint or 0),
                    source=str(source or ""),
                    sentence_id=str(sentence_id or ""),
                    chars=len(str(sentence or "").strip()),
                    preview=_trace_preview(str(sentence or ""), 72),
                    committed_unique_total=int(len(alignment_runtime.committed_seen)),
                )
                return
            if created:
                _trace_event(
                    "alignment_commit_matched_model",
                    seq=int(seq_hint or 0),
                    source=str(source or ""),
                    sentence_id=str(sentence_id or ""),
                    model_hits=int(model_entry.get("hits", 0) or 0),
                    first_model_seq=int(model_entry.get("first_seq", 0) or 0),
                    committed_unique_total=int(len(alignment_runtime.committed_seen)),
                    model_unique_total=int(len(alignment_runtime.model_seen)),
                    committed_hits=int(committed_entry.get("hits", 0) or 0),
                )

        def _emit_alignment_summary(reason: str, seq_hint: int) -> None:
            summary = _summarize_alignment_gap(
                alignment_runtime.model_seen,
                alignment_runtime.committed_seen,
                min_model_hits=2,
                max_samples=6,
            )
            samples = []
            for row in list(summary.get("missing_samples", []) or []):
                src = str(row.get("text", "") or "")
                samples.append(
                    {
                        "preview": _trace_preview(src, 72),
                        "chars": len(src),
                        "hits": int(row.get("hits", 0) or 0),
                        "first_seq": int(row.get("first_seq", 0) or 0),
                        "last_seq": int(row.get("last_seq", 0) or 0),
                        "sources": list(row.get("sources", []) or []),
                    }
                )
            final_samples = []
            for row in list(summary.get("final_missing_samples", []) or []):
                src = str(row.get("text", "") or "")
                final_samples.append(
                    {
                        "preview": _trace_preview(src, 72),
                        "chars": len(src),
                        "hits": int(row.get("hits", 0) or 0),
                        "first_seq": int(row.get("first_seq", 0) or 0),
                        "last_seq": int(row.get("last_seq", 0) or 0),
                        "sources": list(row.get("sources", []) or []),
                    }
                )
            _trace_event(
                "alignment_summary",
                seq=int(seq_hint or 0),
                reason=str(reason or ""),
                model_all_unique=int(summary.get("model_all_unique", 0) or 0),
                model_stable_unique=int(summary.get("model_stable_unique", 0) or 0),
                model_final_unique=int(summary.get("model_final_unique", 0) or 0),
                committed_unique=int(summary.get("committed_unique", 0) or 0),
                missing_unique=int(summary.get("missing_unique", 0) or 0),
                missing_samples=samples,
                final_missing_unique=int(summary.get("final_missing_unique", 0) or 0),
                final_missing_samples=final_samples,
                model_observed_events=int(getattr(alignment_runtime, "model_observed_events", 0) or 0),
                committed_events=int(getattr(alignment_runtime, "committed_events", 0) or 0),
            )

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

        async def _reset_tts_ordering() -> None:
            async with tts_transition_lock:
                tts_runtime.generation += 1
                tts_runtime.next_source_order = 0
                tts_runtime.sentence_orders.clear()
                tts_runtime.ordered.reset()
                tts_runtime.last_wait_key = None
                tts_runtime.stability_wake.set()

        def _tts_output_active() -> bool:
            return bool(
                not tts_runtime.stability_stopping
                and (tts_runtime.broadcast_enabled or tts_runtime.enabled)
            )

        def _set_tts_producer_active(active: bool) -> None:
            if app.state.tts_broadcast.set_producer_active(active):
                _trace_event("tts_producer_status", active=bool(active))

        async def _emit_tts_status(status: str, **payload: Any) -> None:
            message = {
                "type": "tts_status",
                "status": str(status or ""),
                "tts_available": bool(tts_runtime.available),
                "tts_enabled": bool(tts_runtime.enabled),
            }
            if payload:
                message.update(payload)
            await _send_json(message)

        async def _configure_tts(
            enabled: bool,
            client_id: str,
            *,
            emit: bool,
            force_reset: bool = False,
        ) -> bool:
            requested_enabled = bool(enabled)
            requested_client_id = str(client_id or tts_runtime.client_id or "")
            if requested_enabled:
                requested_client_id = _validated_tts_client_id(requested_client_id)
            elif requested_client_id:
                requested_client_id = _validated_tts_client_id(requested_client_id)

            previous_enabled = bool(tts_runtime.enabled)
            previous_client_id = str(tts_runtime.client_id or "")
            if requested_enabled and not tts_runtime.available:
                requested_enabled = False

            changed = (
                requested_enabled != previous_enabled
                or requested_client_id != previous_client_id
            )
            if force_reset or changed:
                await _reset_tts_ordering()

            if previous_enabled and (
                not requested_enabled or requested_client_id != previous_client_id
            ):
                app.state.tts_jobs.cancel_client(
                    str(tts_runtime.owner_key),
                    previous_client_id,
                )

            tts_runtime.enabled = requested_enabled
            tts_runtime.client_id = requested_client_id
            status = "enabled" if requested_enabled else "disabled"
            if bool(enabled) and not tts_runtime.available:
                status = "unavailable"
            if emit:
                await _emit_tts_status(status)
            _trace_event(
                "tts_configured",
                status=status,
                enabled=bool(tts_runtime.enabled),
                available=bool(tts_runtime.available),
                changed=bool(changed),
                generation=int(tts_runtime.generation),
            )
            return bool(tts_runtime.enabled)

        async def _register_tts_source(sentence_id: str, revision: int) -> None:
            if not _tts_output_active():
                return
            sid = str(sentence_id or "")
            if not sid:
                return
            async with tts_transition_lock:
                generation = int(tts_runtime.generation)
                registered = tts_runtime.sentence_orders.get(sid)
                if registered is None:
                    source_order = int(tts_runtime.next_source_order)
                    tts_runtime.next_source_order += 1
                    tts_runtime.sentence_orders[sid] = (source_order, generation)
                else:
                    source_order, registered_generation = registered
                    if int(registered_generation) != generation:
                        return
                registration = tts_runtime.ordered.register(
                    sid,
                    int(revision),
                    int(source_order),
                )
                if registration.accepted:
                    tts_runtime.stability_wake.set()
                if registration.reset:
                    tts_runtime.last_wait_key = None
                    _trace_event(
                        "tts_stability_reset",
                        sentence_hash8=_opaque_identifier_hash8(sid),
                        source_order=int(registration.source_order),
                        previous_revision=int(registration.previous_revision or 0),
                        new_revision=int(registration.revision),
                        previous_quiet_age_ms=int(registration.previous_quiet_age_ms),
                        previous_ready=bool(registration.previous_ready),
                    )
                elif registration.late_after_release:
                    _trace_event(
                        "tts_late_revision_after_release",
                        sentence_hash8=_opaque_identifier_hash8(sid),
                        source_order=int(registration.source_order),
                        released_revision=int(registration.released_revision or 0),
                        incoming_revision=int(registration.revision),
                        elapsed_since_release_ms=int(registration.elapsed_since_release_ms),
                    )
            _trace_event(
                "tts_source_registered",
                sentence_id=sid,
                revision=int(revision),
                source_order=int(source_order),
                generation=generation,
            )

        async def _publish_tts_ready(items: List[Any]) -> None:
            for item in items:
                if not _tts_output_active():
                    return
                tts_runtime.last_wait_key = None
                _trace_event(
                    "tts_stability_release",
                    sentence_hash8=_opaque_identifier_hash8(str(item.sentence_id)),
                    revision=int(item.revision),
                    source_order=int(item.source_order),
                    release_reason=str(item.release_reason),
                    source_quiet_age_ms=int(item.source_quiet_age_ms),
                    translation_ready_age_ms=int(item.translation_ready_age_ms),
                    ordered_backlog_depth=int(tts_runtime.ordered.pending_count),
                )
                broadcast_job = None
                if tts_runtime.broadcast_enabled:
                    pruned_jobs = app.state.tts_broadcast.prune()
                    if pruned_jobs:
                        logger.info(
                            "tts broadcast pruned jobs=%d retained_jobs=%d listeners=%d",
                            pruned_jobs,
                            app.state.tts_broadcast.job_count,
                            app.state.tts_broadcast.listener_count,
                        )
                    try:
                        broadcast_job = app.state.tts_broadcast.publish(item)
                    except TTSBroadcastQueueFull:
                        logger.warning(
                            "tts broadcast queue full retained_jobs=%d listeners=%d",
                            app.state.tts_broadcast.job_count,
                            app.state.tts_broadcast.listener_count,
                        )
                        _trace_event(
                            "tts_broadcast_rejected",
                            reason="queue_full",
                            sentence_id=str(item.sentence_id),
                            revision=int(item.revision),
                            source_order=int(item.source_order),
                        )
                    if broadcast_job is not None:
                        logger.info(
                            "tts broadcast published job_hash=%s source_order=%d listeners=%d retained_jobs=%d",
                            _opaque_identifier_hash8(broadcast_job.job_id),
                            int(broadcast_job.source_order),
                            app.state.tts_broadcast.listener_count,
                            app.state.tts_broadcast.job_count,
                        )

                private_job = None
                if tts_runtime.enabled:
                    try:
                        private_job = app.state.tts_jobs.create(
                            owner_key=str(tts_runtime.owner_key),
                            client_id=str(tts_runtime.client_id),
                            sentence_id=str(item.sentence_id),
                            revision=int(item.revision),
                            source_order=int(item.source_order),
                            target_language=str(item.target_language),
                            text=str(item.text),
                        )
                    except TTSQueueFull:
                        _trace_event(
                            "tts_job_rejected",
                            reason="queue_full",
                            sentence_id=str(item.sentence_id),
                            revision=int(item.revision),
                            source_order=int(item.source_order),
                        )
                        await _emit_tts_status("queue_full")
                    else:
                        await _send_json(
                            {
                                "type": "tts_job",
                                "job_id": private_job.job_id,
                                "sentence_id": private_job.sentence_id,
                                "revision": int(private_job.revision),
                                "source_order": int(private_job.source_order),
                                "target_language": private_job.target_language,
                                "is_stable": True,
                            }
                        )

                if broadcast_job is None and private_job is None:
                    continue
                tts_runtime.session_issued_job_count += 1
                _trace_event(
                    "tts_job_issued",
                    sentence_id=str(item.sentence_id),
                    revision=int(item.revision),
                    source_order=int(item.source_order),
                    target_language=str(item.target_language),
                    translated_chars=len(str(item.text)),
                    translated_hash8=_hash8(str(item.text)),
                    broadcast=bool(broadcast_job is not None),
                    private=bool(private_job is not None),
                )

        async def _mark_tts_translation_ready(
            sentence_id: str,
            revision: int,
            translated: str,
            target_language: str,
        ) -> None:
            def _transition() -> List[Any]:
                registered = tts_runtime.sentence_orders.get(str(sentence_id or ""))
                if not _tts_output_active() or registered is None:
                    return []
                _, registered_generation = registered
                if int(registered_generation) != int(tts_runtime.generation):
                    return []
                accepted = tts_runtime.ordered.mark_ready(
                    str(sentence_id),
                    int(revision),
                    str(translated),
                    _canonical_tts_language(str(target_language)),
                )
                if accepted:
                    tts_runtime.stability_wake.set()
                    wait = tts_runtime.ordered.wait_state(str(sentence_id))
                    if wait is not None and (
                        int(wait.remaining_ms) > 0 or bool(wait.blocked_by_earlier)
                    ):
                        wait_key = (
                            str(sentence_id),
                            int(wait.revision),
                            bool(wait.blocked_by_earlier),
                        )
                        if wait_key != tts_runtime.last_wait_key:
                            tts_runtime.last_wait_key = wait_key
                            _trace_event(
                                "tts_stability_wait",
                                sentence_hash8=_opaque_identifier_hash8(str(sentence_id)),
                                revision=int(wait.revision),
                                source_order=int(wait.source_order),
                                quiet_age_ms=int(wait.quiet_age_ms),
                                required_quiet_ms=int(wait.required_quiet_ms),
                                remaining_ms=int(wait.remaining_ms),
                                blocked_by_earlier=bool(wait.blocked_by_earlier),
                            )
                return tts_runtime.ordered.drain()

            await _run_ordered_tts_transition(
                tts_transition_lock,
                _transition,
                _publish_tts_ready,
            )

        async def _mark_tts_translation_failed(sentence_id: str, revision: int) -> None:
            def _transition() -> List[Any]:
                registered = tts_runtime.sentence_orders.get(str(sentence_id or ""))
                if not _tts_output_active() or registered is None:
                    return []
                _, registered_generation = registered
                if int(registered_generation) != int(tts_runtime.generation):
                    return []
                accepted = tts_runtime.ordered.mark_failed(str(sentence_id), int(revision))
                if accepted:
                    tts_runtime.stability_wake.set()
                _trace_event(
                    "tts_translation_skipped",
                    sentence_id=str(sentence_id or ""),
                    revision=int(revision),
                )
                return tts_runtime.ordered.drain()

            await _run_ordered_tts_transition(
                tts_transition_lock,
                _transition,
                _publish_tts_ready,
            )

        async def _drain_tts_stability(*, force: bool = False) -> None:
            async with tts_transition_lock:
                ready = tts_runtime.ordered.drain(force=force)
                if ready:
                    await _publish_tts_ready(ready)

        async def _tts_stability_scheduler() -> None:
            _trace_event(
                "tts_stability_scheduler_started",
                generation=int(tts_runtime.generation),
                pending_count=int(tts_runtime.ordered.pending_count),
            )
            try:
                while not bool(tts_runtime.stability_stopping):
                    tts_runtime.stability_wake.clear()
                    await _drain_tts_stability()
                    async with tts_transition_lock:
                        deadline = tts_runtime.ordered.next_deadline()
                    if deadline is None:
                        await tts_runtime.stability_wake.wait()
                        continue
                    timeout = max(0.0, float(deadline) - time.monotonic())
                    try:
                        await asyncio.wait_for(
                            tts_runtime.stability_wake.wait(),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                _trace_event(
                    "tts_stability_scheduler_cancelled",
                    generation=int(tts_runtime.generation),
                    pending_count=int(tts_runtime.ordered.pending_count),
                )
                raise
            except Exception as exc:
                tts_runtime.stability_stopping = True
                _trace_event(
                    "tts_stability_scheduler_failed",
                    error_type=type(exc).__name__,
                    generation=int(tts_runtime.generation),
                    pending_count=int(tts_runtime.ordered.pending_count),
                )
                logger.warning(
                    "TTS stability scheduler failed peer=%s error_type=%s pending=%d",
                    peer,
                    type(exc).__name__,
                    int(tts_runtime.ordered.pending_count),
                )
                with suppress(Exception):
                    await _emit_tts_status(
                        "unavailable",
                        reason="stability_scheduler_failed",
                    )

        tts_runtime.stability_task = asyncio.create_task(_tts_stability_scheduler())

        async def _stop_tts_stability_scheduler(*, reason: str) -> None:
            tts_runtime.stability_stopping = True
            tts_runtime.stability_wake.set()
            task = tts_runtime.stability_task
            _trace_event(
                "tts_stability_scheduler_stop",
                reason=str(reason or ""),
                generation=int(tts_runtime.generation),
                pending_count=int(tts_runtime.ordered.pending_count),
            )
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await task
            tts_runtime.stability_task = None

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
                translation_runtime.latest_by_sentence.clear()
                if changed and _tts_output_active():
                    if tts_runtime.enabled:
                        app.state.tts_jobs.cancel_client(
                            str(tts_runtime.owner_key),
                            str(tts_runtime.client_id),
                        )
                    await _reset_tts_ordering()

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
            *,
            sentence_id: str,
            revision: int,
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
                    sentence_id=str(sentence_id or ""),
                    revision=int(revision),
                    src_chars=len(src),
                    src_hash8=_hash8(src),
                )

            t0 = time.monotonic()
            try:
                try:
                    translate_kwargs = {
                        "source_language": effective_source_language,
                        "target_language": target_language,
                    }
                    if translator_accepts_direction:
                        translate_kwargs["translation_direction"] = direction
                    out = await asyncio.to_thread(
                        translator.translate,
                        src,
                        **translate_kwargs,
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
                    sentence_id=str(sentence_id or ""),
                    revision=int(revision),
                    src_chars=len(src),
                    src_hash8=_hash8(src),
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
                sentence_id=str(sentence_id or ""),
                revision=int(revision),
                src_chars=len(src),
                src_hash8=_hash8(src),
                out_chars=len(out or ""),
                out_hash8=_hash8(str(out or "")),
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

        def _mark_latest_translation_request(
            sentence_id: str,
            revision: int,
            sentence_text: str,
            direction: str,
        ) -> int:
            state_gen_hint = int(state_generation)
            translation_runtime.latest_by_sentence[str(sentence_id)] = (
                state_gen_hint,
                int(revision),
                str(sentence_text or "").strip(),
                str(direction or ""),
            )
            return state_gen_hint

        def _current_translation_item(
            sentence_id: str,
            revision: int,
            sentence_text: str,
            state_gen_hint: int,
            direction: str,
            *,
            phase: str,
        ) -> Optional[Dict[str, Any]]:
            sid = str(sentence_id or "")
            src = str(sentence_text or "").strip()
            queued_revision = int(revision)
            queued_generation = int(state_gen_hint)
            queued_direction = str(direction or "")
            item_idx = _find_sentence_item_index(sid)
            current_item = (
                subtitle_state.sentence_items[item_idx]
                if item_idx is not None and item_idx < len(subtitle_state.sentence_items)
                else None
            )
            current_revision = int(current_item.get("revision", 0) or 0) if current_item else 0
            current_text = str(current_item.get("zh", "") or "").strip() if current_item else ""
            current_direction = str(getattr(translation_runtime, "direction", "") or "")
            latest = translation_runtime.latest_by_sentence.get(sid)
            expected = (queued_generation, queued_revision, src, queued_direction)
            is_current = bool(
                current_item is not None
                and queued_generation == int(state_generation)
                and current_revision == queued_revision
                and current_text == src
                and current_direction == queued_direction
                and latest == expected
            )
            if is_current:
                return current_item

            latest_generation = int(latest[0]) if isinstance(latest, tuple) and len(latest) >= 1 else -1
            latest_revision = int(latest[1]) if isinstance(latest, tuple) and len(latest) >= 2 else 0
            latest_text = str(latest[2] or "") if isinstance(latest, tuple) and len(latest) >= 3 else ""
            latest_direction = str(latest[3] or "") if isinstance(latest, tuple) and len(latest) >= 4 else ""
            _trace_event(
                "translation_stale_drop",
                sentence_id=sid,
                queued_revision=queued_revision,
                current_revision=current_revision,
                queued_generation=queued_generation,
                current_generation=int(state_generation),
                queued_direction=queued_direction,
                current_direction=current_direction,
                latest_generation=latest_generation,
                latest_revision=latest_revision,
                latest_direction=latest_direction,
                phase=str(phase or ""),
                src_chars=len(src),
                src_hash8=_hash8(src),
                current_src_chars=len(current_text),
                current_src_hash8=_hash8(current_text),
                latest_src_chars=len(latest_text),
                latest_src_hash8=_hash8(latest_text),
            )
            return None

        async def _translate_and_publish_current(
            sentence_id: str,
            revision: int,
            sentence_text: str,
            language: str,
            seq_hint: int,
            state_gen_hint: int,
            source_language: str,
            target_language: str,
            direction: str,
        ) -> str:
            current_item = _current_translation_item(
                sentence_id,
                revision,
                sentence_text,
                state_gen_hint,
                direction,
                phase="pre_inference",
            )
            if current_item is None:
                return ""
            translated = await _translate_sentence_once(
                sentence_text,
                language,
                int(seq_hint or 0),
                str(source_language or ""),
                str(target_language or ""),
                str(direction or ""),
                sentence_id=str(sentence_id or ""),
                revision=int(revision),
            )
            if not translated:
                current_item = _current_translation_item(
                    sentence_id,
                    revision,
                    sentence_text,
                    state_gen_hint,
                    direction,
                    phase="empty_result",
                )
                if current_item is not None:
                    await _mark_tts_translation_failed(sentence_id, revision)
                return ""
            current_item = _current_translation_item(
                sentence_id,
                revision,
                sentence_text,
                state_gen_hint,
                direction,
                phase="post_inference",
            )
            if current_item is None:
                return ""

            current_item["en"] = translated
            _trace_text_pool(
                "pool_translation_done",
                phase="solidified",
                text=translated,
                reason="translation",
                seq_hint=int(seq_hint or 0),
                sentence_id=str(sentence_id),
                revision=int(revision),
                source_chars=len(str(sentence_text or "").strip()),
                source_hash8=_hash8(str(sentence_text or "")),
                delta_chars=len(translated),
            )
            await _send_json(
                {
                    "type": "sentence_translation",
                    "sentence_id": str(sentence_id),
                    "revision": int(revision),
                    "translation": translated,
                    "seq": int(seq_hint or 0),
                    "is_stable": True,
                }
            )
            await _mark_tts_translation_ready(
                sentence_id,
                revision,
                translated,
                target_language,
            )
            return translated

        async def _translation_worker() -> None:
            async def _translate_one(
                sentence_id: str,
                revision: int,
                sentence_text: str,
                language: str,
                seq_hint: int,
                state_gen_hint: int,
                source_language: str,
                target_language: str,
                direction: str,
            ) -> None:
                await _translate_and_publish_current(
                    sentence_id,
                    revision,
                    sentence_text,
                    language,
                    int(seq_hint or 0),
                    int(state_gen_hint or 0),
                    str(source_language or ""),
                    str(target_language or ""),
                    str(direction or ""),
                )

            active_tasks = set()
            try:
                while True:
                    try:
                        (
                            sentence_id,
                            revision,
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
                            int(revision),
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

        async def _drain_tts_translation_task(phase: str) -> None:
            translation_task = translation_runtime.task
            if translation_task is None or translation_task.done():
                return
            warning_sec = max(
                0.1,
                float(getattr(args, "tts_final_translation_drain_sec", 30.0)),
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(translation_task),
                    timeout=warning_sec,
                )
            except asyncio.TimeoutError:
                _trace_event(
                    "tts_final_translation_drain_timeout",
                    phase=str(phase or ""),
                    timeout_ms=int(warning_sec * 1000),
                    queue_depth=int(translation_runtime.queue.qsize()),
                )
                await _emit_tts_status(
                    "translation_drain_timeout",
                    phase=str(phase or ""),
                    timeout_ms=int(warning_sec * 1000),
                )
                with suppress(Exception):
                    await asyncio.shield(translation_task)
            except Exception:
                pass

        def _request_sentence_translation(
            sentence_id: str,
            revision: int,
            sentence_text: str,
            language: str,
            seq_hint: int,
        ) -> None:
            if translator is None:
                return
            src = str(sentence_text or "").strip()
            if not src:
                return
            direction = str(getattr(translation_runtime, "direction", "zh2en") or "zh2en")
            source_language = str(getattr(translation_runtime, "source_language", "") or "")
            target_language = str(getattr(translation_runtime, "target_language", "") or "")
            state_gen_hint = _mark_latest_translation_request(
                sentence_id,
                revision,
                src,
                direction,
            )

            item = (
                str(sentence_id),
                int(revision),
                src,
                str(language or ""),
                int(seq_hint or 0),
                int(state_gen_hint),
                source_language,
                target_language,
                direction,
            )
            try:
                translation_runtime.queue.put_nowait(item)
            except asyncio.QueueFull:
                dropped_item = None
                with suppress(asyncio.QueueEmpty):
                    dropped_item = translation_runtime.queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    translation_runtime.queue.put_nowait(item)
                if isinstance(dropped_item, tuple) and len(dropped_item) >= 2:
                    asyncio.create_task(
                        _mark_tts_translation_failed(
                            str(dropped_item[0]),
                            int(dropped_item[1]),
                        )
                    )
            _trace_event(
                "translation_queued",
                sentence_id=str(sentence_id),
                revision=int(revision),
                seq=int(seq_hint or 0),
                src_chars=len(src),
                src_hash8=_hash8(src),
                direction=direction,
                source_language=source_language,
                target_language=target_language,
                queue_depth=int(translation_runtime.queue.qsize()),
            )

            if translation_runtime.task is None or translation_runtime.task.done():
                translation_runtime.task = asyncio.create_task(_translation_worker())

        async def _translate_sentence_now(
            sentence_id: str,
            revision: int,
            sentence_text: str,
            language: str,
            seq_hint: int,
        ) -> str:
            if translator is None:
                return ""
            src = str(sentence_text or "").strip()
            if not src:
                return ""
            direction = str(getattr(translation_runtime, "direction", "zh2en") or "zh2en")
            source_language = str(getattr(translation_runtime, "source_language", "") or "")
            target_language = str(getattr(translation_runtime, "target_language", "") or "")
            state_gen_hint = _mark_latest_translation_request(
                sentence_id,
                revision,
                src,
                direction,
            )
            return await _translate_and_publish_current(
                sentence_id,
                revision,
                src,
                language,
                seq_hint,
                state_gen_hint,
                source_language,
                target_language,
                direction,
            )

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

        def _reset_punct_cut_state(reason: str = "") -> None:
            had_signal = bool(
                int(getattr(punct_cut_runtime, "candidate_end", 0) or 0) > 0
                or int(getattr(punct_cut_runtime, "anchor_end", 0) or 0) > 0
            )
            punct_cut_runtime.gate_open = False
            punct_cut_runtime.gate_chars = 0
            punct_cut_runtime.gate_open_at = 0.0
            punct_cut_runtime.candidate_end = 0
            punct_cut_runtime.candidate_token = ""
            punct_cut_runtime.candidate_since = 0.0
            punct_cut_runtime.candidate_hits = 0
            punct_cut_runtime.anchor_end = 0
            punct_cut_runtime.anchor_token = ""
            punct_cut_runtime.anchor_locked_at = 0.0
            punct_cut_runtime.anchor_seq = 0
            punct_cut_runtime.triggered = False
            if had_signal and reason:
                _trace_event("punct_cut_state_reset", reason=str(reason), segment_id=int(getattr(segment_runtime, "id", 0) or 0))

        def _maybe_get_punct_timeout_cut(
            snapshot_text: str,
            seq_hint: int,
            segment_age_ms: float,
        ) -> Optional[Dict[str, Any]]:
            if not punct_cut_enabled:
                return None
            snapshot = str(snapshot_text or "").strip()
            if not snapshot:
                return None
            age_ms = max(0.0, float(segment_age_ms))
            if age_ms < float(punct_cut_start_ms):
                return None
            now_mono = time.monotonic()

            if not bool(getattr(punct_cut_runtime, "gate_open", False)):
                punct_cut_runtime.gate_open = True
                punct_cut_runtime.gate_chars = len(snapshot)
                punct_cut_runtime.gate_open_at = now_mono
                _trace_event(
                    "punct_cut_gate_open",
                    seq=int(seq_hint or 0),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    gate_chars=int(punct_cut_runtime.gate_chars),
                    segment_age_ms=int(age_ms),
                )

            gate_chars = int(getattr(punct_cut_runtime, "gate_chars", 0) or 0)
            gate_chars = max(0, min(len(snapshot), gate_chars))
            found = _find_first_boundary_after(snapshot, gate_chars, punct_cut_pattern)
            if found is None:
                return None
            boundary_end, boundary_token = found

            prev_candidate_end = int(getattr(punct_cut_runtime, "candidate_end", 0) or 0)
            prev_candidate_token = str(getattr(punct_cut_runtime, "candidate_token", "") or "")
            if boundary_end != prev_candidate_end or boundary_token != prev_candidate_token:
                punct_cut_runtime.candidate_end = int(boundary_end)
                punct_cut_runtime.candidate_token = str(boundary_token or "")
                punct_cut_runtime.candidate_since = now_mono
                punct_cut_runtime.candidate_hits = 1
                _trace_event(
                    "punct_cut_candidate_set",
                    seq=int(seq_hint or 0),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    boundary_end=int(boundary_end),
                    token=str(boundary_token or ""),
                    gate_chars=int(gate_chars),
                )
            else:
                punct_cut_runtime.candidate_hits = int(getattr(punct_cut_runtime, "candidate_hits", 0) or 0) + 1

            stable_elapsed_ms = (now_mono - float(getattr(punct_cut_runtime, "candidate_since", now_mono) or now_mono)) * 1000.0
            anchor_end = int(getattr(punct_cut_runtime, "anchor_end", 0) or 0)
            if anchor_end <= 0:
                if (
                    int(getattr(punct_cut_runtime, "candidate_hits", 0) or 0) >= int(punct_cut_stable_hits)
                    and stable_elapsed_ms >= float(punct_cut_stable_ms)
                ):
                    punct_cut_runtime.anchor_end = int(boundary_end)
                    punct_cut_runtime.anchor_token = str(boundary_token or "")
                    punct_cut_runtime.anchor_locked_at = now_mono
                    punct_cut_runtime.anchor_seq = int(seq_hint or 0)
                    anchor_end = int(boundary_end)
                    _trace_event(
                        "punct_cut_anchor_locked",
                        seq=int(seq_hint or 0),
                        segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                        boundary_end=int(boundary_end),
                        token=str(boundary_token or ""),
                        stable_hits=int(getattr(punct_cut_runtime, "candidate_hits", 0) or 0),
                        stable_ms=int(stable_elapsed_ms),
                    )

            if anchor_end <= 0:
                return None
            if bool(getattr(punct_cut_runtime, "triggered", False)):
                return None

            wait_elapsed_ms = (now_mono - float(getattr(punct_cut_runtime, "anchor_locked_at", now_mono) or now_mono)) * 1000.0
            if wait_elapsed_ms < float(punct_cut_wait_ms):
                return None

            resolved = _resolve_boundary_for_anchor(snapshot, anchor_end, punct_cut_pattern)
            if resolved is None:
                return None
            split_end, split_token = resolved
            split_left, split_right = _split_text_at_boundary(snapshot, split_end)
            if not split_left:
                return None
            if int(punct_cut_max_carry_chars) > 0 and len(split_right) > int(punct_cut_max_carry_chars):
                _trace_event(
                    "punct_cut_skip_large_carry",
                    seq=int(seq_hint or 0),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    anchor_end=int(anchor_end),
                    split_end=int(split_end),
                    token=str(split_token or ""),
                    left_chars=len(split_left),
                    right_chars=len(split_right),
                    max_carry_chars=int(punct_cut_max_carry_chars),
                )
                punct_cut_runtime.anchor_end = int(split_end)
                punct_cut_runtime.anchor_token = str(split_token or "")
                punct_cut_runtime.anchor_locked_at = now_mono
                punct_cut_runtime.anchor_seq = int(seq_hint or 0)
                punct_cut_runtime.triggered = False
                return None
            punct_cut_runtime.triggered = True
            _trace_event(
                "punct_cut_triggered",
                seq=int(seq_hint or 0),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                anchor_end=int(anchor_end),
                split_end=int(split_end),
                token=str(split_token or ""),
                wait_ms=int(wait_elapsed_ms),
                left_chars=len(split_left),
                right_chars=len(split_right),
            )
            return {
                "should_cut": True,
                "split_end": int(split_end),
                "split_token": str(split_token or ""),
                "split_left": str(split_left or ""),
                "split_right": str(split_right or ""),
            }

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
            pending_reason = str(getattr(subtitle_state, "pending_prefix_reason", "") or "")
            pending_miss_count = int(getattr(subtitle_state, "pending_prefix_miss_count", 0) or 0)
            pending_terminal = str(
                getattr(subtitle_state, "pending_prefix_terminal_text", "") or ""
            ).strip()
            merged = raw
            overlap_cap = int(getattr(subtitle_state, "boundary_overlap_cap_chars", 12) or 12)
            if pending_prefix:
                if not raw:
                    merged = pending_prefix
                else:
                    min_overlap = 8 if pending_reason == "punct_timeout_cut" else 4
                    max_overlap = max(int(min_overlap), int(overlap_cap))
                    trimmed, overlap = trim_prefix_overlap(
                        pending_prefix,
                        raw,
                        min_overlap=int(min_overlap),
                        max_overlap=int(max_overlap),
                    )
                    if overlap >= int(min_overlap):
                        merged = _join_segments([pending_prefix, str(trimmed or "").strip()]).strip()
                        pending_miss_count = 0
                        _clear_pending_prefix_boundary_evidence()
                        _trace_event(
                            "pending_prefix_overlap_trimmed",
                            seq=int(seq_no or 0),
                            overlap_chars=int(overlap),
                            min_overlap=int(min_overlap),
                            cap_chars=int(max_overlap),
                            pending_reason=str(pending_reason or ""),
                            pending_chars=len(pending_prefix),
                            raw_chars=len(raw),
                            merged_chars=len(merged),
                            boundary_join_mode="overlap",
                            pending_segment_id=int(getattr(subtitle_state, "pending_prefix_segment_id", 0) or 0),
                        )
                    elif raw.startswith(pending_prefix) or (pending_prefix and pending_prefix in raw):
                        merged = raw
                        pending_miss_count = 0
                        subtitle_state.pending_prefix_text = ""
                        subtitle_state.pending_prefix_segment_id = 0
                        subtitle_state.pending_prefix_reason = ""
                        _clear_pending_prefix_boundary_evidence()
                        _trace_event(
                            "pending_prefix_cleared_consumed_by_raw",
                            seq=int(seq_no or 0),
                            pending_reason=str(pending_reason or ""),
                            pending_chars=len(pending_prefix),
                            raw_chars=len(raw),
                        )
                    elif pending_prefix.startswith(raw) and len(raw) <= max(10, int(len(pending_prefix) * 0.45)):
                        merged = pending_prefix
                        pending_miss_count = min(8, pending_miss_count + 1)
                        _trace_event(
                            "pending_prefix_hold_short_raw",
                            seq=int(seq_no or 0),
                            pending_reason=str(pending_reason or ""),
                            pending_chars=len(pending_prefix),
                            raw_chars=len(raw),
                            miss_count=int(pending_miss_count),
                        )
                    else:
                        pending_miss_count = min(8, pending_miss_count + 1)
                        if pending_reason == "hard_cut":
                            raw_completed, _ = _split_sentences_and_tail(raw)
                            if pending_terminal and raw_completed:
                                merged = f"{pending_terminal} {raw}".strip()
                                pending_miss_count = 0
                                subtitle_state.pending_prefix_is_separate = True
                                subtitle_state.boundary_anchor_text = ""
                                subtitle_state.boundary_anchor_segment_id = 0
                                _trace_event(
                                    "pending_prefix_terminal_boundary_preserved",
                                    seq=int(seq_no or 0),
                                    pending_chars=len(pending_prefix),
                                    terminal_chars=len(pending_terminal),
                                    raw_chars=len(raw),
                                    raw_completed_count=len(raw_completed),
                                    merged_chars=len(merged),
                                )
                            elif _should_hard_cut_fallback_merge(pending_prefix, raw):
                                merged = dedup_segment_join(pending_prefix, raw, min_overlap=2).strip()
                                _trace_event(
                                    "pending_prefix_hard_cut_fallback_merge",
                                    seq=int(seq_no or 0),
                                    pending_chars=len(pending_prefix),
                                    raw_chars=len(raw),
                                    merged_chars=len(merged),
                                    miss_count=int(pending_miss_count),
                                    boundary_join_mode=_classify_boundary_join_mode(
                                        pending_prefix,
                                        raw,
                                        merged,
                                    ),
                                )
                            else:
                                merged = raw
                                subtitle_state.pending_prefix_text = ""
                                subtitle_state.pending_prefix_segment_id = 0
                                subtitle_state.pending_prefix_reason = ""
                                _clear_pending_prefix_boundary_evidence()
                                pending_miss_count = 0
                                _trace_event(
                                    "pending_prefix_hard_cut_fallback_skip",
                                    seq=int(seq_no or 0),
                                    pending_chars=len(pending_prefix),
                                    raw_chars=len(raw),
                                    pending_reason=str(pending_reason or ""),
                                )
                        else:
                            merged = raw
                            completed_now, _ = _split_sentences_and_tail(raw)
                            should_drop_pending = bool(
                                completed_now
                                or pending_miss_count >= 2
                                or len(raw) >= max(24, int(len(pending_prefix) * 0.8))
                            )
                            if should_drop_pending:
                                subtitle_state.pending_prefix_text = ""
                                subtitle_state.pending_prefix_segment_id = 0
                                subtitle_state.pending_prefix_reason = ""
                                _clear_pending_prefix_boundary_evidence()
                                pending_miss_count = 0
                                _trace_event(
                                    "pending_prefix_drop_no_overlap",
                                    seq=int(seq_no or 0),
                                    pending_reason=str(pending_reason or ""),
                                    pending_chars=len(pending_prefix),
                                    raw_chars=len(raw),
                                    has_completed=bool(completed_now),
                                )
            else:
                pending_miss_count = 0
            subtitle_state.pending_prefix_miss_count = int(max(0, pending_miss_count))

            boundary_anchor = str(getattr(subtitle_state, "boundary_anchor_text", "") or "").strip()
            if overlap_cap > 0 and boundary_anchor and merged:
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

        def _stabilize_partial_text(raw_text: str, seq_hint: int) -> Tuple[str, bool]:
            raw = str(raw_text or "").strip()
            if not raw:
                stream_text_state.accepted_text = ""
                stream_text_state.reset_candidate_text = ""
                stream_text_state.reset_candidate_hits = 0
                stream_text_state.reset_candidate_since = 0.0
                return "", False

            prev = str(getattr(stream_text_state, "accepted_text", "") or "").strip()
            if not prev:
                stream_text_state.accepted_text = raw
                stream_text_state.reset_candidate_text = ""
                stream_text_state.reset_candidate_hits = 0
                stream_text_state.reset_candidate_since = 0.0
                return raw, False

            should_hold = _should_hold_partial_reset(
                prev_text=prev,
                next_text=raw,
                min_prev_chars=partial_reset_guard_min_prev_chars,
                max_next_ratio=partial_reset_guard_max_ratio,
            )
            if not should_hold:
                stream_text_state.accepted_text = raw
                stream_text_state.reset_candidate_text = ""
                stream_text_state.reset_candidate_hits = 0
                stream_text_state.reset_candidate_since = 0.0
                return raw, False

            now_mono = time.monotonic()
            candidate_text = str(getattr(stream_text_state, "reset_candidate_text", "") or "")
            if raw == candidate_text:
                stream_text_state.reset_candidate_hits = int(getattr(stream_text_state, "reset_candidate_hits", 0) or 0) + 1
            else:
                stream_text_state.reset_candidate_text = raw
                stream_text_state.reset_candidate_hits = 1
                stream_text_state.reset_candidate_since = now_mono

            hold_sec = max(
                0.0,
                now_mono - float(getattr(stream_text_state, "reset_candidate_since", now_mono) or now_mono),
            )
            hits = int(getattr(stream_text_state, "reset_candidate_hits", 0) or 0)
            release = _should_release_partial_reset_guard(
                candidate_hits=hits,
                hold_sec=hold_sec,
                min_hits=partial_reset_guard_release_hits,
                max_hold_sec=partial_reset_guard_max_hold_sec,
            )
            if not release:
                _trace_event(
                    "partial_reset_guard_hold",
                    seq=int(seq_hint or 0),
                    prev_chars=len(prev),
                    next_chars=len(raw),
                    candidate_hits=int(hits),
                    hold_ms=int(hold_sec * 1000.0),
                )
                return prev, True

            stream_text_state.accepted_text = raw
            stream_text_state.reset_candidate_text = ""
            stream_text_state.reset_candidate_hits = 0
            stream_text_state.reset_candidate_since = 0.0
            _trace_event(
                "partial_reset_guard_release",
                seq=int(seq_hint or 0),
                prev_chars=len(prev),
                next_chars=len(raw),
                reason="hits" if hits >= int(partial_reset_guard_release_hits) else "timeout",
                candidate_hits=int(hits),
                hold_ms=int(hold_sec * 1000.0),
            )
            return raw, False

        def _apply_incremental_text_fields(payload: Dict[str, Any]) -> None:
            full_text = str(payload.get("text", "") or "").strip()
            prev_text = str(stream_text_state.last_text or "")
            delta_text, text_reset = _compute_text_delta(prev_text, full_text)
            payload["state_text"] = full_text
            payload["delta_text"] = delta_text
            payload["text_reset"] = bool(text_reset)
            if text_reset:
                pending_prefix_now = str(getattr(subtitle_state, "pending_prefix_text", "") or "").strip()
                tentative_now = str(getattr(subtitle_state, "tentative_tail", "") or "").strip()
                _trace_event(
                    "partial_text_reset_detected",
                    seq=int(payload.get("seq", 0) or 0),
                    prev_chars=len(prev_text.strip()),
                    next_chars=len(full_text),
                    delta_chars=len(str(delta_text or "").strip()),
                    prev_hash8=_hash8(prev_text),
                    next_hash8=_hash8(full_text),
                    pending_prefix_chars=len(pending_prefix_now),
                    pending_prefix_hash8=_hash8(pending_prefix_now),
                    tentative_chars=len(tentative_now),
                    tentative_hash8=_hash8(tentative_now),
                    prev_preview=_trace_preview(prev_text, 96),
                    next_preview=_trace_preview(full_text, 96),
                )
            stream_text_state.last_text = full_text

        def _attach_stability(
            payload: Dict[str, Any],
            *,
            is_stable: bool,
            phase: str,
            reason: str,
            tentative_text: str = "",
            sentence_id: str = "",
        ) -> None:
            tentative = str(tentative_text or "").strip()
            stable = bool(is_stable)
            payload["is_stable"] = stable
            payload["stability"] = {
                "is_stable": stable,
                "phase": str(phase or ""),
                "reason": str(reason or ""),
                "sentence_id": str(sentence_id or payload.get("sentence_id", "") or ""),
                "segment_id": int(getattr(segment_runtime, "id", 0) or 0),
                "seq": int(payload.get("seq", 0) or 0),
                "committed_count": int(len(subtitle_state.committed_sentences)),
                "tentative_chars": int(len(tentative)),
                "unstable_chars": int(0 if stable else len(tentative)),
            }

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
            cut_boundary_end: int = 0,
        ) -> bool:
            nonlocal state, seq, last_text_snapshot, last_text_advance_at, last_idle_commit_at, last_partial_emit_at
            nonlocal segment_final_context_applied
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
                context_final_applied = False
                async with infer_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, local_state)
                    _guard_streaming_context_output(local_state, reason=str(reason or "segment_finalize"))
                    context_final_applied = await _apply_segment_final_context(
                        local_state,
                        reason=str(reason or ""),
                    )
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
                if context_final_applied:
                    segment_final_context_applied = True

            seq = max(int(seq), int(seq_hint or 0)) + 1
            raw_final_text = str(getattr(local_state, "text", "") or snapshot_text or "").strip()
            final_text = str(raw_final_text or "")
            punct_cut_carry_text = ""
            punct_cut_resolved_end = 0
            if str(reason or "") == "punct_timeout_cut" and int(cut_boundary_end or 0) > 0 and raw_final_text:
                resolved = _resolve_boundary_for_anchor(raw_final_text, int(cut_boundary_end or 0), punct_cut_pattern)
                if resolved is not None:
                    punct_cut_resolved_end, punct_cut_token = resolved
                    split_left, split_right = _split_text_at_boundary(raw_final_text, punct_cut_resolved_end)
                    if split_left:
                        final_text = str(split_left or "").strip()
                        punct_cut_carry_text = str(split_right or "").strip()
                        _trace_event(
                            "punct_cut_finalize_split",
                            seq=int(seq),
                            segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                            requested_end=int(cut_boundary_end or 0),
                            resolved_end=int(punct_cut_resolved_end),
                            token=str(punct_cut_token or ""),
                            commit_chars=len(final_text),
                            carry_chars=len(punct_cut_carry_text),
                        )
            final_language = str(getattr(local_state, "language", "") or snapshot_language or "")
            commit_tail_on_finalize = _should_commit_tail_on_segment_finalize(
                str(reason or ""),
                final_text,
                force_finalize=bool(force_finalize),
            )
            finalize_completed_preview, finalize_tail_preview = _split_sentences_and_tail(final_text)
            defer_short_english_slice_commit = bool(
                str(reason or "") in {"vad_silence", "hard_cut", "punct_timeout_cut"}
                and len(finalize_completed_preview) == 1
                and not str(finalize_tail_preview or "").strip()
                and _is_short_english_sentence_for_early_commit(
                    str(finalize_completed_preview[0] or ""),
                    min_words=int(early_translation_min_english_words),
                    min_chars=int(early_translation_min_english_chars),
                )
            )
            if defer_short_english_slice_commit:
                commit_tail_on_finalize = False
                _trace_event(
                    "short_english_slice_commit_deferred",
                    seq=int(seq),
                    reason=str(reason or ""),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    sentence_chars=len(str(finalize_completed_preview[0] or "").strip()),
                    english_words=int(_english_word_count(str(finalize_completed_preview[0] or ""))),
                    min_english_words=int(early_translation_min_english_words),
                    min_english_chars=int(early_translation_min_english_chars),
                )

            _track_text_progress(final_text)
            tentative_after_finalize = await _update_sentence_commits(
                final_text,
                final_language,
                seq,
                force_tail=bool(commit_tail_on_finalize),
                holdback_newest=bool(defer_short_english_slice_commit),
                commit_tail_if_no_completed=bool(commit_tail_on_finalize),
                commit_tail_always=bool(commit_tail_on_finalize),
                commit_all_completed=not bool(defer_short_english_slice_commit),
                slice_commit=True,
                translate_now=False,
                canonical_segment_correction=bool(context_final_applied),
                final_reconcile=True,
            )
            pending_prefix = ""
            if not commit_tail_on_finalize:
                pending_prefix = str(tentative_after_finalize or "").strip()
                if not pending_prefix:
                    _, pending_tail = _split_sentences_and_tail(final_text)
                    pending_prefix = str(pending_tail or "").strip()
            if punct_cut_carry_text:
                if pending_prefix:
                    pending_prefix = dedup_segment_join(pending_prefix, punct_cut_carry_text, min_overlap=2).strip()
                else:
                    pending_prefix = str(punct_cut_carry_text or "").strip()
            pending_prefix_terminal_text = ""
            if finalize_completed_preview:
                terminal_candidate = str(finalize_completed_preview[-1] or "").strip()
                continuation_candidate = _strip_short_english_fragment_period(
                    terminal_candidate,
                    min_words=int(early_translation_min_english_words),
                    min_chars=int(early_translation_min_english_chars),
                )
                if (
                    pending_prefix
                    and continuation_candidate != terminal_candidate
                    and pending_prefix == continuation_candidate
                ):
                    pending_prefix_terminal_text = terminal_candidate
            overlap_cap_chars = int(getattr(subtitle_state, "boundary_overlap_cap_chars", 12) or 12)
            boundary_anchor_chars = max(4, overlap_cap_chars * 2)
            boundary_anchor = str(final_text[-boundary_anchor_chars:] if final_text else "").strip()
            subtitle_state.pending_prefix_text = str(pending_prefix or "")
            subtitle_state.pending_prefix_segment_id = int(getattr(segment_runtime, "id", 0) or 0)
            subtitle_state.pending_prefix_reason = str(reason or "")
            subtitle_state.pending_prefix_miss_count = 0
            subtitle_state.pending_prefix_terminal_text = str(pending_prefix_terminal_text or "")
            subtitle_state.pending_prefix_is_separate = False
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
                terminal_boundary_chars=len(str(pending_prefix_terminal_text or "")),
                boundary_anchor_chars=len(str(subtitle_state.boundary_anchor_text or "").strip()),
            )
            subtitle_state.commit_base = int(len(subtitle_state.committed_sentences))
            subtitle_state.prev_completed_sentences = []
            subtitle_state.tentative_tail = ""
            _reset_completed_candidate_cursor()
            _reset_early_translation_holdback_state()

            overlap_audio = np.zeros((0,), dtype=np.float32)
            local_audio_accum = getattr(local_state, "audio_accum", None)
            if isinstance(local_audio_accum, np.ndarray) and int(local_audio_accum.size) > 0 and segment_overlap_samples > 0:
                take = min(int(local_audio_accum.size), int(segment_overlap_samples))
                overlap_audio = np.asarray(local_audio_accum[-take:], dtype=np.float32).copy()

            context_elapsed_sec = float(total_consumed_samples) / float(SAMPLE_RATE)
            new_state = await asyncio.to_thread(
                _new_vllm_state,
                session_force_language,
                context_elapsed_sec,
                session_context_terms,
            )
            _trace_asr_context(
                session_force_language,
                context_elapsed_sec,
                session_context_terms,
            )
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
            _reset_punct_cut_state("segment_finalize")

            stream_text_state.last_text = ""
            stream_text_state.accepted_text = ""
            stream_text_state.reset_candidate_text = ""
            stream_text_state.reset_candidate_hits = 0
            stream_text_state.reset_candidate_since = 0.0
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
                raw_final_text_chars=len(raw_final_text),
                overlap_samples=int(overlap_audio.size),
                commit_tail=bool(commit_tail_on_finalize),
                defer_short_english_slice_commit=bool(defer_short_english_slice_commit),
                pending_prefix_chars=len(str(subtitle_state.pending_prefix_text or "").strip()),
                pending_prefix_terminal_chars=len(
                    str(getattr(subtitle_state, "pending_prefix_terminal_text", "") or "").strip()
                ),
                punct_cut_carry_chars=len(punct_cut_carry_text),
                punct_cut_split_end=int(punct_cut_resolved_end),
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
            cut_reason = ""
            force_finalize = False
            cut_boundary_end = 0
            cut_by_punct_timeout = False
            if decision.should_cut:
                cut_reason = str(decision.reason)
                force_finalize = bool(decision.force_finalize)
            if not cut_reason:
                return

            if cut_reason == "vad_silence" and not _text_ready_for_vad_cut(snapshot, force=bool(force_finalize)):
                _trace_event(
                    "segment_cut_deferred",
                    reason=str(cut_reason),
                    seq=int(seq_hint or 0),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    silence_ms=int(float(signal.get("silence_ms", 0.0) or 0.0)),
                    segment_age_ms=int(segment_age_ms),
                )
                return
            _trace_event(
                "segment_cut_decision",
                reason=str(cut_reason),
                seq=int(seq_hint or 0),
                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                silence_ms=int(float(signal.get("silence_ms", 0.0) or 0.0)),
                segment_age_ms=int(segment_age_ms),
                force_finalize=bool(force_finalize),
                text_chars=len(snapshot),
                cut_boundary_end=int(cut_boundary_end),
                cut_by_punct_timeout=bool(cut_by_punct_timeout),
            )
            await _finalize_segment_and_rotate(
                reason=str(cut_reason),
                seq_hint=int(seq_hint or 0),
                snapshot_text=snapshot,
                snapshot_language=str(language or ""),
                force_finalize=bool(force_finalize),
                cut_boundary_end=int(cut_boundary_end),
            )

        def _is_committed_sentence_upgrade(old_text: str, new_text: str) -> bool:
            return _should_accept_sentence_upgrade(old_text, new_text)

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
            canonical_segment_correction: bool = False,
            final_reconcile: bool = False,
        ) -> str:
            seq_no = int(seq_hint or 0)
            raw_full_text = str(full_text or "").strip()
            total_committed_count = len(subtitle_state.committed_sentences)
            commit_base = int(getattr(subtitle_state, "commit_base", 0) or 0)
            commit_base = max(0, min(commit_base, total_committed_count))
            prev_committed_count = int(total_committed_count - commit_base)
            canonical_existing_text = _join_segments(
                subtitle_state.committed_sentences[commit_base:total_committed_count]
            )
            canonical_existing_compact = _compact_asr_compare_text(canonical_existing_text)
            canonical_covered_spans: List[Tuple[int, int]] = []
            prev_tail_text = str(subtitle_state.tentative_tail or "")
            prev_completed_count = len(subtitle_state.prev_completed_sentences)
            pending_prefix_before = str(getattr(subtitle_state, "pending_prefix_text", "") or "").strip()
            pending_prefix_reason = str(getattr(subtitle_state, "pending_prefix_reason", "") or "")
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
            processed_count = int(getattr(subtitle_state, "processed_completed_count", 0) or 0)
            candidate_cursor_before = int(processed_count)
            boundary_context = bool(pending_prefix_before or boundary_anchor_before)
            carry_duplicate_dropped = False
            duplicate_filter_paused = bool(
                time.monotonic() < float(getattr(subtitle_state, "duplicate_filter_pause_until", 0.0) or 0.0)
            )
            duplicate_filter_pause_left_ms = max(
                0,
                int(
                    (
                        float(getattr(subtitle_state, "duplicate_filter_pause_until", 0.0) or 0.0)
                        - time.monotonic()
                    )
                    * 1000.0
                ),
            )
            if (
                pending_prefix_before
                and completed
                and processed_count == 0
                and total_committed_count > 0
            ):
                prev_sentence = str(subtitle_state.committed_sentences[total_committed_count - 1] or "").strip()
                first_sentence = str(completed[0] or "").strip()
                if duplicate_filter_paused:
                    _trace_event(
                        "carry_duplicate_filter_skipped_regression_guard",
                        seq=seq_no,
                        pause_left_ms=int(duplicate_filter_pause_left_ms),
                        pending_prefix_chars=len(pending_prefix_before),
                        candidate_chars=len(first_sentence),
                        raw_chars=len(raw_full_text),
                        pause_reason=str(getattr(subtitle_state, "duplicate_filter_pause_reason", "") or ""),
                    )
                elif _is_probable_pending_prefix_duplicate(
                    prev_sentence,
                    first_sentence,
                    raw_full_text,
                    pending_prefix_before,
                ):
                    completed = completed[1:]
                    carry_duplicate_dropped = True
                    trimmed_pending, trimmed = _trim_pending_prefix_leading_sentence(
                        pending_prefix_before,
                        first_sentence,
                    )
                    if trimmed:
                        pending_prefix_prev = str(pending_prefix_before or "")
                        subtitle_state.pending_prefix_text = str(trimmed_pending or "")
                        subtitle_state.pending_prefix_miss_count = 0
                        _clear_pending_prefix_boundary_evidence()
                        _trace_text_pool(
                            "pending_prefix_trimmed",
                            phase="generating",
                            text=str(subtitle_state.pending_prefix_text or ""),
                            reason="carry_duplicate_filtered",
                            seq_hint=int(seq_no or 0),
                            delta_chars=(
                                len(str(subtitle_state.pending_prefix_text or "").strip())
                                - len(str(pending_prefix_prev or "").strip())
                            ),
                            sentence_id="",
                        )
                        pending_prefix_before = str(subtitle_state.pending_prefix_text or "").strip()
                    _trace_event(
                        "carry_duplicate_filtered",
                        seq=seq_no,
                        duplicate_chars=len(first_sentence),
                        raw_chars=len(raw_full_text),
                        pending_prefix_chars=len(pending_prefix_before),
                        commit_base=int(commit_base),
                        total_committed_count=int(total_committed_count),
                        prev_sentence_hash8=_hash8(prev_sentence),
                        candidate_hash8=_hash8(first_sentence),
                        raw_hash8=_hash8(raw_full_text),
                        pending_prefix_hash8=_hash8(pending_prefix_before),
                        prev_sentence_preview=_trace_preview(prev_sentence, 72),
                        candidate_preview=_trace_preview(first_sentence, 72),
                    )

            raw_completed_count_before_stabilize = int(len(completed))
            completed, committed_backfilled = _stabilize_completed_prefix_with_cursor(completed)
            raw_committed_underflow = max(0, int(processed_count - raw_completed_count_before_stabilize))
            if committed_backfilled > 0:
                _trace_event(
                    "completed_regression_backfilled",
                    seq=seq_no,
                    commit_base=int(commit_base),
                    committed_count=int(committed_count),
                    raw_completed_count=int(raw_completed_count_before_stabilize),
                    stabilized_completed_count=int(len(completed)),
                    backfilled_count=int(committed_backfilled),
                    raw_committed_underflow=int(raw_committed_underflow),
                    slice_commit=bool(slice_commit),
                    force_tail=bool(force_tail),
                )
            carry_overlap_skip = 0
            if boundary_context and processed_count == 0 and total_committed_count > 0 and completed:
                overlap = _count_leading_completed_committed_overlap(
                    completed,
                    subtitle_state.committed_sentences,
                    max_overlap=6,
                )
                if overlap > 0:
                    overlap_chars = sum(len(str(completed[i] or "").strip()) for i in range(overlap))
                    raw_chars_now = len(raw_full_text)
                    hard_cut_prefix_replay = bool(
                        pending_prefix_reason == "hard_cut"
                        and (int(overlap) >= 2 or int(overlap_chars) >= 80)
                    )
                    if hard_cut_prefix_replay or _should_apply_carry_overlap_skip(
                        overlap_count=int(overlap),
                        overlap_chars=int(overlap_chars),
                        raw_chars=int(raw_chars_now),
                    ):
                        carry_overlap_skip = int(overlap)
                        for skipped_idx in range(int(processed_count), int(carry_overlap_skip)):
                            skipped_text = str(completed[skipped_idx] or "").strip()
                            _record_completed_candidate(skipped_idx, skipped_text, "")
                            _trace_event(
                                "candidate_action",
                                seq=seq_no,
                                idx=int(skipped_idx),
                                action="structural_overlap_skip",
                                sentence_id="",
                                text_hash8=_hash8(skipped_text),
                                segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                            )
                        _trace_event(
                            "carry_overlap_commit_skip",
                            seq=seq_no,
                            overlap_count=int(overlap),
                            overlap_chars=int(overlap_chars),
                            raw_chars=int(raw_chars_now),
                            raw_limit=max(12, min(24, int(float(overlap_chars) * 0.35))),
                            pending_prefix_chars=len(pending_prefix_before),
                            pending_prefix_reason=pending_prefix_reason,
                            hard_cut_prefix_replay=bool(hard_cut_prefix_replay),
                            committed_count=int(committed_count),
                            total_committed_count=int(total_committed_count),
                        )
            processed_count = int(getattr(subtitle_state, "processed_completed_count", 0) or 0)
            effective_committed_count = max(int(processed_count), int(carry_overlap_skip))

            ready_end = effective_committed_count
            if force_tail or commit_all_completed:
                ready_end = len(completed)
            else:
                upper = min(len(completed), len(subtitle_state.prev_completed_sentences))
                i = effective_committed_count
                while i < upper and completed[i] == subtitle_state.prev_completed_sentences[i]:
                    i += 1
                ready_end = i
            ready_end = max(ready_end, effective_committed_count)
            if not force_tail and holdback_newest:
                # Keep the newest completed sentence as tentative text so it can
                # continue growing without being frozen as a committed row too early.
                newest_holdback = max(effective_committed_count, len(completed) - 1)
                ready_end = min(ready_end, newest_holdback)
                ready_end = max(ready_end, effective_committed_count)

            early_translation_promoted = False
            early_translation_waiting = False
            early_translation_short_english = False
            early_translation_required_sec = float(early_translation_stable_sec)
            early_translation_required_hits = int(early_translation_stable_hits)
            early_translation_stable_age_ms = 0
            early_translation_terminal_first_seen_ms = 0
            if (
                not force_tail
                and bool(holdback_newest)
                and int(ready_end) == int(len(completed) - 1)
                and int(ready_end) >= int(effective_committed_count)
                and int(ready_end) < int(len(completed))
            ):
                candidate = str(completed[int(ready_end)] or "").strip()
                now_mono = time.monotonic()
                if candidate and candidate == str(getattr(subtitle_state, "early_holdback_text", "") or ""):
                    subtitle_state.early_holdback_hits = int(getattr(subtitle_state, "early_holdback_hits", 0) or 0) + 1
                else:
                    subtitle_state.early_holdback_text = candidate
                    subtitle_state.early_holdback_since = now_mono
                    subtitle_state.early_holdback_first_seen_ms = int(time.time() * 1000)
                    subtitle_state.early_holdback_hits = 1 if candidate else 0
                stable_sec = max(
                    0.0,
                    now_mono - float(getattr(subtitle_state, "early_holdback_since", now_mono) or now_mono),
                )
                stable_hits = int(getattr(subtitle_state, "early_holdback_hits", 0) or 0)
                early_translation_stable_age_ms = int(stable_sec * 1000.0)
                early_translation_terminal_first_seen_ms = int(
                    getattr(subtitle_state, "early_holdback_first_seen_ms", 0) or 0
                )
                early_translation_short_english = _is_short_english_sentence_for_early_commit(
                    candidate,
                    min_words=int(early_translation_min_english_words),
                    min_chars=int(early_translation_min_english_chars),
                )
                early_translation_required_sec = (
                    float(early_translation_short_stable_sec)
                    if early_translation_short_english
                    else float(early_translation_stable_sec)
                )
                early_translation_required_hits = (
                    int(early_translation_short_stable_hits)
                    if early_translation_short_english
                    else int(early_translation_stable_hits)
                )
                if (
                    candidate
                    and stable_hits >= int(early_translation_required_hits)
                    and stable_sec >= float(early_translation_required_sec)
                ):
                    ready_end += 1
                    early_translation_promoted = True
                    _trace_event(
                        "early_translation_stable_commit",
                        seq=seq_no,
                        sentence_chars=len(candidate),
                        terminal_first_seen_ms=int(early_translation_terminal_first_seen_ms),
                        stable_age_ms=int(early_translation_stable_age_ms),
                        stable_hits=int(stable_hits),
                        required_hits=int(early_translation_required_hits),
                        required_stable_ms=int(float(early_translation_required_sec) * 1000.0),
                        short_english=bool(early_translation_short_english),
                        english_words=int(_english_word_count(candidate)),
                    )
                    _reset_early_translation_holdback_state()
                elif candidate:
                    early_translation_waiting = True
                    _trace_event(
                        "early_translation_stability_wait",
                        seq=seq_no,
                        sentence_chars=len(candidate),
                        terminal_first_seen_ms=int(early_translation_terminal_first_seen_ms),
                        stable_age_ms=int(early_translation_stable_age_ms),
                        stable_hits=int(stable_hits),
                        required_hits=int(early_translation_required_hits),
                        required_stable_ms=int(float(early_translation_required_sec) * 1000.0),
                        short_english=bool(early_translation_short_english),
                        english_words=int(_english_word_count(candidate)),
                    )
            else:
                _reset_early_translation_holdback_state()

            short_english_slice_fragment_held = False
            short_english_slice_fragment_index = -1
            short_english_slice_fragment_text = ""
            if bool(slice_commit) and not force_tail and int(ready_end) > int(effective_committed_count):
                for idx in range(int(effective_committed_count), int(ready_end)):
                    candidate = str(completed[int(idx)] or "").strip()
                    if not _is_short_english_slice_fragment(
                        candidate,
                        min_words=int(early_translation_min_english_words),
                        min_chars=int(early_translation_min_english_chars),
                    ):
                        continue
                    ready_end = int(idx)
                    short_english_slice_fragment_held = True
                    short_english_slice_fragment_index = int(idx)
                    short_english_slice_fragment_text = candidate
                    _trace_event(
                        "short_english_slice_fragment_hold",
                        seq=seq_no,
                        idx=int(idx),
                        sentence_chars=len(candidate),
                        english_words=int(_english_word_count(candidate)),
                        min_english_words=int(early_translation_min_english_words),
                        min_english_chars=int(early_translation_min_english_chars),
                        slice_commit=bool(slice_commit),
                    )
                    break

            ready_new_commits = max(0, int(ready_end - effective_committed_count))
            suppressed_new_commits = max(0, int(len(completed) - ready_end))
            completed_underflow = max(0, int(prev_completed_count - len(completed)))
            prev_completed_joined = _join_segments([str(seg or "").strip() for seg in subtitle_state.prev_completed_sentences])
            completed_joined = _join_segments([str(seg or "").strip() for seg in completed])
            if completed_underflow > 0 or raw_committed_underflow > 0:
                pause_sec = max(1.2, min(3.0, 1.2 + 0.4 * float(completed_underflow)))
                pause_until = time.monotonic() + pause_sec
                prev_pause_until = float(getattr(subtitle_state, "duplicate_filter_pause_until", 0.0) or 0.0)
                subtitle_state.duplicate_filter_pause_until = max(prev_pause_until, pause_until)
                subtitle_state.duplicate_filter_pause_reason = "completed_regression"

            should_sample = (
                seq_no <= 3
                or seq_no % subtitle_trace_log_partial_every == 0
                or force_tail
                or commit_tail_always
                or commit_all_completed
                or bool(slice_commit)
                or bool(carry_duplicate_dropped)
                or bool(early_translation_promoted)
                or bool(early_translation_waiting)
                or bool(short_english_slice_fragment_held)
                or ready_end != effective_committed_count
            )
            if completed_underflow > 0:
                matched_prefix = _count_matching_sentence_prefix(
                    subtitle_state.prev_completed_sentences,
                    completed,
                )
                _trace_event(
                    "completed_regression_detected",
                    seq=seq_no,
                    prev_completed_count=int(prev_completed_count),
                    completed_count=int(len(completed)),
                    committed_count=int(committed_count),
                    dropped_completed=int(completed_underflow),
                    raw_chars=len(raw_full_text),
                    effective_chars=len(effective_full_text),
                    prev_completed_hash8=_hash8(prev_completed_joined),
                    completed_hash8=_hash8(completed_joined),
                    raw_hash8=_hash8(raw_full_text),
                    effective_hash8=_hash8(effective_full_text),
                    prev_completed_preview=_trace_preview(prev_completed_joined, 96),
                    completed_preview=_trace_preview(completed_joined, 96),
                    matched_prefix_count=int(matched_prefix),
                )
                _trace_event(
                    "completed_regression_sentence_diff",
                    seq=seq_no,
                    dropped_completed=int(completed_underflow),
                    prev_completed_count=int(prev_completed_count),
                    completed_count=int(len(completed)),
                    matched_prefix_count=int(matched_prefix),
                    prev_sentences=_sentence_signature_rows(subtitle_state.prev_completed_sentences),
                    new_sentences=_sentence_signature_rows(completed),
                )
                _trace_text_pool(
                    "regression_snapshot_generating",
                    phase="generating",
                    text=str(effective_full_text or ""),
                    reason="completed_regression",
                    seq_hint=int(seq_no),
                    delta_chars=0,
                    dropped_completed=int(completed_underflow),
                    matched_prefix_count=int(matched_prefix),
                )
                _trace_text_pool(
                    "regression_snapshot_solidified",
                    phase="solidified",
                    text=_join_segments(subtitle_state.committed_sentences),
                    reason="completed_regression",
                    seq_hint=int(seq_no),
                    delta_chars=0,
                    dropped_completed=int(completed_underflow),
                    matched_prefix_count=int(matched_prefix),
                )
            if raw_committed_underflow > 0 and completed_underflow <= 0:
                _trace_event(
                    "committed_underflow_detected",
                    seq=seq_no,
                    committed_count=int(committed_count),
                    raw_completed_count=int(raw_completed_count_before_stabilize),
                    stabilized_completed_count=int(len(completed)),
                    raw_committed_underflow=int(raw_committed_underflow),
                    prev_sentences=_sentence_signature_rows(subtitle_state.prev_completed_sentences),
                    new_sentences=_sentence_signature_rows(completed),
                )
                _trace_text_pool(
                    "underflow_snapshot_generating",
                    phase="generating",
                    text=str(effective_full_text or ""),
                    reason="committed_underflow",
                    seq_hint=int(seq_no),
                    delta_chars=0,
                    raw_committed_underflow=int(raw_committed_underflow),
                )
                _trace_text_pool(
                    "underflow_snapshot_solidified",
                    phase="solidified",
                    text=_join_segments(subtitle_state.committed_sentences),
                    reason="committed_underflow",
                    seq_hint=int(seq_no),
                    delta_chars=0,
                    raw_committed_underflow=int(raw_committed_underflow),
                )
            if suppressed_new_commits > 0 and (should_sample or suppressed_new_commits > 1):
                if short_english_slice_fragment_held:
                    suppressed_reason = "short_english_slice_fragment"
                else:
                    suppressed_reason = "holdback_newest" if bool(holdback_newest and not force_tail) else "stability_guard"
                _trace_event(
                    "commit_suppressed",
                    seq=seq_no,
                    reason=suppressed_reason,
                    committed_count=int(committed_count),
                    completed_count=int(len(completed)),
                    ready_end=int(ready_end),
                    ready_new_commits=int(ready_new_commits),
                    suppressed_new_commits=int(suppressed_new_commits),
                    force_tail=bool(force_tail),
                    holdback_newest=bool(holdback_newest),
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
                    effective_committed_count=int(effective_committed_count),
                    candidate_cursor_before=int(candidate_cursor_before),
                    candidate_cursor_after=int(max(effective_committed_count, ready_end)),
                    carry_overlap_skip=int(carry_overlap_skip),
                    prev_completed_count=int(prev_completed_count),
                    ready_end=int(ready_end),
                    holdback_newest=bool(holdback_newest),
                    force_tail=bool(force_tail),
                    commit_tail_if_no_completed=bool(commit_tail_if_no_completed),
                    commit_tail_always=bool(commit_tail_always),
                    commit_all_completed=bool(commit_all_completed),
                    slice_commit=bool(slice_commit),
                    translate_now=bool(translate_now),
                    canonical_segment_correction=bool(canonical_segment_correction),
                    early_translation_promoted=bool(early_translation_promoted),
                    early_translation_waiting=bool(early_translation_waiting),
                    early_translation_short_english=bool(early_translation_short_english),
                    early_translation_stable_age_ms=int(early_translation_stable_age_ms),
                    early_translation_required_hits=int(early_translation_required_hits),
                    early_translation_required_stable_ms=int(float(early_translation_required_sec) * 1000.0),
                    short_english_slice_fragment_held=bool(short_english_slice_fragment_held),
                    short_english_slice_fragment_index=int(short_english_slice_fragment_index),
                    pending_prefix_chars=len(pending_prefix_before),
                    boundary_anchor_chars=len(boundary_anchor_before),
                    duplicate_filter_paused=bool(duplicate_filter_paused),
                    duplicate_filter_pause_left_ms=int(duplicate_filter_pause_left_ms),
                )
            if should_sample or completed_underflow > 0 or suppressed_new_commits > 0:
                committed_last = (
                    str(subtitle_state.committed_sentences[total_committed_count - 1] or "").strip()
                    if total_committed_count > 0
                    else ""
                )
                _trace_event(
                    "commit_probe",
                    seq=seq_no,
                    raw_chars=len(raw_full_text),
                    effective_chars=len(effective_full_text),
                    tail_chars=len(str(tail or "").strip()),
                    committed_count=int(committed_count),
                    effective_committed_count=int(effective_committed_count),
                    candidate_cursor_before=int(candidate_cursor_before),
                    candidate_cursor_after=int(max(effective_committed_count, ready_end)),
                    completed_count=int(len(completed)),
                    prev_completed_count=int(prev_completed_count),
                    ready_end=int(ready_end),
                    ready_new_commits=int(ready_new_commits),
                    suppressed_new_commits=int(suppressed_new_commits),
                    early_translation_promoted=bool(early_translation_promoted),
                    early_translation_waiting=bool(early_translation_waiting),
                    early_translation_short_english=bool(early_translation_short_english),
                    early_translation_stable_age_ms=int(early_translation_stable_age_ms),
                    short_english_slice_fragment_held=bool(short_english_slice_fragment_held),
                    short_english_slice_fragment_hash8=_hash8(short_english_slice_fragment_text),
                    completed_underflow=int(completed_underflow),
                    raw_hash8=_hash8(raw_full_text),
                    effective_hash8=_hash8(effective_full_text),
                    pending_prefix_hash8=_hash8(pending_prefix_before),
                    pending_prefix_chars=len(pending_prefix_before),
                    boundary_anchor_hash8=_hash8(boundary_anchor_before),
                    boundary_anchor_chars=len(boundary_anchor_before),
                    prev_completed_hash8=_hash8(prev_completed_joined),
                    completed_hash8=_hash8(completed_joined),
                    committed_last_hash8=_hash8(committed_last),
                    raw_preview=_trace_preview(raw_full_text, 96),
                    effective_preview=_trace_preview(effective_full_text, 96),
                    duplicate_filter_paused=bool(duplicate_filter_paused),
                    duplicate_filter_pause_left_ms=int(duplicate_filter_pause_left_ms),
                )

            candidate_sentence_ids = list(getattr(subtitle_state, "candidate_sentence_ids", []) or [])
            update_upper = min(processed_count, len(completed), len(candidate_sentence_ids))
            committed_added = 0
            upgraded_count = 0
            evaluated_sentence_ids = set()
            for i in range(update_upper):
                sentence_id = str(candidate_sentence_ids[i] or "")
                global_idx = _find_sentence_item_index(sentence_id)
                if global_idx is None or global_idx >= len(subtitle_state.committed_sentences):
                    continue
                evaluated_sentence_ids.add(sentence_id)
                upgraded = str(completed[i] or "").strip()
                current = str(subtitle_state.committed_sentences[global_idx] or "").strip()
                accepted_upgrade = _is_committed_sentence_upgrade(current, upgraded)
                accepted_context_correction = bool(
                    canonical_segment_correction
                    and _should_accept_context_sentence_correction(current, upgraded)
                )
                small_upgrade_source = ""
                small_upgrade_observation = None

                if accepted_upgrade or accepted_context_correction:
                    _reset_deferred_sentence_upgrade(
                        sentence_id,
                        seq_no=seq_no,
                        reason="superseded_by_existing_upgrade",
                        replacement_text=upgraded,
                    )
                elif _is_monotonic_sentence_extension(current, upgraded):
                    if final_reconcile:
                        accepted_upgrade = True
                        small_upgrade_source = "final_reconcile"
                    else:
                        small_upgrade_observation = _observe_deferred_sentence_upgrade(
                            subtitle_state.deferred_sentence_upgrades,
                            sentence_id,
                            upgraded,
                            seq_no,
                            time.monotonic(),
                            _SMALL_UPGRADE_REQUIRED_HITS,
                            _SMALL_UPGRADE_STABLE_SEC,
                        )
                        if small_upgrade_observation.transition in {"started", "changed"}:
                            if small_upgrade_observation.transition == "changed":
                                _trace_event(
                                    "sentence_upgrade_candidate_reset",
                                    seq=seq_no,
                                    sentence_id=sentence_id,
                                    reason="candidate_changed",
                                    candidate_hash8=_hash8(small_upgrade_observation.previous_text),
                                    candidate_chars=len(small_upgrade_observation.previous_text),
                                    replacement_hash8=_hash8(upgraded),
                                    replacement_chars=len(upgraded),
                                )
                            _trace_event(
                                "sentence_upgrade_deferred",
                                seq=seq_no,
                                sentence_id=sentence_id,
                                transition=small_upgrade_observation.transition,
                                old_chars=len(current),
                                new_chars=len(upgraded),
                                growth_chars=max(0, len(upgraded) - len(current)),
                                candidate_hash8=_hash8(upgraded),
                                hits=small_upgrade_observation.hits,
                                stable_age_ms=small_upgrade_observation.stable_ms,
                                old_preview=_trace_preview(current, 96),
                                new_preview=_trace_preview(upgraded, 96),
                            )
                        if small_upgrade_observation.ready:
                            accepted_upgrade = True
                            small_upgrade_source = "streaming_stable"
                elif sentence_id in subtitle_state.deferred_sentence_upgrades:
                    previous = subtitle_state.deferred_sentence_upgrades.get(sentence_id)
                    _trace_event(
                        "sentence_upgrade_rejected",
                        seq=seq_no,
                        sentence_id=sentence_id,
                        reason="candidate_retracted_or_rewritten",
                        candidate_hash8=_hash8(previous.text if previous is not None else ""),
                        replacement_hash8=_hash8(upgraded),
                        replacement_chars=len(upgraded),
                    )
                    _reset_deferred_sentence_upgrade(
                        sentence_id,
                        seq_no=seq_no,
                        reason="candidate_retracted_or_rewritten",
                        replacement_text=upgraded,
                    )
                if not accepted_upgrade and not accepted_context_correction:
                    continue
                upgraded_count += 1
                subtitle_state.committed_sentences[global_idx] = upgraded
                sentence_item = subtitle_state.sentence_items[global_idx]
                sentence_item["zh"] = upgraded
                revision = int(sentence_item.get("revision", 1) or 1) + 1
                sentence_item["revision"] = int(revision)
                await _register_tts_source(sentence_id, revision)
                _record_completed_candidate(i, upgraded, sentence_id)
                preserved_translation = str(sentence_item.get("en", "") or "")
                if small_upgrade_source:
                    accepted_candidate = subtitle_state.deferred_sentence_upgrades.pop(sentence_id, None)
                    _trace_event(
                        "sentence_upgrade_small_commit",
                        seq=seq_no,
                        sentence_id=sentence_id,
                        source=small_upgrade_source,
                        old_chars=len(current),
                        new_chars=len(upgraded),
                        growth_chars=max(0, len(upgraded) - len(current)),
                        hits=int(
                            small_upgrade_observation.hits
                            if small_upgrade_observation is not None
                            else getattr(accepted_candidate, "hits", 0)
                        ),
                        stable_age_ms=int(
                            small_upgrade_observation.stable_ms
                            if small_upgrade_observation is not None
                            else 0
                        ),
                        candidate_hash8=_hash8(upgraded),
                        old_preview=_trace_preview(current, 96),
                        new_preview=_trace_preview(upgraded, 96),
                    )
                _trace_event(
                    "sentence_upgrade_commit",
                    seq=seq_no,
                    idx=int(i),
                    global_idx=int(global_idx),
                    sentence_id=sentence_id,
                    revision=int(revision),
                    old_chars=len(current),
                    new_chars=len(upgraded),
                    preserved_translation_chars=len(preserved_translation.strip()),
                    slice_commit=bool(slice_commit),
                    context_correction=bool(accepted_context_correction and not accepted_upgrade),
                )
                _trace_event(
                    "candidate_action",
                    seq=seq_no,
                    idx=int(i),
                    action="upgrade",
                    sentence_id=sentence_id,
                    revision=int(revision),
                    text_hash8=_hash8(upgraded),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                )
                _track_committed_sentence(
                    upgraded,
                    int(seq_no or 0),
                    "sentence_updated",
                    sentence_id,
                )
                _trace_text_pool(
                    "pool_solidified_update",
                    phase="solidified",
                    text=upgraded,
                    reason="sentence_updated",
                    seq_hint=int(seq_hint or 0),
                    sentence_id=sentence_id,
                    revision=int(revision),
                    delta_chars=max(0, len(upgraded) - len(current)),
                    slice_commit=bool(slice_commit),
                )
                event = {
                    "type": "sentence_updated",
                    "sentence_id": sentence_id,
                    "revision": int(revision),
                    "text": upgraded,
                    "language": str(language or ""),
                    "seq": int(seq_hint or 0),
                    "ts_ms": int(time.time() * 1000),
                    "slice_commit": bool(slice_commit),
                }
                _attach_stability(
                    event,
                    is_stable=True,
                    phase="solidified",
                    reason="sentence_updated",
                    sentence_id=sentence_id,
                )
                await _send_json(event)
                if translate_now:
                    await _translate_sentence_now(
                        sentence_id,
                        revision,
                        upgraded,
                        language,
                        seq_hint,
                    )
                else:
                    _request_sentence_translation(sentence_id, revision, upgraded, language, seq_hint)

            for stale_sentence_id in list(subtitle_state.deferred_sentence_upgrades):
                if stale_sentence_id not in evaluated_sentence_ids:
                    _reset_deferred_sentence_upgrade(
                        stale_sentence_id,
                        seq_no=seq_no,
                        reason="candidate_disappeared",
                    )

            for i in range(effective_committed_count, ready_end):
                sentence = str(completed[i] or "").strip()
                if not sentence:
                    _record_completed_candidate(i, "", "")
                    continue
                original_sentence = sentence
                if boundary_context and subtitle_state.committed_sentences:
                    prev_for_overlap = str(subtitle_state.committed_sentences[-1] or "").strip()
                    trimmed_sentence = _trim_leading_boundary_overlap(prev_for_overlap, sentence)
                    if trimmed_sentence != sentence:
                        _trace_event(
                            "leading_boundary_overlap_trimmed",
                            seq=seq_no,
                            idx=int(i),
                            prev_chars=len(prev_for_overlap),
                            old_chars=len(sentence),
                            new_chars=len(trimmed_sentence),
                            old_hash8=_hash8(sentence),
                            new_hash8=_hash8(trimmed_sentence),
                            slice_commit=bool(slice_commit),
                        )
                        sentence = trimmed_sentence
                        completed[i] = sentence
                    if not sentence:
                        _record_completed_candidate(i, original_sentence, "")
                        _trace_event(
                            "candidate_action",
                            seq=seq_no,
                            idx=int(i),
                            action="structural_overlap_skip",
                            sentence_id="",
                            text_hash8=_hash8(original_sentence),
                            segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                        )
                        continue
                candidate_compact = _compact_asr_compare_text(sentence)
                covered_min_chars = 6 if _has_cjk(sentence) else 12
                if (
                    canonical_segment_correction
                    and len(candidate_compact) >= int(covered_min_chars)
                    and _consume_unmatched_compact_occurrence(
                        canonical_existing_compact,
                        candidate_compact,
                        canonical_covered_spans,
                    )
                ):
                    _record_completed_candidate(i, sentence, "")
                    _trace_event(
                        "canonical_covered_commit_suppressed",
                        seq=seq_no,
                        idx=int(i),
                        sentence_chars=len(sentence),
                        existing_chars=len(canonical_existing_text),
                        text_hash8=_hash8(sentence),
                        segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    )
                    _trace_event(
                        "candidate_action",
                        seq=seq_no,
                        idx=int(i),
                        action="canonical_covered_skip",
                        sentence_id="",
                        text_hash8=_hash8(sentence),
                        segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                    )
                    continue
                sentence_id = _new_sentence_id()
                revision = 1
                commit_ts_ms = int(time.time() * 1000)
                subtitle_state.committed_sentences.append(sentence)
                subtitle_state.sentence_items.append(
                    {
                        "id": sentence_id,
                        "zh": sentence,
                        "en": "",
                        "revision": int(revision),
                        "ts_ms": int(commit_ts_ms),
                        "seq": int(seq_hint or 0),
                    }
                )
                await _register_tts_source(sentence_id, revision)
                committed_added += 1
                _trace_event(
                    "sentence_new_commit",
                    seq=seq_no,
                    idx=int(i),
                    global_idx=int(len(subtitle_state.committed_sentences) - 1),
                    sentence_id=sentence_id,
                    revision=int(revision),
                    chars=len(sentence),
                    slice_commit=bool(slice_commit),
                )
                _track_committed_sentence(
                    sentence,
                    int(seq_no or 0),
                    "sentence_committed",
                    sentence_id,
                )
                _trace_text_pool(
                    "pool_solidified_append",
                    phase="solidified",
                    text=sentence,
                    reason="sentence_committed",
                    seq_hint=int(seq_hint or 0),
                    sentence_id=sentence_id,
                    revision=int(revision),
                    delta_chars=len(sentence),
                    slice_commit=bool(slice_commit),
                )
                event = {
                    "type": "sentence_committed",
                    "sentence_id": sentence_id,
                    "revision": int(revision),
                    "text": sentence,
                    "language": str(language or ""),
                    "seq": int(seq_hint or 0),
                    "ts_ms": int(commit_ts_ms),
                    "slice_commit": bool(slice_commit),
                }
                _attach_stability(
                    event,
                    is_stable=True,
                    phase="solidified",
                    reason="sentence_committed",
                    sentence_id=sentence_id,
                )
                await _send_json(event)
                if translate_now:
                    await _translate_sentence_now(
                        sentence_id,
                        revision,
                        sentence,
                        language,
                        seq_hint,
                    )
                else:
                    _request_sentence_translation(sentence_id, revision, sentence, language, seq_hint)
                _record_completed_candidate(i, sentence, sentence_id)
                _trace_event(
                    "candidate_action",
                    seq=seq_no,
                    idx=int(i),
                    action="commit",
                    sentence_id=sentence_id,
                    revision=int(revision),
                    text_hash8=_hash8(sentence),
                    segment_id=int(getattr(segment_runtime, "id", 0) or 0),
                )

            if should_sample or committed_added > 0 or upgraded_count > 0 or completed_underflow > 0:
                _trace_event(
                    "commit_apply_result",
                    seq=seq_no,
                    updated_count=int(upgraded_count),
                    committed_added=int(committed_added),
                    committed_count_after=int(len(subtitle_state.committed_sentences) - commit_base),
                    total_committed_count_after=int(len(subtitle_state.committed_sentences)),
                    candidate_cursor_before=int(candidate_cursor_before),
                    candidate_cursor_after=int(
                        getattr(subtitle_state, "processed_completed_count", 0) or 0
                    ),
                    ready_new_commits=int(ready_new_commits),
                    suppressed_new_commits=int(suppressed_new_commits),
                    early_translation_promoted=bool(early_translation_promoted),
                    early_translation_waiting=bool(early_translation_waiting),
                    early_translation_short_english=bool(early_translation_short_english),
                    short_english_slice_fragment_held=bool(short_english_slice_fragment_held),
                    short_english_slice_fragment_index=int(short_english_slice_fragment_index),
                    completed_underflow=int(completed_underflow),
                    force_tail=bool(force_tail),
                    slice_commit=bool(slice_commit),
                    canonical_segment_correction=bool(canonical_segment_correction),
                )

            if pending_prefix_before and (committed_added > 0 or bool(force_tail)):
                if bool(getattr(subtitle_state, "pending_prefix_is_separate", False)):
                    _trace_event(
                        "pending_prefix_retained_for_candidate_alignment",
                        seq=seq_no,
                        pending_prefix_chars=len(pending_prefix_before),
                        terminal_chars=len(
                            str(getattr(subtitle_state, "pending_prefix_terminal_text", "") or "")
                        ),
                        committed_added=int(committed_added),
                        force_tail=bool(force_tail),
                    )
                else:
                    subtitle_state.pending_prefix_text = ""
                    subtitle_state.pending_prefix_segment_id = 0
                    subtitle_state.pending_prefix_reason = ""
                    subtitle_state.pending_prefix_miss_count = 0
                    _clear_pending_prefix_boundary_evidence()
                    _trace_event(
                        "pending_prefix_cleared_commit",
                        seq=seq_no,
                        pending_prefix_chars=len(pending_prefix_before),
                        pending_prefix_hash8=_hash8(pending_prefix_before),
                        committed_added=int(committed_added),
                        force_tail=bool(force_tail),
                    )
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
                if short_english_slice_fragment_held and pending_segments:
                    pending_segments[0] = _strip_short_english_fragment_period(
                        pending_segments[0],
                        min_words=int(early_translation_min_english_words),
                        min_chars=int(early_translation_min_english_chars),
                    )
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
                            is_stable=bool(p.get("is_stable", False)),
                            stability_phase=str((p.get("stability") or {}).get("phase", "") if isinstance(p.get("stability"), dict) else ""),
                            stability_reason=str((p.get("stability") or {}).get("reason", "") if isinstance(p.get("stability"), dict) else ""),
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
                            is_stable=bool(p.get("is_stable", False)),
                            stability_phase=str((p.get("stability") or {}).get("phase", "") if isinstance(p.get("stability"), dict) else ""),
                        )
            async with send_lock:
                await websocket.send_json(payload)

        def _drop_oldest_audio() -> bool:
            nonlocal queue_samples, audio_spill_generation, audio_spill_samples
            try:
                _, dropped_wav = audio_queue.get_nowait()
                queue_samples = max(0, int(queue_samples - int(getattr(dropped_wav, "size", 0))))
                stats.queue_dropped += 1
                return True
            except asyncio.QueueEmpty:
                if not audio_spill_parts:
                    return False
                dropped_wav = audio_spill_parts.pop(0)
                dropped_samples = int(getattr(dropped_wav, "size", 0))
                audio_spill_samples = max(0, int(audio_spill_samples - dropped_samples))
                queue_samples = max(0, int(queue_samples - dropped_samples))
                stats.queue_dropped += 1
                if not audio_spill_parts:
                    audio_spill_generation = None
                return True

        def _flush_audio_spill() -> bool:
            nonlocal audio_spill_generation, audio_spill_samples
            if not audio_spill_parts or audio_queue.full():
                return False
            spill_frames = len(audio_spill_parts)
            spill_samples = int(audio_spill_samples)
            merged = (
                audio_spill_parts[0]
                if spill_frames == 1
                else np.concatenate(audio_spill_parts, axis=0)
            )
            try:
                audio_queue.put_nowait((int(audio_spill_generation or 0), merged))
            except asyncio.QueueFull:
                return False
            audio_spill_parts.clear()
            audio_spill_generation = None
            audio_spill_samples = 0
            stats.queue_spill_flushes += 1
            _trace_event(
                "audio_queue_spill_flush",
                spill_frames=int(spill_frames),
                spill_samples=int(spill_samples),
                queue_samples=int(queue_samples),
                queue_depth=int(audio_queue.qsize()),
            )
            return True

        def _append_audio_spill(gen: int, wav: np.ndarray) -> None:
            nonlocal queue_samples, audio_spill_generation, audio_spill_samples
            if audio_spill_parts and int(audio_spill_generation or 0) != int(gen):
                raise RuntimeError("audio spill generation changed before queue reset")
            spill_started = not audio_spill_parts
            audio_spill_generation = int(gen)
            audio_spill_parts.append(wav)
            wav_samples = int(wav.size)
            audio_spill_samples += wav_samples
            queue_samples += wav_samples
            stats.queue_spill_samples_peak = max(
                int(stats.queue_spill_samples_peak),
                int(audio_spill_samples),
            )
            if spill_started:
                _trace_event(
                    "audio_queue_spill_start",
                    frame_samples=int(wav_samples),
                    queue_samples=int(queue_samples),
                    queue_depth=int(audio_queue.qsize()),
                )

        def _clear_audio_queue() -> int:
            dropped = 0
            while _drop_oldest_audio():
                dropped += 1
            return dropped

        def _enqueue_audio(gen: int, wav: np.ndarray) -> None:
            nonlocal queue_samples
            dropped_now = 0
            now_mono = time.monotonic()
            if audio_spill_parts:
                _append_audio_spill(gen, wav)
                _flush_audio_spill()
            else:
                try:
                    audio_queue.put_nowait((gen, wav))
                    queue_samples += int(wav.size)
                except asyncio.QueueFull:
                    _append_audio_spill(gen, wav)
            pressure = backpressure.evaluate(int(queue_samples))
            if pressure.drop_oldest:
                while backpressure.evaluate(int(queue_samples)).drop_oldest:
                    if not _drop_oldest_audio():
                        break
                    dropped_now += 1
                _flush_audio_spill()
            if pressure.reason == "hard_overflow":
                if float(getattr(backpressure_runtime, "hard_overflow_since", 0.0) or 0.0) <= 0.0:
                    backpressure_runtime.hard_overflow_since = now_mono
                    _trace_event(
                        "audio_backpressure_hard_start",
                        queue_sec=round(float(pressure.queue_sec), 3),
                        queue_samples=int(queue_samples),
                        queue_depth=int(audio_queue.qsize()),
                    )
            else:
                hard_since = float(getattr(backpressure_runtime, "hard_overflow_since", 0.0) or 0.0)
                if hard_since > 0.0:
                    _trace_event(
                        "audio_backpressure_hard_end",
                        reason=str(backpressure_runtime.reason),
                        duration_ms=int(max(0.0, (now_mono - hard_since) * 1000.0)),
                        queue_sec=round(float(pressure.queue_sec), 3),
                        queue_samples=int(queue_samples),
                        queue_depth=int(audio_queue.qsize()),
                    )
                backpressure_runtime.hard_overflow_since = 0.0
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
            if _should_use_high_batch_merge(
                queue_depth=depth_before,
                audio_queue_size=audio_queue_size,
                under_pressure=bool(pressure.under_pressure),
            ):
                high_scale = float(pressure.suggested_batch_scale) if bool(pressure.under_pressure) else 1.0
                target_samples = max(target_samples, int(max(1.0, high_scale) * float(consumer_high_batch_samples)))
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
            nonlocal backpressure_runtime

            def _relieve_hard_overflow_if_needed() -> None:
                hard_since = float(getattr(backpressure_runtime, "hard_overflow_since", 0.0) or 0.0)
                if hard_since <= 0.0:
                    return
                now_mono = time.monotonic()
                if (now_mono - hard_since) < float(hard_overflow_relief_sec):
                    return
                last_relief = float(getattr(backpressure_runtime, "last_relief_at", 0.0) or 0.0)
                if last_relief > 0.0 and (now_mono - last_relief) < float(hard_overflow_relief_sec):
                    return

                target_samples = int(max(0.0, float(queue_target_sec)) * SAMPLE_RATE)
                trimmed = 0
                while int(queue_samples) > int(target_samples):
                    if not _drop_oldest_audio():
                        break
                    trimmed += 1

                backpressure_runtime.last_relief_at = now_mono
                pressure = backpressure.evaluate(int(queue_samples))
                if bool(pressure.drop_oldest):
                    backpressure_runtime.hard_overflow_since = now_mono
                else:
                    backpressure_runtime.hard_overflow_since = 0.0

                if trimmed > 0:
                    _trace_event(
                        "audio_backpressure_relief",
                        reason="sustained_hard_overflow",
                        trimmed_frames=int(trimmed),
                        queue_sec=round(float(pressure.queue_sec), 3),
                        queue_samples=int(queue_samples),
                        queue_depth=int(audio_queue.qsize()),
                    )

            while not stop_consumer.is_set():
                _flush_audio_spill()
                if finish_requested and audio_queue.empty() and not audio_spill_parts:
                    break
                _relieve_hard_overflow_if_needed()
                try:
                    gen, wav = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    _flush_audio_spill()
                    _relieve_hard_overflow_if_needed()
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
                _flush_audio_spill()

                async with state_lock:
                    local_state = state
                    local_gen = state_generation
                if local_state is None or gen != local_gen:
                    continue
                wav = _coalesce_audio_frames(gen, wav)
                total_consumed_samples += int(wav.size)
                vad_signal = _update_backend_vad(wav)
                pending_text_snapshot = str(last_text_snapshot or "").strip()
                language_hint = str(getattr(local_state, "language", "") or "")
                should_skip_decode = _should_skip_stream_decode(
                    in_speech=bool(getattr(backend_vad, "in_speech", False)),
                    silence_ms=float(vad_signal.get("silence_ms", 0.0) or 0.0),
                    segment_elapsed_ms=float(vad_signal.get("segment_elapsed_ms", 0.0) or 0.0),
                    snr_db=float(vad_signal.get("snr_db", 0.0) or 0.0),
                    vad_silence_ms=float(vad_silence_trigger_ms),
                    vad_exit_snr_db=float(vad_exit_snr_db),
                    has_pending_text=bool(pending_text_snapshot),
                )
                if should_skip_decode:
                    stats.silent_decode_skipped += 1
                    if stats.silent_decode_skipped <= 3 or stats.silent_decode_skipped % 40 == 0:
                        _trace_event(
                            "silent_decode_skipped",
                            skipped=int(stats.silent_decode_skipped),
                            silence_ms=int(float(vad_signal.get("silence_ms", 0.0) or 0.0)),
                            snr_db=round(float(vad_signal.get("snr_db", 0.0) or 0.0), 2),
                            queue_depth=int(audio_queue.qsize()),
                            queue_sec=round(float(backpressure.evaluate(int(queue_samples)).queue_sec), 3),
                        )
                    await _maybe_vad_silence_cut(
                        vad_signal,
                        pending_text_snapshot,
                        language_hint,
                        int(seq or 0),
                    )
                    await _maybe_idle_tail_commit()
                    continue
                if stats.silent_decode_skipped > 0:
                    _trace_event(
                        "silent_decode_resumed",
                        skipped=int(stats.silent_decode_skipped),
                        queue_depth=int(audio_queue.qsize()),
                        queue_sec=round(float(backpressure.evaluate(int(queue_samples)).queue_sec), 3),
                    )
                stats.silent_decode_skipped = 0

                if use_vllm_streaming:
                    async with infer_lock:
                        await asyncio.to_thread(asr.streaming_transcribe, wav, local_state)
                        _guard_streaming_context_output(local_state, reason="partial")
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
                    _track_model_observed_sentences(
                        payload_text,
                        int(payload.get("seq", 0) or 0),
                        "partial_raw",
                    )
                    if seq <= 3 or seq % subtitle_trace_log_partial_every == 0:
                        _trace_event(
                            "partial_model_output",
                            seq=int(seq),
                            language=str(payload.get("language", "") or ""),
                            text_chars=len(str(payload.get("text", "") or "").strip()),
                            queue_depth=int(audio_queue.qsize()),
                        )
                    stable_text, reset_guard_hold = _stabilize_partial_text(
                        payload.get("text", ""),
                        int(payload.get("seq", 0) or 0),
                    )
                    payload["text"] = stable_text
                    payload["reset_guard_hold"] = bool(reset_guard_hold)
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
                    tentative_text = str(payload.get("tentative_text", "") or "").strip()
                    partial_is_stable = bool((not tentative_text) and not reset_guard_hold)
                    _attach_stability(
                        payload,
                        is_stable=partial_is_stable,
                        phase="solidified" if partial_is_stable else "generating",
                        reason=(
                            "reset_guard_hold"
                            if bool(reset_guard_hold)
                            else ("tentative_tail" if tentative_text else "no_tentative_tail")
                        ),
                        tentative_text=tentative_text,
                    )
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
                            context=str(
                                getattr(local_state, "streaming_context", "") or ""
                            ),
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
                _track_model_observed_sentences(
                    payload_text,
                    int(payload.get("seq", 0) or 0),
                    "partial_raw",
                )
                if seq <= 3 or seq % subtitle_trace_log_partial_every == 0:
                    _trace_event(
                        "partial_model_output",
                        seq=int(seq),
                        language=str(payload.get("language", "") or ""),
                        text_chars=len(str(payload.get("text", "") or "").strip()),
                        queue_depth=int(audio_queue.qsize()),
                    )
                stable_text, reset_guard_hold = _stabilize_partial_text(
                    payload.get("text", ""),
                    int(payload.get("seq", 0) or 0),
                )
                payload["text"] = stable_text
                payload["reset_guard_hold"] = bool(reset_guard_hold)
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
                tentative_text = str(payload.get("tentative_text", "") or "").strip()
                partial_is_stable = bool((not tentative_text) and not reset_guard_hold)
                _attach_stability(
                    payload,
                    is_stable=partial_is_stable,
                    phase="solidified" if partial_is_stable else "generating",
                    reason=(
                        "reset_guard_hold"
                        if bool(reset_guard_hold)
                        else ("tentative_tail" if tentative_text else "no_tentative_tail")
                    ),
                    tentative_text=tentative_text,
                )
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
            stop_context_final_applied = False
            if use_vllm_streaming:
                async with infer_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, local_state)
                    _guard_streaming_context_output(
                        local_state,
                        reason=str(finish_reason or finish_mode or "stop"),
                    )
                    stop_context_final_applied = await _apply_segment_final_context(
                        local_state,
                        reason=str(finish_reason or finish_mode or "stop"),
                    )
                if finish_mode == "stop" and final_redecode_on_stop:
                    if _tts_output_active():
                        await _drain_tts_translation_task("before_final_redecode")
                    async with tts_transition_lock:
                        issued_tts_jobs = int(tts_runtime.session_issued_job_count)
                    if issued_tts_jobs > 0:
                        _trace_event(
                            "final_redecode_skipped",
                            reason="tts_jobs_already_issued",
                            issued_jobs=issued_tts_jobs,
                            full_audio_samples=int(full_audio_samples),
                            cap_samples=int(final_redecode_max_samples),
                        )
                    elif segment_final_context_applied or stop_context_final_applied:
                        _trace_event(
                            "final_redecode_skipped",
                            reason="segment_final_context_active",
                            full_audio_samples=int(full_audio_samples),
                            cap_samples=int(final_redecode_max_samples),
                        )
                    elif full_audio_overflow:
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
                                context=str(
                                    getattr(local_state, "final_context", "") or ""
                                ),
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
                translation_runtime.latest_by_sentence.clear()
                await _reset_tts_ordering()
                subtitle_state.commit_base = 0
                _reset_completed_candidate_cursor()
                subtitle_state.prev_completed_sentences = []
                subtitle_state.tentative_tail = ""
                subtitle_state.pending_prefix_text = ""
                subtitle_state.pending_prefix_segment_id = 0
                subtitle_state.pending_prefix_reason = ""
                subtitle_state.pending_prefix_miss_count = 0
                _clear_pending_prefix_boundary_evidence()
                subtitle_state.boundary_anchor_text = ""
                subtitle_state.boundary_anchor_segment_id = 0
                _reset_early_translation_holdback_state()
                alignment_runtime.committed_seen = {}
                alignment_runtime.committed_events = 0
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
            _track_model_observed_sentences(
                payload.get("text", ""),
                int(payload.get("seq", 0) or 0),
                "final_raw",
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
                canonical_segment_correction=bool(stop_context_final_applied),
                final_reconcile=True,
            )
            _apply_incremental_text_fields(payload)
            payload["committed_text"] = _join_segments(subtitle_state.committed_sentences)
            final_text_norm = _normalize_sentence_for_duplicate_compare(str(payload.get("text", "") or ""))
            committed_text_norm = _normalize_sentence_for_duplicate_compare(str(payload.get("committed_text", "") or ""))
            reconcile_allowed = bool(final_text_norm and final_text_norm != committed_text_norm)
            if reconcile_allowed and committed_text_norm and not canonical_redecode_applied:
                reconcile_allowed = False
                _trace_event(
                    "final_commit_reconcile_skipped_noncanonical",
                    seq=int(payload.get("seq", 0) or 0),
                    final_chars=len(str(payload.get("text", "") or "").strip()),
                    committed_chars=len(str(payload.get("committed_text", "") or "").strip()),
                    finish_mode=str(finish_mode),
                    finish_reason=str(finish_reason),
                )
            if (
                reconcile_allowed
                and committed_text_norm
                and committed_text_norm.endswith(final_text_norm)
                and len(committed_text_norm) >= len(final_text_norm) + 6
            ):
                reconcile_allowed = False
                _trace_event(
                    "final_commit_reconcile_skipped_suffix_superstring",
                    seq=int(payload.get("seq", 0) or 0),
                    final_chars=len(str(payload.get("text", "") or "").strip()),
                    committed_chars=len(str(payload.get("committed_text", "") or "").strip()),
                )
            if reconcile_allowed:
                _trace_event(
                    "final_commit_reconcile_start",
                    seq=int(payload.get("seq", 0) or 0),
                    final_chars=len(str(payload.get("text", "") or "").strip()),
                    committed_chars=len(str(payload.get("committed_text", "") or "").strip()),
                    final_hash8=_hash8(str(payload.get("text", "") or "")),
                    committed_hash8=_hash8(str(payload.get("committed_text", "") or "")),
                )
                subtitle_state.stream_uid = f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}"
                subtitle_state.next_sentence_id = 1
                subtitle_state.committed_sentences = []
                subtitle_state.sentence_items = []
                translation_runtime.latest_by_sentence.clear()
                await _reset_tts_ordering()
                subtitle_state.commit_base = 0
                _reset_completed_candidate_cursor()
                subtitle_state.prev_completed_sentences = []
                subtitle_state.tentative_tail = ""
                subtitle_state.pending_prefix_text = ""
                subtitle_state.pending_prefix_segment_id = 0
                subtitle_state.pending_prefix_reason = ""
                subtitle_state.pending_prefix_miss_count = 0
                _clear_pending_prefix_boundary_evidence()
                subtitle_state.boundary_anchor_text = ""
                subtitle_state.boundary_anchor_segment_id = 0
                _reset_early_translation_holdback_state()
                alignment_runtime.committed_seen = {}
                alignment_runtime.committed_events = 0
                await _send_json({"type": "sentence_reset", "reason": "final_commit_reconcile"})
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
                payload["committed_text"] = _join_segments(subtitle_state.committed_sentences)
                _trace_event(
                    "final_commit_reconcile_done",
                    seq=int(payload.get("seq", 0) or 0),
                    final_chars=len(str(payload.get("text", "") or "").strip()),
                    committed_chars=len(str(payload.get("committed_text", "") or "").strip()),
                    committed_count=int(len(subtitle_state.committed_sentences)),
                )
            if _tts_output_active():
                await _drain_tts_translation_task("before_final")
                await _drain_tts_stability(force=True)
            else:
                translation_task = translation_runtime.task
                if translation_task is not None and not translation_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(translation_task),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        pass
                    except Exception:
                        pass
            payload["translation"] = _committed_translation_text()
            _attach_stability(
                payload,
                is_stable=True,
                phase="final",
                reason=str(finish_reason or finish_mode or "final"),
                tentative_text=str(payload.get("tentative_text", "") or ""),
            )
            finished = True
            stats.final_msgs += 1
            await _send_json(payload)
            _set_tts_producer_active(False)

        try:
            if use_vllm_streaming:
                state = await asyncio.to_thread(
                    _new_vllm_state,
                    session_force_language,
                    0.0,
                    session_context_terms,
                )
                _trace_asr_context(
                    session_force_language,
                    0.0,
                    session_context_terms,
                )
            else:
                state = _new_transformers_state(
                    session_force_language,
                    0.0,
                    session_context_terms,
                )

            await _send_json(
                {
                    "type": "ready",
                    "sample_rate": SAMPLE_RATE,
                    "translation_direction": str(translation_runtime.direction or "zh2en"),
                    "translation_source_language": str(translation_runtime.source_language or ""),
                    "translation_target_language": str(translation_runtime.target_language or ""),
                    "tts_available": bool(tts_runtime.available),
                    "tts_enabled": bool(tts_runtime.enabled),
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
                        requested_force_language = session_force_language
                        if "language" in payload:
                            requested_force_language = _normalize_force_language(payload.get("language"))
                        requested_translation_direction = _normalize_translation_direction(
                            payload.get("translation_direction", translation_runtime.direction)
                        )
                        requested_tts_enabled = bool(payload.get("tts_enabled", False))
                        requested_tts_client_id = str(payload.get("tts_client_id", "") or "")
                        requested_context_terms: Optional[Tuple[str, ...]] = None
                        try:
                            if requested_tts_enabled:
                                _validated_tts_client_id(requested_tts_client_id)
                            if "asr_context_terms" in payload:
                                requested_context_terms = normalize_session_context_terms(
                                    payload.get("asr_context_terms"),
                                    max_terms=asr_context_max_terms,
                                    max_chars=asr_context_max_chars,
                                )
                        except ValueError as exc:
                            stats.last_error = f"start failed: {type(exc).__name__}"
                            _trace_event(
                                "start_failed",
                                **_safe_exception_trace_fields(exc),
                            )
                            await _send_json(
                                {"type": "error", "message": f"start failed: {exc}"}
                            )
                            continue

                        requested_effective_terms = _context_terms_for(
                            requested_force_language,
                            0.0,
                            requested_context_terms,
                        )
                        context_text = " ".join(requested_effective_terms)
                        context_metadata = {
                            "context_source": (
                                "session" if requested_context_terms is not None else "schedule"
                            ),
                            "context_active": bool(requested_effective_terms),
                            "context_term_count": len(requested_effective_terms),
                            "context_chars": len(context_text),
                            "context_sha256": hashlib.sha256(
                                context_text.encode("utf-8")
                            ).hexdigest(),
                        }
                        _trace_event(
                            "start_received",
                            requested_language=str(payload.get("language", "") or ""),
                            requested_translation_direction=str(
                                payload.get("translation_direction", "") or ""
                            ),
                            **context_metadata,
                        )
                        try:
                            if use_vllm_streaming:
                                new_state = await asyncio.to_thread(
                                    _new_vllm_state,
                                    requested_force_language,
                                    0.0,
                                    requested_context_terms,
                                )
                                _trace_asr_context(
                                    requested_force_language,
                                    0.0,
                                    requested_context_terms,
                                )
                            else:
                                new_state = _new_transformers_state(
                                    requested_force_language,
                                    0.0,
                                    requested_context_terms,
                                )

                            session_force_language = requested_force_language
                            session_context_terms = requested_context_terms
                            finish_mode = "stop"
                            finish_reason = "stop"
                            finish_requested = False
                            finished = False
                            stream_text_state.last_text = ""
                            stream_text_state.accepted_text = ""
                            stream_text_state.reset_candidate_text = ""
                            stream_text_state.reset_candidate_hits = 0
                            stream_text_state.reset_candidate_since = 0.0
                            alignment_runtime.model_seen = {}
                            alignment_runtime.committed_seen = {}
                            alignment_runtime.model_observed_events = 0
                            alignment_runtime.committed_events = 0
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
                            _reset_punct_cut_state("start")
                            full_audio_parts = []
                            full_audio_samples = 0
                            full_audio_overflow = False
                            segment_final_context_applied = False
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
                            translation_runtime.latest_by_sentence.clear()
                            translation_runtime.direction = requested_translation_direction
                            (
                                translation_runtime.source_language,
                                translation_runtime.target_language,
                            ) = _resolve_direction_languages(requested_translation_direction)
                            tts_runtime.session_issued_job_count = 0
                            await _configure_tts(
                                requested_tts_enabled,
                                requested_tts_client_id,
                                emit=False,
                                force_reset=True,
                            )
                            _set_tts_producer_active(True)
                            subtitle_state.stream_uid = f"{int(time.time() * 1000)}-{int(time.monotonic_ns() % 1000000)}"
                            subtitle_state.next_sentence_id = 1
                            subtitle_state.committed_sentences = []
                            subtitle_state.sentence_items = []
                            subtitle_state.commit_base = 0
                            _reset_completed_candidate_cursor()
                            subtitle_state.prev_completed_sentences = []
                            subtitle_state.tentative_tail = ""
                            subtitle_state.pending_prefix_text = ""
                            subtitle_state.pending_prefix_segment_id = 0
                            subtitle_state.pending_prefix_reason = ""
                            subtitle_state.pending_prefix_miss_count = 0
                            _clear_pending_prefix_boundary_evidence()
                            subtitle_state.boundary_anchor_text = ""
                            subtitle_state.boundary_anchor_segment_id = 0
                            _reset_early_translation_holdback_state()
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
                                **context_metadata,
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
                                    "tts_available": bool(tts_runtime.available),
                                    "tts_enabled": bool(tts_runtime.enabled),
                                    "asr_context_active": bool(requested_effective_terms),
                                    "asr_context_term_count": len(requested_effective_terms),
                                    "asr_context_chars": len(context_text),
                                }
                            )
                        except Exception as exc:
                            stats.last_error = f"start failed: {type(exc).__name__}"
                            _trace_event(
                                "start_failed",
                                **_safe_exception_trace_fields(exc),
                            )
                            await _send_json(
                                {
                                    "type": "error",
                                    "message": "start failed: backend state initialization failed",
                                }
                            )
                        continue

                    if msg_type == "set_translation_direction":
                        requested_direction = _normalize_translation_direction(payload.get("translation_direction"))
                        await _set_translation_direction(
                            requested_direction,
                            clear_pending=True,
                            emit=True,
                        )
                        continue

                    if msg_type == "set_tts_enabled":
                        try:
                            await _configure_tts(
                                bool(payload.get("enabled", False)),
                                str(payload.get("tts_client_id", "") or ""),
                                emit=True,
                            )
                        except HTTPException as exc:
                            await _send_json(
                                {"type": "error", "message": str(exc.detail)}
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
                        _trace_event(
                            "finish_queue_preserved",
                            queue_depth=int(audio_queue.qsize()),
                            note="drain_for_tail_accuracy",
                        )
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
            await _stop_tts_stability_scheduler(reason="ws_close")
            _set_tts_producer_active(False)
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
            _emit_alignment_summary("ws_close", int(seq or 0))
            _trace_event(
                "ws_close",
                active_connections=int(runtime.active_connections),
                finished=bool(finished),
                raw_frames=int(stats.raw_frames),
                partial_msgs=int(stats.partial_msgs),
                final_msgs=int(stats.final_msgs),
                queue_dropped=int(stats.queue_dropped),
                queue_depth_peak=int(stats.queue_depth_peak),
                queue_spill_flushes=int(stats.queue_spill_flushes),
                queue_spill_samples_peak=int(stats.queue_spill_samples_peak),
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
            if trace_file_handle is not None:
                with suppress(Exception):
                    trace_file_handle.close()

    return app


def _build_tts_synthesizer(args: argparse.Namespace, *, translator: Any) -> Any:
    if not bool(getattr(args, "enable_tts", False)):
        return None
    if translator is None:
        logger.warning("TTS disabled because translation is not enabled")
        return None

    runtime_modules = ("onnxruntime", "kokoro_onnx", "misaki", "soundfile")
    missing_modules = [name for name in runtime_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        logger.warning(
            "TTS disabled because optional runtime modules are missing: %s",
            ", ".join(missing_modules),
        )
        return None

    try:
        config = KokoroTTSConfig(
            english_model_path=Path(str(args.tts_en_model_path)).expanduser(),
            english_voices_path=Path(str(args.tts_en_voices_path)).expanduser(),
            chinese_model_path=Path(str(args.tts_zh_model_path)).expanduser(),
            chinese_voices_path=Path(str(args.tts_zh_voices_path)).expanduser(),
            chinese_config_path=Path(str(args.tts_zh_vocab_path)).expanduser(),
            english_voice=str(args.tts_en_voice),
            chinese_voice=str(args.tts_zh_voice),
            speed=float(args.tts_speed),
            cpu_threads=int(args.tts_cpu_threads),
            max_chars=int(args.tts_max_text_chars),
        )
        return KokoroOnnxSynthesizer(config=config)
    except (TTSConfigurationError, ImportError) as exc:
        logger.warning("TTS disabled because initialization failed: %s", exc)
    except Exception as exc:
        logger.warning("TTS disabled because initialization failed: %s", type(exc).__name__)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoxBridge Streaming Web Demo (HTTPS + WebSocket)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--backend", default="vllm", choices=["transformers", "vllm"], help="Inference backend")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8024, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    p.add_argument(
        "--mm-processor-cache-gb",
        type=_non_negative_float_arg,
        default=0.5,
        help="vLLM multimodal processor cache budget per process in GiB",
    )
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
        "--asr-context-schedule",
        default="",
        help="Optional JSON glossary schedule applied when each streaming state is created",
    )
    p.add_argument(
        "--asr-context-max-terms",
        type=int,
        default=24,
        help="Maximum glossary terms sent as ASR context for one streaming state",
    )
    p.add_argument(
        "--asr-context-max-chars",
        type=int,
        default=160,
        help="Maximum ASR context characters, truncated only at whole-term boundaries",
    )
    p.add_argument(
        "--asr-context-lookaround-sec",
        type=float,
        default=30.0,
        help="Include glossary windows this many seconds before or after current audio time",
    )
    p.add_argument(
        "--asr-context-apply-mode",
        default="segment_final",
        choices=["segment_final", "streaming"],
        help=(
            "Apply glossary context once to complete segments (safe default), or expose it to every "
            "streaming decode for compatibility"
        ),
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
        default=128,
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
    p.add_argument(
        "--early-translation-stable-sec",
        type=float,
        default=0.8,
        help="Seconds a held-back completed sentence must remain unchanged before early commit/translation",
    )
    p.add_argument(
        "--early-translation-stable-hits",
        type=int,
        default=3,
        help="Consecutive observations required before early commit/translation of the newest completed sentence",
    )
    p.add_argument(
        "--early-translation-short-stable-sec",
        type=float,
        default=1.2,
        help="Stricter unchanged duration for early translation of short English terminal sentences",
    )
    p.add_argument(
        "--early-translation-short-stable-hits",
        type=int,
        default=4,
        help="Stricter observation count for early translation of short English terminal sentences",
    )
    p.add_argument(
        "--early-translation-min-english-words",
        type=int,
        default=6,
        help="Do not early-commit English completed sentences shorter than this many words unless they also meet the char threshold",
    )
    p.add_argument(
        "--early-translation-min-english-chars",
        type=int,
        default=32,
        help="Do not early-commit English completed sentences shorter than this many characters unless they also meet the word threshold",
    )
    p.add_argument(
        "--enable-tts",
        action="store_true",
        help="Enable optional CPU Kokoro speech for stable translated sentences",
    )
    p.add_argument(
        "--tts-en-model-path",
        default="/data/Qwen3-ASR/models/kokoro/kokoro-v1.0.onnx",
        help="Kokoro v1.0 English ONNX model",
    )
    p.add_argument(
        "--tts-en-voices-path",
        default="/data/Qwen3-ASR/models/kokoro/voices-v1.0.bin",
        help="Kokoro v1.0 English voices",
    )
    p.add_argument(
        "--tts-zh-model-path",
        default="/data/Qwen3-ASR/models/kokoro/kokoro-v1.1-zh.onnx",
        help="Kokoro v1.1 Chinese ONNX model",
    )
    p.add_argument(
        "--tts-zh-voices-path",
        default="/data/Qwen3-ASR/models/kokoro/voices-v1.1-zh.bin",
        help="Kokoro v1.1 Chinese voices",
    )
    p.add_argument(
        "--tts-zh-vocab-path",
        default="/data/Qwen3-ASR/models/kokoro/config-v1.1-zh.json",
        help="Kokoro v1.1 Chinese vocabulary config",
    )
    p.add_argument("--tts-en-voice", default="af_heart", help="English Kokoro voice")
    p.add_argument("--tts-zh-voice", default="zf_001", help="Chinese Kokoro voice")
    p.add_argument("--tts-speed", type=float, default=1.05, help="Kokoro speaking speed")
    p.add_argument(
        "--tts-cpu-threads",
        type=int,
        default=4,
        help="ONNX Runtime intra-op CPU threads for Kokoro",
    )
    p.add_argument(
        "--tts-max-text-chars",
        type=int,
        default=1000,
        help="Reject a single TTS job above this translated character count",
    )
    p.add_argument(
        "--tts-job-ttl-sec",
        type=float,
        default=1800.0,
        help="Maximum lifetime of an unacknowledged in-memory TTS job",
    )
    p.add_argument(
        "--tts-max-client-jobs",
        type=int,
        default=4096,
        help="Maximum unread TTS jobs retained for one browser client",
    )
    p.add_argument(
        "--tts-listener-queue-size",
        type=_positive_int_arg,
        default=128,
        help="Maximum unread TTS metadata events buffered per listener device",
    )
    p.add_argument(
        "--tts-final-translation-drain-sec",
        type=float,
        default=30.0,
        help="Warn after this long while continuing to wait for stable translations before final",
    )
    p.add_argument(
        "--tts-revision-stable-sec",
        type=_non_negative_float_arg,
        default=3.0,
        help="Require this long without a source revision before publishing translated speech",
    )

    p.add_argument("--client-chunk-ms", type=int, default=200, help="Client capture chunk length in milliseconds")
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
        "--backpressure-hard-relief-sec",
        type=float,
        default=6.0,
        help="If hard overflow persists beyond this duration, trim queue down to target seconds.",
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
        "--punct-cut-start-sec",
        type=float,
        default=0.0,
        help="Deprecated no-op (punctuation-timeout cut removed).",
    )
    p.add_argument(
        "--punct-cut-wait-sec",
        type=float,
        default=0.0,
        help="Deprecated no-op (punctuation-timeout cut removed).",
    )
    p.add_argument(
        "--punct-cut-stable-sec",
        type=float,
        default=0.45,
        help="Deprecated no-op (punctuation-timeout cut removed).",
    )
    p.add_argument(
        "--punct-cut-stable-hits",
        type=int,
        default=2,
        help="Deprecated no-op (punctuation-timeout cut removed).",
    )
    p.add_argument(
        "--punct-cut-max-carry-chars",
        type=int,
        default=12,
        help="Deprecated no-op (punctuation-timeout cut removed).",
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
        "--subtitle-trace-log-file",
        default="",
        help="Optional JSONL file path for backend subtitle trace rows",
    )
    p.add_argument(
        "--subtitle-trace-log-partial-every",
        type=int,
        default=20,
        help="Sample interval for backend partial trace events (1 means every partial)",
    )

    p.add_argument("--auth-enabled", action="store_true", help="Require login for HTTP page, websocket, and debug file access")
    p.add_argument("--auth-username", default="admin", help="Single-user login name when auth is enabled")
    p.add_argument(
        "--auth-password-hash",
        default="",
        help="PBKDF2 password hash for login; can also be set with VOXBRIDGE_AUTH_PASSWORD_HASH",
    )
    p.add_argument(
        "--auth-cookie-secure",
        action="store_true",
        help="Set Secure on the session cookie; enable when serving over HTTPS",
    )
    p.add_argument(
        "--auth-session-ttl-sec",
        type=int,
        default=12 * 60 * 60,
        help="Authenticated session lifetime in seconds",
    )
    p.add_argument(
        "--disable-debug-file",
        action="store_true",
        help="Disable the /__debug/file endpoint entirely",
    )

    p.add_argument("--ssl-certfile", default=None, help="Path to TLS certificate file (enables HTTPS/WSS)")
    p.add_argument("--ssl-keyfile", default=None, help="Path to TLS private key file")
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return p.parse_args()


def _uvicorn_run_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "ssl_certfile": args.ssl_certfile,
        "ssl_keyfile": args.ssl_keyfile,
        "access_log": False,
    }


def main() -> None:
    from qwen_asr import Qwen3ASRModel
    import torch

    global _INSTANCE_LOCK_HANDLE
    args = parse_args()
    try:
        _build_auth_config(args)
        _load_asr_context_schedule(args)
    except RuntimeError as exc:
        logger.error("auth configuration failed: %s", exc)
        raise SystemExit(2) from exc
    except ValueError as exc:
        logger.error("ASR context configuration failed: %s", exc)
        raise SystemExit(2) from exc

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
                **_vllm_model_kwargs(args),
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

        tts_synthesizer = _build_tts_synthesizer(args, translator=translator)
        if tts_synthesizer is not None:
            logger.info(
                "CPU Kokoro TTS enabled threads=%d voices=%s,%s",
                int(args.tts_cpu_threads),
                str(args.tts_en_voice),
                str(args.tts_zh_voice),
            )

        app = _create_app(
            args,
            asr,
            translator=translator,
            tts_synthesizer=tts_synthesizer,
        )

        uvicorn.run(app, **_uvicorn_run_options(args))
    finally:
        _release_instance_lock(_INSTANCE_LOCK_HANDLE)
        _INSTANCE_LOCK_HANDLE = None


if __name__ == "__main__":
    main()
