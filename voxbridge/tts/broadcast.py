from __future__ import annotations

import asyncio
import secrets
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from .jobs import TTSReadyItem


class TTSBroadcastError(Exception):
    """Base exception for translated-speech broadcast operations."""


class TTSBroadcastNotFound(TTSBroadcastError):
    """Raised when a job or listener assignment is unavailable."""


class TTSBroadcastQueueFull(TTSBroadcastError):
    """Raised rather than evicting a job that a listener has not received."""


@dataclass(frozen=True, slots=True)
class BroadcastTTSJob:
    job_id: str
    sentence_id: str
    revision: int
    source_order: int
    target_language: str
    text: str
    created_at: float
    expires_at: float
    audio_bytes: bytes | None = None
    sample_rate: int | None = None
    duration_ms: int | None = None


@dataclass(slots=True)
class TTSListenerSubscription:
    listener_id: str
    owner_key: str
    queue: asyncio.Queue[dict[str, object]]
    overflowed: asyncio.Event


@dataclass(slots=True)
class _BroadcastJobState:
    job: BroadcastTTSJob
    pending_listener_ids: set[str]
    in_flight: int = 0


class TTSBroadcastHub:
    """Fan stable translations out to independent, future-only listeners."""

    def __init__(
        self,
        *,
        ttl_sec: float,
        max_jobs: int,
        listener_queue_size: int,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        if listener_queue_size <= 0:
            raise ValueError("listener_queue_size must be positive")
        self._ttl_sec = float(ttl_sec)
        self._max_jobs = int(max_jobs)
        self._listener_queue_size = int(listener_queue_size)
        self._clock = clock
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._listeners: dict[str, TTSListenerSubscription] = {}
        self._jobs: dict[str, _BroadcastJobState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def _new_token_locked(self) -> str:
        while True:
            token = self._require_text(self._token_factory(), "generated token")
            if token not in self._listeners and token not in self._jobs:
                return token

    def _cleanup_job_locked(self, job_id: str, now: float | None = None) -> bool:
        state = self._jobs.get(job_id)
        if state is None or state.in_flight > 0:
            return False
        current = self._clock() if now is None else float(now)
        if state.pending_listener_ids and state.job.expires_at > current:
            return False
        del self._jobs[job_id]
        return True

    def _prune_locked(self, now: float) -> int:
        removed = 0
        for job_id, state in list(self._jobs.items()):
            if state.job.expires_at <= now and state.in_flight == 0:
                del self._jobs[job_id]
                removed += 1
        return removed

    def _unregister_locked(self, listener_id: str, owner_key: str) -> int:
        subscription = self._listeners.get(listener_id)
        if subscription is None or subscription.owner_key != owner_key:
            return 0
        del self._listeners[listener_id]
        removed_assignments = 0
        now = self._clock()
        for job_id, state in list(self._jobs.items()):
            if listener_id in state.pending_listener_ids:
                state.pending_listener_ids.remove(listener_id)
                removed_assignments += 1
            self._cleanup_job_locked(job_id, now)
        return removed_assignments

    @property
    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    @property
    def job_count(self) -> int:
        with self._lock:
            self._prune_locked(self._clock())
            return len(self._jobs)

    def register(self, owner_key: str) -> TTSListenerSubscription:
        owner = self._require_text(owner_key, "owner_key")
        with self._lock:
            self._prune_locked(self._clock())
            listener_id = self._new_token_locked()
            subscription = TTSListenerSubscription(
                listener_id=listener_id,
                owner_key=owner,
                queue=asyncio.Queue(maxsize=self._listener_queue_size),
                overflowed=asyncio.Event(),
            )
            self._listeners[listener_id] = subscription
            return subscription

    def unregister(self, listener_id: str, owner_key: str) -> int:
        listener = self._require_text(listener_id, "listener_id")
        owner = self._require_text(owner_key, "owner_key")
        with self._lock:
            return self._unregister_locked(listener, owner)

    def publish(self, item: TTSReadyItem) -> BroadcastTTSJob | None:
        sentence_id = self._require_text(item.sentence_id, "sentence_id")
        target_language = self._require_text(item.target_language, "target_language")
        text = self._require_text(item.text, "text")
        if item.revision < 0 or item.source_order < 0:
            raise ValueError("revision and source_order must not be negative")

        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            if not self._listeners:
                return None
            if len(self._jobs) >= self._max_jobs:
                raise TTSBroadcastQueueFull("TTS broadcast job queue is full")

            listener_ids = set(self._listeners)
            job = BroadcastTTSJob(
                job_id=self._new_token_locked(),
                sentence_id=sentence_id,
                revision=int(item.revision),
                source_order=int(item.source_order),
                target_language=target_language,
                text=text,
                created_at=now,
                expires_at=now + self._ttl_sec,
            )
            self._jobs[job.job_id] = _BroadcastJobState(
                job=job,
                pending_listener_ids=listener_ids,
            )
            event: dict[str, object] = {
                "type": "tts_job",
                "job_id": job.job_id,
                "sentence_id": job.sentence_id,
                "revision": job.revision,
                "source_order": job.source_order,
                "target_language": job.target_language,
                "is_stable": True,
            }
            overflowed: list[TTSListenerSubscription] = []
            for listener_id in listener_ids:
                subscription = self._listeners.get(listener_id)
                if subscription is None:
                    continue
                try:
                    subscription.queue.put_nowait(dict(event))
                except asyncio.QueueFull:
                    subscription.overflowed.set()
                    overflowed.append(subscription)
            for subscription in overflowed:
                self._unregister_locked(subscription.listener_id, subscription.owner_key)

            if job.job_id not in self._jobs:
                return None
            return self._jobs[job.job_id].job

    def claim_audio(
        self,
        job_id: str,
        listener_id: str,
        owner_key: str,
    ) -> BroadcastTTSJob:
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            state = self._jobs.get(str(job_id or ""))
            subscription = self._listeners.get(str(listener_id or ""))
            if (
                state is None
                or state.job.expires_at <= now
                or subscription is None
                or subscription.owner_key != str(owner_key or "")
                or subscription.listener_id not in state.pending_listener_ids
            ):
                raise TTSBroadcastNotFound("TTS broadcast job not found")
            state.in_flight += 1
            return state.job

    def release_audio(self, job_id: str) -> None:
        with self._lock:
            state = self._jobs.get(str(job_id or ""))
            if state is None:
                return
            if state.in_flight > 0:
                state.in_flight -= 1
            self._cleanup_job_locked(state.job.job_id)

    def cache_audio(
        self,
        job_id: str,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        duration_ms: int,
    ) -> BroadcastTTSJob:
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise ValueError("audio_bytes must be non-empty bytes")
        if sample_rate <= 0 or duration_ms < 0:
            raise ValueError("invalid audio metadata")
        with self._lock:
            state = self._jobs.get(str(job_id or ""))
            if state is None:
                raise TTSBroadcastNotFound("TTS broadcast job not found")
            state.job = replace(
                state.job,
                audio_bytes=audio_bytes,
                sample_rate=int(sample_rate),
                duration_ms=int(duration_ms),
            )
            return state.job

    def acknowledge(self, job_id: str, listener_id: str, owner_key: str) -> bool:
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            state = self._jobs.get(str(job_id or ""))
            subscription = self._listeners.get(str(listener_id or ""))
            if (
                state is None
                or subscription is None
                or subscription.owner_key != str(owner_key or "")
                or subscription.listener_id not in state.pending_listener_ids
            ):
                return False
            state.pending_listener_ids.remove(subscription.listener_id)
            self._cleanup_job_locked(state.job.job_id, now)
            return True

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked(self._clock())
