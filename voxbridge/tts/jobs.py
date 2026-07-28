from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable


class TTSJobError(Exception):
    """Base exception for TTS job operations."""


class TTSJobNotFound(TTSJobError):
    """Raised when a job is absent, expired, or owned by another session."""


class TTSQueueFull(TTSJobError):
    """Raised rather than evicting an unread job."""


@dataclass(frozen=True, slots=True)
class TTSJob:
    job_id: str
    owner_key: str
    client_id: str
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


@dataclass(frozen=True, slots=True)
class TTSReadyItem:
    sentence_id: str
    revision: int
    source_order: int
    target_language: str
    text: str


@dataclass(slots=True)
class _OrderedEntry:
    sentence_id: str
    revision: int
    source_order: int
    status: str = "waiting"
    target_language: str | None = None
    text: str | None = None


class OrderedTTSBuffer:
    """Release completed translations in source order exactly once."""

    def __init__(self) -> None:
        self._entries: dict[int, _OrderedEntry] = {}
        self._sentence_orders: dict[str, int] = {}
        self._emitted_sentence_ids: set[str] = set()
        self._next_order = 0
        self._lock = threading.RLock()

    def register(self, sentence_id: str, revision: int, source_order: int) -> None:
        if not isinstance(sentence_id, str) or not sentence_id.strip():
            raise ValueError("sentence_id must be a non-empty string")
        if revision < 0 or source_order < 0:
            raise ValueError("revision and source_order must not be negative")
        with self._lock:
            if sentence_id in self._emitted_sentence_ids or source_order < self._next_order:
                return
            previous_order = self._sentence_orders.get(sentence_id)
            if previous_order is not None and previous_order != source_order:
                raise ValueError("sentence_id cannot change source_order")
            current = self._entries.get(source_order)
            if current is not None and current.sentence_id != sentence_id:
                raise ValueError("source_order is already registered")
            if current is not None and revision <= current.revision:
                return
            self._sentence_orders[sentence_id] = source_order
            self._entries[source_order] = _OrderedEntry(sentence_id, int(revision), int(source_order))

    def mark_ready(
        self,
        sentence_id: str,
        revision: int,
        text: str,
        target_language: str,
    ) -> list[TTSReadyItem]:
        if not isinstance(text, str) or not text.strip():
            return self.mark_failed(sentence_id, revision)
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError("target_language must be a non-empty string")
        with self._lock:
            entry = self._current_entry(sentence_id, revision)
            if entry is None:
                return []
            entry.status = "ready"
            entry.text = text
            entry.target_language = target_language
            return self._drain_locked()

    def mark_failed(self, sentence_id: str, revision: int) -> list[TTSReadyItem]:
        with self._lock:
            entry = self._current_entry(sentence_id, revision)
            if entry is None:
                return []
            entry.status = "failed"
            entry.text = None
            entry.target_language = None
            return self._drain_locked()

    def _current_entry(self, sentence_id: str, revision: int) -> _OrderedEntry | None:
        order = self._sentence_orders.get(sentence_id)
        if order is None:
            return None
        entry = self._entries.get(order)
        if entry is None or entry.revision != revision:
            return None
        return entry

    def _drain_locked(self) -> list[TTSReadyItem]:
        ready: list[TTSReadyItem] = []
        while True:
            entry = self._entries.get(self._next_order)
            if entry is None or entry.status == "waiting":
                break
            del self._entries[self._next_order]
            self._sentence_orders.pop(entry.sentence_id, None)
            self._next_order += 1
            if entry.status == "failed":
                continue
            if entry.sentence_id in self._emitted_sentence_ids:
                continue
            self._emitted_sentence_ids.add(entry.sentence_id)
            ready.append(
                TTSReadyItem(
                    sentence_id=entry.sentence_id,
                    revision=entry.revision,
                    source_order=entry.source_order,
                    target_language=entry.target_language or "",
                    text=entry.text or "",
                )
            )
        return ready

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sentence_orders.clear()
            self._emitted_sentence_ids.clear()
            self._next_order = 0


class TTSJobRegistry:
    def __init__(
        self,
        *,
        ttl_sec: float,
        max_client_jobs: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        if max_client_jobs <= 0:
            raise ValueError("max_client_jobs must be positive")
        self._ttl_sec = float(ttl_sec)
        self._max_client_jobs = int(max_client_jobs)
        self._clock = clock
        self._jobs: dict[str, TTSJob] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def _prune_locked(self, now: float) -> int:
        expired_ids = [job_id for job_id, job in self._jobs.items() if job.expires_at <= now]
        for job_id in expired_ids:
            del self._jobs[job_id]
        return len(expired_ids)

    def create(
        self,
        *,
        owner_key: str,
        client_id: str,
        sentence_id: str,
        revision: int,
        source_order: int,
        target_language: str,
        text: str,
    ) -> TTSJob:
        owner_key = self._require_text(owner_key, "owner_key")
        client_id = self._require_text(client_id, "client_id")
        sentence_id = self._require_text(sentence_id, "sentence_id")
        target_language = self._require_text(target_language, "target_language")
        text = self._require_text(text, "text")
        if revision < 0:
            raise ValueError("revision must not be negative")
        if source_order < 0:
            raise ValueError("source_order must not be negative")

        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            client_job_count = sum(
                job.owner_key == owner_key and job.client_id == client_id for job in self._jobs.values()
            )
            if client_job_count >= self._max_client_jobs:
                raise TTSQueueFull("TTS job queue is full")

            job_id = secrets.token_urlsafe(24)
            while job_id in self._jobs:
                job_id = secrets.token_urlsafe(24)
            job = TTSJob(
                job_id=job_id,
                owner_key=owner_key,
                client_id=client_id,
                sentence_id=sentence_id,
                revision=int(revision),
                source_order=int(source_order),
                target_language=target_language,
                text=text,
                created_at=now,
                expires_at=now + self._ttl_sec,
            )
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str, owner_key: str) -> TTSJob:
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            job = self._jobs.get(job_id)
            if job is None or job.owner_key != owner_key:
                raise TTSJobNotFound("TTS job not found")
            return job

    def cache_audio(
        self,
        job_id: str,
        owner_key: str,
        audio_bytes: bytes,
        *,
        sample_rate: int | None = None,
        duration_ms: int | None = None,
    ) -> TTSJob:
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise ValueError("audio_bytes must be non-empty bytes")
        with self._lock:
            job = self.get(job_id, owner_key)
            cached = replace(
                job,
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
            )
            self._jobs[job_id] = cached
            return cached

    def acknowledge(self, job_id: str, owner_key: str) -> bool:
        with self._lock:
            try:
                self.get(job_id, owner_key)
            except TTSJobNotFound:
                return False
            del self._jobs[job_id]
            return True

    def cancel_client(self, owner_key: str, client_id: str) -> int:
        with self._lock:
            self._prune_locked(self._clock())
            matching_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.owner_key == owner_key and job.client_id == client_id
            ]
            for job_id in matching_ids:
                del self._jobs[job_id]
            return len(matching_ids)

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked(self._clock())
