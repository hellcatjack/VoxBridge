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
    release_reason: str = "quiet_window"
    source_quiet_age_ms: int = 0
    translation_ready_age_ms: int = 0


@dataclass(frozen=True, slots=True)
class TTSRevisionRegistration:
    accepted: bool
    reset: bool
    late_after_release: bool
    sentence_id: str
    revision: int
    source_order: int
    previous_revision: int | None = None
    previous_quiet_age_ms: int = 0
    previous_ready: bool = False
    released_revision: int | None = None
    elapsed_since_release_ms: int = 0


@dataclass(frozen=True, slots=True)
class TTSWaitState:
    sentence_id: str
    revision: int
    source_order: int
    quiet_age_ms: int
    required_quiet_ms: int
    remaining_ms: int
    blocked_by_earlier: bool


@dataclass(slots=True)
class _RevisionStableEntry:
    sentence_id: str
    revision: int
    source_order: int
    changed_at: float
    status: str = "waiting"
    target_language: str | None = None
    text: str | None = None
    translation_ready_at: float | None = None


class RevisionStableTTSBuffer:
    """Release current translated revisions after a monotonic quiet window."""

    def __init__(
        self,
        *,
        stable_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stable_sec < 0:
            raise ValueError("stable_sec must not be negative")
        self._stable_sec = float(stable_sec)
        self._clock = clock
        self._entries: dict[int, _RevisionStableEntry] = {}
        self._sentence_orders: dict[str, int] = {}
        self._order_sentences: dict[int, str] = {}
        self._released: dict[str, tuple[int, int, float]] = {}
        self._next_order = 0
        self._lock = threading.RLock()

    @staticmethod
    def _require_sentence_id(sentence_id: str) -> str:
        sid = str(sentence_id or "").strip()
        if not sid:
            raise ValueError("sentence_id must be a non-empty string")
        return sid

    @staticmethod
    def _elapsed_ms(start: float, end: float) -> int:
        return int(round(max(0.0, float(end) - float(start)) * 1000.0))

    def register(
        self,
        sentence_id: str,
        revision: int,
        source_order: int,
    ) -> TTSRevisionRegistration:
        sid = self._require_sentence_id(sentence_id)
        if revision < 0 or source_order < 0:
            raise ValueError("revision and source_order must not be negative")
        now = self._clock()
        with self._lock:
            known_order = self._sentence_orders.get(sid)
            if known_order is not None and known_order != source_order:
                raise ValueError("sentence_id cannot change source_order")
            known_sentence = self._order_sentences.get(source_order)
            if known_sentence is not None and known_sentence != sid:
                raise ValueError("source_order is already registered")

            released = self._released.get(sid)
            if released is not None:
                released_revision, released_order, released_at = released
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=revision > released_revision,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(released_order),
                    released_revision=int(released_revision),
                    elapsed_since_release_ms=self._elapsed_ms(released_at, now),
                )

            if source_order < self._next_order:
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                )

            current = self._entries.get(source_order)
            if current is not None and revision <= current.revision:
                return TTSRevisionRegistration(
                    accepted=False,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                    previous_revision=int(current.revision),
                )

            self._sentence_orders[sid] = int(source_order)
            self._order_sentences[int(source_order)] = sid
            if current is None:
                self._entries[source_order] = _RevisionStableEntry(
                    sid,
                    int(revision),
                    int(source_order),
                    float(now),
                )
                return TTSRevisionRegistration(
                    accepted=True,
                    reset=False,
                    late_after_release=False,
                    sentence_id=sid,
                    revision=int(revision),
                    source_order=int(source_order),
                )

            previous_revision = int(current.revision)
            previous_ready = current.status == "ready"
            previous_quiet_age_ms = self._elapsed_ms(current.changed_at, now)
            self._entries[source_order] = _RevisionStableEntry(
                sid,
                int(revision),
                int(source_order),
                float(now),
            )
            return TTSRevisionRegistration(
                accepted=True,
                reset=True,
                late_after_release=False,
                sentence_id=sid,
                revision=int(revision),
                source_order=int(source_order),
                previous_revision=previous_revision,
                previous_quiet_age_ms=previous_quiet_age_ms,
                previous_ready=previous_ready,
            )

    def mark_ready(
        self,
        sentence_id: str,
        revision: int,
        text: str,
        target_language: str,
    ) -> bool:
        sid = self._require_sentence_id(sentence_id)
        translated = str(text or "").strip()
        language = str(target_language or "").strip()
        if translated and not language:
            raise ValueError("target_language must be a non-empty string")
        now = self._clock()
        with self._lock:
            entry = self._current_entry(sid, int(revision))
            if entry is None:
                return False
            if not translated:
                entry.status = "failed"
                entry.text = None
                entry.target_language = None
                entry.translation_ready_at = None
                return True
            entry.status = "ready"
            entry.text = translated
            entry.target_language = language
            entry.translation_ready_at = float(now)
            return True

    def mark_failed(self, sentence_id: str, revision: int) -> bool:
        sid = self._require_sentence_id(sentence_id)
        with self._lock:
            entry = self._current_entry(sid, int(revision))
            if entry is None:
                return False
            entry.status = "failed"
            entry.text = None
            entry.target_language = None
            entry.translation_ready_at = None
            return True

    def _current_entry(
        self,
        sentence_id: str,
        revision: int,
    ) -> _RevisionStableEntry | None:
        order = self._sentence_orders.get(sentence_id)
        if order is None:
            return None
        entry = self._entries.get(order)
        if entry is None or entry.revision != revision:
            return None
        return entry

    def wait_state(self, sentence_id: str) -> TTSWaitState | None:
        sid = self._require_sentence_id(sentence_id)
        now = self._clock()
        with self._lock:
            order = self._sentence_orders.get(sid)
            entry = self._entries.get(order) if order is not None else None
            if entry is None or entry.status != "ready":
                return None
            quiet_age_ms = self._elapsed_ms(entry.changed_at, now)
            required_ms = int(round(self._stable_sec * 1000.0))
            return TTSWaitState(
                sentence_id=sid,
                revision=int(entry.revision),
                source_order=int(entry.source_order),
                quiet_age_ms=quiet_age_ms,
                required_quiet_ms=required_ms,
                remaining_ms=max(0, required_ms - quiet_age_ms),
                blocked_by_earlier=entry.source_order != self._next_order,
            )

    def next_deadline(self) -> float | None:
        with self._lock:
            entry = self._entries.get(self._next_order)
            if entry is None or entry.status != "ready":
                return None
            return float(entry.changed_at + self._stable_sec)

    def drain(self, *, force: bool = False) -> list[TTSReadyItem]:
        now = self._clock()
        ready: list[TTSReadyItem] = []
        with self._lock:
            while True:
                entry = self._entries.get(self._next_order)
                if entry is None or entry.status == "waiting":
                    break
                if entry.status == "failed":
                    del self._entries[self._next_order]
                    self._next_order += 1
                    continue
                if not force and now < entry.changed_at + self._stable_sec:
                    break
                del self._entries[self._next_order]
                self._next_order += 1
                self._released[entry.sentence_id] = (
                    int(entry.revision),
                    int(entry.source_order),
                    float(now),
                )
                ready.append(
                    TTSReadyItem(
                        sentence_id=entry.sentence_id,
                        revision=int(entry.revision),
                        source_order=int(entry.source_order),
                        target_language=str(entry.target_language or ""),
                        text=str(entry.text or ""),
                        release_reason="final_force" if force else "quiet_window",
                        source_quiet_age_ms=self._elapsed_ms(entry.changed_at, now),
                        translation_ready_age_ms=self._elapsed_ms(
                            entry.translation_ready_at or now,
                            now,
                        ),
                    )
                )
        return ready

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sentence_orders.clear()
            self._order_sentences.clear()
            self._released.clear()
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
