# coding=utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


class AudioPreRollBuffer:
    """Retain only audio that has not yet been sent to the ASR backend."""

    def __init__(self, *, sample_rate: int, duration_sec: float) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.capacity_samples = max(0, int(round(self.sample_rate * max(0.0, float(duration_sec)))))
        self._samples = np.empty(0, dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        return int(self._samples.size)

    def append(self, audio: np.ndarray) -> None:
        if self.capacity_samples <= 0:
            return
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size <= 0:
            return
        if samples.size >= self.capacity_samples:
            self._samples = samples[-self.capacity_samples :].copy()
            return
        combined = np.concatenate((self._samples, samples))
        self._samples = combined[-self.capacity_samples :].copy()

    def prepend_to(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        replayed_samples = int(self._samples.size)
        if replayed_samples <= 0:
            return samples, 0
        combined = np.concatenate((self._samples, samples))
        self._samples = np.empty(0, dtype=np.float32)
        return combined, replayed_samples

    def clear(self) -> None:
        self._samples = np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class ShadowVadObservation:
    available: bool
    frames: int = 0
    pending_samples: int = 0
    last_probability: Optional[float] = None
    mean_probability: Optional[float] = None
    max_probability: Optional[float] = None
    is_speech: bool = False
    state_changed: bool = False
    error: str = ""


class SileroShadowObserver:
    """Collect stateful Silero probabilities without making control decisions."""

    def __init__(
        self,
        *,
        runner: Callable[[np.ndarray], float],
        sample_rate: int = 16_000,
        frame_samples: int = 512,
        threshold: float = 0.5,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_samples = max(1, int(frame_samples))
        self.threshold = min(1.0, max(0.0, float(threshold)))
        self._runner: Optional[Callable[[np.ndarray], float]] = runner
        self._pending = np.empty(0, dtype=np.float32)
        self._is_speech = False
        self._error = ""

    @property
    def available(self) -> bool:
        return self._runner is not None

    @property
    def error(self) -> str:
        return self._error

    def feed(self, audio: np.ndarray) -> ShadowVadObservation:
        if self._runner is None:
            return ShadowVadObservation(
                available=False,
                pending_samples=int(self._pending.size),
                is_speech=bool(self._is_speech),
                error=self._error,
            )

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size > 0:
            self._pending = np.concatenate((self._pending, samples))

        probabilities: list[float] = []
        changed = False
        try:
            while self._pending.size >= self.frame_samples:
                frame = np.ascontiguousarray(self._pending[: self.frame_samples], dtype=np.float32)
                self._pending = self._pending[self.frame_samples :]
                probability = min(1.0, max(0.0, float(self._runner(frame))))
                probabilities.append(probability)
                next_is_speech = probability >= self.threshold
                if next_is_speech != self._is_speech:
                    changed = True
                self._is_speech = next_is_speech
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._runner = None
            return ShadowVadObservation(
                available=False,
                frames=len(probabilities),
                pending_samples=int(self._pending.size),
                is_speech=bool(self._is_speech),
                state_changed=changed,
                error=self._error,
            )

        if not probabilities:
            return ShadowVadObservation(
                available=True,
                pending_samples=int(self._pending.size),
                is_speech=bool(self._is_speech),
            )
        return ShadowVadObservation(
            available=True,
            frames=len(probabilities),
            pending_samples=int(self._pending.size),
            last_probability=float(probabilities[-1]),
            mean_probability=float(sum(probabilities) / len(probabilities)),
            max_probability=float(max(probabilities)),
            is_speech=bool(self._is_speech),
            state_changed=bool(changed),
        )


def create_silero_onnx_observer(
    *,
    threshold: float = 0.5,
    load_model: Optional[Callable[..., Any]] = None,
) -> SileroShadowObserver:
    if load_model is None:
        from silero_vad import load_silero_vad

        load_model = load_silero_vad

    import torch

    model = load_model(onnx=True, opset_version=16)
    reset_states = getattr(model, "reset_states", None)
    if callable(reset_states):
        reset_states()

    def _run(frame: np.ndarray) -> float:
        tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        value = model(tensor, 16_000)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    return SileroShadowObserver(
        runner=_run,
        sample_rate=16_000,
        frame_samples=512,
        threshold=threshold,
    )
