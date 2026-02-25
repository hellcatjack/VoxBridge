# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackpressureDecision:
    under_pressure: bool
    drop_oldest: bool
    suggested_batch_scale: float
    queue_sec: float
    reason: str


class QueueBackpressureController:
    """
    Queue-duration based backpressure policy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_queue_sec: float = 3.0,
        max_queue_sec: float = 5.0,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.target_queue_sec = max(0.2, float(target_queue_sec))
        self.max_queue_sec = max(self.target_queue_sec, float(max_queue_sec))

    def samples_to_sec(self, queue_samples: int) -> float:
        samples = max(0, int(queue_samples))
        return float(samples) / float(self.sample_rate)

    def evaluate(self, queue_samples: int) -> BackpressureDecision:
        q_sec = self.samples_to_sec(queue_samples)
        if q_sec >= self.max_queue_sec:
            return BackpressureDecision(
                under_pressure=True,
                drop_oldest=True,
                suggested_batch_scale=2.0,
                queue_sec=q_sec,
                reason="hard_overflow",
            )
        if q_sec >= self.target_queue_sec:
            ratio = min(2.0, max(1.0, q_sec / max(self.target_queue_sec, 1e-6)))
            return BackpressureDecision(
                under_pressure=True,
                drop_oldest=False,
                suggested_batch_scale=ratio,
                queue_sec=q_sec,
                reason="soft_pressure",
            )
        return BackpressureDecision(
            under_pressure=False,
            drop_oldest=False,
            suggested_batch_scale=1.0,
            queue_sec=q_sec,
            reason="normal",
        )

