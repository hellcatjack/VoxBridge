# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentCutDecision:
    should_cut: bool
    reason: str
    force_finalize: bool
    silence_ms: float
    segment_age_ms: float


class SegmentPolicy:
    """
    Decide when to finalize one streaming segment and rotate to a new state.
    """

    def __init__(
        self,
        vad_silence_ms: float = 800.0,
        hard_cut_ms: float = 30000.0,
        min_segment_ms: float = 4000.0,
        min_active_ms: float = 1200.0,
    ) -> None:
        self.vad_silence_ms = max(80.0, float(vad_silence_ms))
        self.hard_cut_ms = max(1000.0, float(hard_cut_ms))
        self.min_segment_ms = max(200.0, float(min_segment_ms))
        self.min_active_ms = max(0.0, float(min_active_ms))

    def evaluate(
        self,
        *,
        silence_ms: float,
        segment_age_ms: float,
        segment_active_ms: float,
        has_pending_text: bool,
        vad_candidate: bool,
        vad_force: bool,
    ) -> SegmentCutDecision:
        silence = max(0.0, float(silence_ms))
        age = max(0.0, float(segment_age_ms))
        active = max(0.0, float(segment_active_ms))
        has_text = bool(has_pending_text)

        if age >= self.hard_cut_ms:
            return SegmentCutDecision(
                should_cut=has_text,
                reason="hard_cut",
                force_finalize=True,
                silence_ms=silence,
                segment_age_ms=age,
            )

        vad_ready = bool(vad_candidate) and silence >= self.vad_silence_ms
        enough_age = age >= self.min_segment_ms
        enough_active = active >= self.min_active_ms
        if vad_ready and enough_age and enough_active and has_text:
            return SegmentCutDecision(
                should_cut=True,
                reason="vad_silence",
                force_finalize=bool(vad_force),
                silence_ms=silence,
                segment_age_ms=age,
            )

        return SegmentCutDecision(
            should_cut=False,
            reason="none",
            force_finalize=False,
            silence_ms=silence,
            segment_age_ms=age,
        )

