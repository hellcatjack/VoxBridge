from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or ""), flags=re.UNICODE).casefold()


@dataclass(frozen=True)
class ReferenceCue:
    start_ms: int
    end_ms: int
    text: str
    best_score: float = 0.0
    best_committed_text: str = ""


@dataclass(frozen=True)
class FinalSuffixGap:
    start_ms: int
    reference_text: str
    committed_text: str
    missing_suffix: str
    score: float


@dataclass(frozen=True)
class ReferenceCoverageReport:
    reference_cue_count: int
    committed_count: int
    translated_sentence_count: int
    likely_missing_cues: List[ReferenceCue]
    suspected_final_suffix_gaps: List[FinalSuffixGap]
    duplicate_normalized_texts: dict[str, int]
    translation_missing_sentence_ids: List[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_reference_cues(path: Path, duration_sec: float | None) -> List[ReferenceCue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    duration_ms = None if duration_sec is None else max(0, int(float(duration_sec) * 1000.0))
    cues: List[ReferenceCue] = []
    for event in payload.get("events", []):
        start_ms = int(event.get("tStartMs", 0) or 0)
        if duration_ms is not None and start_ms >= duration_ms:
            continue
        end_ms = start_ms + int(event.get("dDurationMs", 0) or 0)
        if duration_ms is not None and end_ms > duration_ms:
            continue
        text = "".join(
            str(segment.get("utf8", "") or "")
            for segment in event.get("segs", [])
            if isinstance(segment, dict)
        ).replace("\n", " ").strip()
        if not normalize_text(text):
            continue
        cues.append(ReferenceCue(start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def _latest_committed(events: Iterable[dict[str, Any]]) -> List[tuple[str, str]]:
    order: List[str] = []
    latest: dict[str, str] = {}
    anonymous = 0
    for event in events:
        if str(event.get("type", "")) not in {"sentence_committed", "sentence_updated"}:
            continue
        sentence_id = str(event.get("sentence_id", "") or "").strip()
        if not sentence_id:
            anonymous += 1
            sentence_id = f"__anonymous_{anonymous}"
        if sentence_id not in latest:
            order.append(sentence_id)
        latest[sentence_id] = str(event.get("text", "") or "").strip()
    return [(sentence_id, latest[sentence_id]) for sentence_id in order if normalize_text(latest[sentence_id])]


def _alignment_score(reference: str, committed: str) -> tuple[float, SequenceMatcher[str]]:
    matcher: SequenceMatcher[str] = SequenceMatcher(None, reference, committed, autojunk=False)
    matching_chars = sum(block.size for block in matcher.get_matching_blocks())
    reference_coverage = matching_chars / float(max(1, len(reference)))
    return max(float(matcher.ratio()), float(reference_coverage)), matcher


def _best_committed_alignment(
    reference: str,
    committed: Sequence[tuple[str, str]],
) -> tuple[float, str, str, SequenceMatcher[str] | None]:
    best_score = 0.0
    best_id = ""
    best_text = ""
    best_matcher = None
    for sentence_id, text in committed:
        normalized = normalize_text(text)
        if not normalized:
            continue
        score, matcher = _alignment_score(reference, normalized)
        if score > best_score:
            best_score = score
            best_id = sentence_id
            best_text = text
            best_matcher = matcher
    return best_score, best_id, best_text, best_matcher


def _aligned_final_suffix_gap(
    cue: ReferenceCue,
    committed_text: str,
    score: float,
    matcher: SequenceMatcher[str] | None,
) -> FinalSuffixGap | None:
    reference = normalize_text(cue.text)
    committed = normalize_text(committed_text)
    if matcher is None or score < 0.5 or not reference or not committed:
        return None
    for block in reversed(matcher.get_matching_blocks()[:-1]):
        reference_end = int(block.a + block.size)
        committed_end = int(block.b + block.size)
        if committed_end != len(committed):
            continue
        suffix = reference[reference_end:]
        if 1 <= len(suffix) <= 3 and block.size >= 2:
            return FinalSuffixGap(
                start_ms=cue.start_ms,
                reference_text=cue.text,
                committed_text=committed_text,
                missing_suffix=suffix,
                score=round(score, 4),
            )
        break
    return None


def analyze_reference_coverage(
    reference_json3: Path | str,
    events: Sequence[dict[str, Any]],
    *,
    duration_sec: float | None = None,
    missing_score_threshold: float = 0.4,
    minimum_cue_chars: int = 6,
) -> ReferenceCoverageReport:
    cues = _load_reference_cues(Path(reference_json3), duration_sec)
    committed = _latest_committed(events)
    translated_ids = {
        str(event.get("sentence_id", "") or "").strip()
        for event in events
        if str(event.get("type", "")) == "sentence_translation"
        and str(event.get("translation", "") or "").strip()
    }

    likely_missing: List[ReferenceCue] = []
    suffix_gaps: List[FinalSuffixGap] = []
    for cue in cues:
        normalized = normalize_text(cue.text)
        score, _, committed_text, matcher = _best_committed_alignment(normalized, committed)
        observed = ReferenceCue(
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text=cue.text,
            best_score=round(score, 4),
            best_committed_text=committed_text,
        )
        if len(normalized) >= max(1, int(minimum_cue_chars)) and score < float(missing_score_threshold):
            likely_missing.append(observed)
        suffix_gap = _aligned_final_suffix_gap(observed, committed_text, score, matcher)
        if suffix_gap is not None:
            suffix_gaps.append(suffix_gap)

    normalized_counts = Counter(normalize_text(text) for _, text in committed)
    duplicates = {
        text: count
        for text, count in normalized_counts.items()
        if text and count > 1
    }
    translation_missing = [
        sentence_id
        for sentence_id, _ in committed
        if not sentence_id.startswith("__anonymous_") and sentence_id not in translated_ids
    ]
    return ReferenceCoverageReport(
        reference_cue_count=len(cues),
        committed_count=len(committed),
        translated_sentence_count=len(translated_ids),
        likely_missing_cues=likely_missing,
        suspected_final_suffix_gaps=suffix_gaps,
        duplicate_normalized_texts=duplicates,
        translation_missing_sentence_ids=translation_missing,
    )
