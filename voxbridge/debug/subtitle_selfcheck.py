from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from voxbridge.cli.demo_streaming_ws import _split_sentences_and_tail


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class SubtitleSelfcheckResult:
    partial_count: int
    final_count: int
    max_chars: int
    max_completed_sentences: int
    completed_sentence_drops: int
    hard_rewrites: int
    examples: List[Dict[str, Any]]


def analyze_subtitle_events(events: Iterable[Dict[str, Any]]) -> SubtitleSelfcheckResult:
    prev_text = ""
    prev_completed = 0
    partial_count = 0
    final_count = 0
    max_chars = 0
    max_completed = 0
    drops = 0
    rewrites = 0
    examples: List[Dict[str, Any]] = []

    for idx, msg in enumerate(events):
        msg_type = str(msg.get("type", "")).lower()
        if msg_type not in {"partial", "final"}:
            continue

        text = str(msg.get("text", "") or "").strip()
        completed, tail = _split_sentences_and_tail(text)
        completed_count = len(completed)
        chars = len(text)
        max_chars = max(max_chars, chars)
        max_completed = max(max_completed, completed_count)

        if msg_type == "partial":
            partial_count += 1
        elif msg_type == "final":
            final_count += 1

        if prev_completed > 0 and completed_count < prev_completed:
            drops += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "kind": "completed_drop",
                        "index": idx,
                        "from": prev_completed,
                        "to": completed_count,
                        "chars": chars,
                        "text": text[:160],
                    }
                )

        if prev_text:
            lcp = _lcp_len(prev_text, text)
            threshold = max(4, int(len(prev_text) * 0.25))
            if len(prev_text) >= 20 and lcp < threshold:
                rewrites += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "kind": "hard_rewrite",
                            "index": idx,
                            "lcp": lcp,
                            "prev_chars": len(prev_text),
                            "chars": chars,
                            "tail_chars": len(tail),
                            "text": text[:160],
                        }
                    )

        prev_text = text
        prev_completed = completed_count

    return SubtitleSelfcheckResult(
        partial_count=partial_count,
        final_count=final_count,
        max_chars=max_chars,
        max_completed_sentences=max_completed,
        completed_sentence_drops=drops,
        hard_rewrites=rewrites,
        examples=examples,
    )


def summarize_result(result: SubtitleSelfcheckResult) -> str:
    lines = [
        f"partials={result.partial_count}",
        f"finals={result.final_count}",
        f"max_chars={result.max_chars}",
        f"max_completed_sentences={result.max_completed_sentences}",
        f"completed_sentence_drops={result.completed_sentence_drops}",
        f"hard_rewrites={result.hard_rewrites}",
    ]
    if result.examples:
        lines.append("examples:")
        for ex in result.examples:
            kind = ex.get("kind", "event")
            lines.append(f"  - {kind}: {ex}")
    return "\n".join(lines)

