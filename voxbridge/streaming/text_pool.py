# coding=utf-8
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


def dedup_segment_join(prev_text: str, new_text: str, min_overlap: int = 2) -> str:
    prev = str(prev_text or "")
    nxt = str(new_text or "")
    if not prev:
        return nxt
    if not nxt:
        return prev
    if nxt in prev:
        return prev

    max_k = min(len(prev), len(nxt))
    best = 0
    for k in range(max_k, max(0, int(min_overlap) - 1), -1):
        if prev[-k:] == nxt[:k]:
            best = k
            break
    if best <= 0:
        return f"{prev}{nxt}"
    return f"{prev}{nxt[best:]}"


def trim_prefix_overlap(
    reference_text: str,
    candidate_text: str,
    *,
    min_overlap: int = 2,
    max_overlap: int = 16,
) -> tuple[str, int]:
    ref = str(reference_text or "")
    cand = str(candidate_text or "")
    if not ref or not cand:
        return cand, 0

    ref = re.sub(r"[\\s。！？!?…，,、；;：:]+$", "", ref).strip()
    if not ref:
        return cand, 0

    min_k = max(1, int(min_overlap))
    cap = max(min_k, int(max_overlap))
    full_max = min(len(ref), len(cand))
    best = 0
    for k in range(full_max, min_k - 1, -1):
        if ref[-k:] == cand[:k]:
            best = k
            break
    if best <= 0:
        return cand, 0
    if best > cap:
        return cand, 0
    return cand[best:], best


@dataclass
class SolidifiedSentence:
    sentence_id: str
    text: str
    translation: str = ""
    ts_ms: int = 0
    segment_id: int = 0


class TextPool:
    """
    Two-pool subtitle state:
    - generating_text: mutable in-progress text for current segment
    - solidified: committed sentence timeline
    """

    def __init__(self) -> None:
        self.generating_text = ""
        self.solidified: List[SolidifiedSentence] = []
        self._index: Dict[str, int] = {}

    def set_generating(self, text: str) -> None:
        self.generating_text = str(text or "").strip()

    def reset_generating(self) -> None:
        self.generating_text = ""

    def append_solidified(self, sentence: SolidifiedSentence) -> None:
        sid = str(sentence.sentence_id or "").strip()
        if not sid:
            raise ValueError("sentence_id is required")
        if sid in self._index:
            self.update_solidified(sid, sentence.text)
            return
        self._index[sid] = len(self.solidified)
        self.solidified.append(sentence)

    def update_solidified(self, sentence_id: str, text: str) -> bool:
        sid = str(sentence_id or "").strip()
        idx = self._index.get(sid)
        if idx is None:
            return False
        cur = self.solidified[idx]
        nxt = str(text or "").strip()
        if not nxt or nxt == cur.text:
            return False
        cur.text = nxt
        return True

    def update_translation(self, sentence_id: str, translation: str) -> bool:
        sid = str(sentence_id or "").strip()
        idx = self._index.get(sid)
        if idx is None:
            return False
        cur = self.solidified[idx]
        nxt = str(translation or "").strip()
        if nxt == cur.translation:
            return False
        cur.translation = nxt
        return True

    def snapshot(self) -> Dict[str, object]:
        return {
            "generating_text": self.generating_text,
            "solidified_count": len(self.solidified),
            "solidified_ids": [x.sentence_id for x in self.solidified],
            "solidified_text": [x.text for x in self.solidified],
        }
