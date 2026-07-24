# coding=utf-8

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_SENTENCE_PUNCTUATION = re.compile(r"[。！？!?；;，,:：\r\n]")
_PERIOD_AT_BOUNDARY = re.compile(r"\.(?=[\"'”’\)\]）】》]*(?:\s|$))")
_DOTTED_INITIALISM = re.compile(r"(?:[A-Z]\.){2,}$")


def _contains_sentence_punctuation(value: str) -> bool:
    if _SENTENCE_PUNCTUATION.search(value):
        return True
    for match in _PERIOD_AT_BOUNDARY.finditer(value):
        token_start = match.start()
        while token_start > 0 and not value[token_start - 1].isspace():
            token_start -= 1
        if _DOTTED_INITIALISM.fullmatch(value[token_start : match.end()]):
            continue
        return True
    return False


def _terms(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} terms must be a list of strings")

    cleaned: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise ValueError(f"{field} term {index} must be a string")
        if _contains_sentence_punctuation(raw):
            raise ValueError(f"{field} term {index} contains sentence punctuation")
        term = " ".join(raw.split())
        if not term:
            raise ValueError(f"{field} term {index} must not be empty")
        cleaned.append(term)
    return tuple(cleaned)


def normalize_session_context_terms(
    value: Any,
    *,
    max_terms: int,
    max_chars: int,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("ASR context terms must be a list of strings")

    term_limit = max(0, int(max_terms))
    char_limit = max(0, int(max_chars))
    selected: list[str] = []
    seen: set[str] = set()
    context_chars = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise ValueError(f"ASR context term {index} must be a string")
        term = raw.strip()
        if not term:
            continue
        if any(char.isspace() for char in term):
            raise ValueError(f"ASR context term {index} contains internal whitespace")
        if _contains_sentence_punctuation(term):
            raise ValueError(f"ASR context term {index} contains sentence punctuation")
        key = term.casefold()
        if key in seen:
            continue
        if len(selected) >= term_limit:
            raise ValueError(f"ASR context accepts at most {term_limit} terms")
        added_chars = len(term) + (1 if selected else 0)
        if context_chars + added_chars > char_limit:
            raise ValueError(f"ASR context accepts at most {char_limit} characters")
        selected.append(term)
        seen.add(key)
        context_chars += added_chars
    return tuple(selected)


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


@dataclass(frozen=True)
class ContextSegment:
    start_sec: float
    end_sec: float
    terms: tuple[str, ...]
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextSchedule:
    language: str
    global_terms: tuple[str, ...]
    segments: tuple[ContextSegment, ...]
    version: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ContextSchedule":
        if not isinstance(raw, Mapping):
            raise ValueError("context schedule must be a JSON object")
        version = raw.get("version")
        if version != 1:
            raise ValueError(f"unsupported context schedule version: {version!r}")
        language = str(raw.get("language") or "").strip()
        if not language:
            raise ValueError("context schedule language must not be empty")

        segment_values = raw.get("segments", [])
        if not isinstance(segment_values, Sequence) or isinstance(segment_values, (str, bytes)):
            raise ValueError("context schedule segments must be a list")

        segments: list[ContextSegment] = []
        previous_end = 0.0
        for index, item in enumerate(segment_values):
            if not isinstance(item, Mapping):
                raise ValueError(f"segment {index} must be a JSON object")
            start = _finite_number(item.get("start_sec"), f"segment {index} start_sec")
            end = _finite_number(item.get("end_sec"), f"segment {index} end_sec")
            if start < 0 or end <= start:
                raise ValueError(f"segment {index} must have 0 <= start_sec < end_sec")
            if segments and start < previous_end:
                raise ValueError(f"segment {index} overlaps the previous segment")
            segments.append(
                ContextSegment(
                    start_sec=start,
                    end_sec=end,
                    terms=_terms(item.get("terms", []), f"segment {index}"),
                    anchors=_terms(item.get("anchors", []), f"segment {index} anchors"),
                )
            )
            previous_end = end

        return cls(
            version=1,
            language=language,
            global_terms=_terms(raw.get("global_terms", []), "global"),
            segments=tuple(segments),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ContextSchedule":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load context schedule {source}: {exc}") from exc
        return cls.from_mapping(raw)

    def terms_at(
        self,
        elapsed_sec: float,
        *,
        language: str | None = None,
        lookaround_sec: float = 0.0,
        max_terms: int = 24,
        max_chars: int = 160,
    ) -> tuple[str, ...]:
        if not language or language.strip().casefold() != self.language.casefold():
            return ()
        elapsed = max(0.0, _finite_number(elapsed_sec, "elapsed_sec"))
        lookaround = max(0.0, _finite_number(lookaround_sec, "lookaround_sec"))
        term_limit = max(0, int(max_terms))
        char_limit = max(0, int(max_chars))
        if term_limit == 0 or char_limit == 0:
            return ()

        candidates = list(self.global_terms)
        for segment in self.segments:
            if segment.start_sec - lookaround <= elapsed <= segment.end_sec + lookaround:
                candidates.extend(segment.terms)

        selected: list[str] = []
        seen: set[str] = set()
        used_chars = 0
        for term in candidates:
            key = term.casefold()
            if key in seen:
                continue
            added_chars = len(term) + (1 if selected else 0)
            if len(selected) >= term_limit or used_chars + added_chars > char_limit:
                break
            selected.append(term)
            seen.add(key)
            used_chars += added_chars
        return tuple(selected)

    def context_at(
        self,
        elapsed_sec: float,
        *,
        language: str | None = None,
        lookaround_sec: float = 0.0,
        max_terms: int = 24,
        max_chars: int = 160,
    ) -> str:
        return " ".join(
            self.terms_at(
                elapsed_sec,
                language=language,
                lookaround_sec=lookaround_sec,
                max_terms=max_terms,
                max_chars=max_chars,
            )
        )
