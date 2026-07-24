import json

import pytest

from voxbridge.streaming.context_schedule import (
    ContextSchedule,
    normalize_session_context_terms,
)


def test_context_schedule_selects_current_and_nearby_terms_without_sentences():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "Chinese",
            "global_terms": ["出埃及记"],
            "segments": [
                {"start_sec": 0, "end_sec": 120, "terms": ["暗兰", "约基别"]},
                {"start_sec": 120, "end_sec": 240, "terms": ["利未支派"]},
            ],
        }
    )

    assert schedule.context_at(
        125,
        language="Chinese",
        lookaround_sec=10,
        max_terms=3,
        max_chars=40,
    ) == (
        "出埃及记 暗兰 约基别"
    )


def test_context_schedule_truncates_only_at_term_boundaries():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "Chinese",
            "global_terms": ["出埃及记", "亚伦", "摩西"],
            "segments": [],
        }
    )

    assert schedule.context_at(
        0,
        language="Chinese",
        max_terms=10,
        max_chars=7,
    ) == "出埃及记 亚伦"


def test_context_schedule_deduplicates_terms_without_reordering():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "Chinese",
            "global_terms": ["耶和华", "摩西"],
            "segments": [
                {
                    "start_sec": 0,
                    "end_sec": 60,
                    "terms": ["摩西", "亚伦", "耶和华"],
                }
            ],
        }
    )

    assert schedule.context_at(
        30,
        language="Chinese",
        max_terms=10,
        max_chars=80,
    ) == "耶和华 摩西 亚伦"


def test_context_schedule_returns_empty_for_a_different_language():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "Chinese",
            "global_terms": ["出埃及记"],
            "segments": [],
        }
    )

    assert schedule.context_at(0, language="English") == ""


def test_context_schedule_returns_empty_when_session_language_is_unknown():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "Chinese",
            "global_terms": ["出埃及记"],
            "segments": [],
        }
    )

    assert schedule.context_at(0, language=None) == ""


@pytest.mark.parametrize(
    "term",
    [
        "神对摩西说。",
        "亚伦，摩西",
        "完整句子！",
        "两行\n术语",
        "This is a complete sentence.",
        "First term: second term",
        'He said "Go home."',
        "Go home.)",
    ],
)
def test_context_schedule_rejects_sentence_like_terms(term):
    with pytest.raises(ValueError, match="term"):
        ContextSchedule.from_mapping(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": [term],
                "segments": [],
            }
        )


@pytest.mark.parametrize("term", ["U.S.", "U.S. policy"])
def test_context_schedule_allows_dotted_initialisms(term):
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "English",
            "global_terms": [term],
            "segments": [],
        }
    )

    assert schedule.context_at(0, language="English") == term


def test_context_schedule_rejects_overlapping_segments():
    with pytest.raises(ValueError, match="overlap"):
        ContextSchedule.from_mapping(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": [],
                "segments": [
                    {"start_sec": 0, "end_sec": 120, "terms": ["暗兰"]},
                    {"start_sec": 100, "end_sec": 200, "terms": ["约基别"]},
                ],
            }
        )


def test_context_schedule_loads_json_from_path(tmp_path):
    path = tmp_path / "context.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "Chinese",
                "global_terms": ["出埃及记"],
                "segments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schedule = ContextSchedule.from_path(path)

    assert schedule.context_at(0, language="Chinese") == "出埃及记"


def test_context_schedule_exposes_selected_terms_without_losing_boundaries():
    schedule = ContextSchedule.from_mapping(
        {
            "version": 1,
            "language": "English",
            "global_terms": ["Second Kings", "Elisha"],
            "segments": [],
        }
    )

    assert schedule.terms_at(0, language="English") == ("Second Kings", "Elisha")
    assert schedule.context_at(0, language="English") == "Second Kings Elisha"


def test_session_context_terms_trim_skip_empty_and_dedupe_case_insensitively():
    assert normalize_session_context_terms(
        ["  摩西  ", "Moses", "", "  ", "mOsEs", "亚伦"],
        max_terms=24,
        max_chars=160,
    ) == ("摩西", "Moses", "亚伦")


@pytest.mark.parametrize(
    ("raw", "message", "secret"),
    [
        ("Moses", "list of strings", "Moses"),
        (["Moses", 7], "term 1 must be a string", "Moses"),
        (["Second Kings"], "internal whitespace", "Second Kings"),
        (["SECRET_TERM."], "sentence punctuation", "SECRET_TERM"),
        (["摩西，亚伦"], "sentence punctuation", "摩西"),
    ],
)
def test_session_context_terms_reject_invalid_payloads_without_echoing_values(
    raw,
    message,
    secret,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        normalize_session_context_terms(raw, max_terms=24, max_chars=160)

    assert secret not in str(exc_info.value)


def test_session_context_terms_enforce_server_limits_after_deduplication():
    assert normalize_session_context_terms(
        ["Moses", "moses"],
        max_terms=1,
        max_chars=20,
    ) == ("Moses",)

    with pytest.raises(ValueError, match="at most 2 terms"):
        normalize_session_context_terms(
            ["Moses", "Aaron", "Elisha"],
            max_terms=2,
            max_chars=160,
        )

    with pytest.raises(ValueError, match="at most 8 characters"):
        normalize_session_context_terms(
            ["Moses", "Aaron"],
            max_terms=24,
            max_chars=8,
        )


def test_session_context_terms_reject_limits_before_retaining_more_input():
    with pytest.raises(ValueError, match="at most 1 terms"):
        normalize_session_context_terms(
            ["Moses", "Aaron", 7],
            max_terms=1,
            max_chars=160,
        )

    with pytest.raises(ValueError, match="at most 5 characters"):
        normalize_session_context_terms(
            ["Moses", "Aaron", 7],
            max_terms=24,
            max_chars=5,
        )


def test_session_context_terms_allow_dotted_uppercase_initialisms():
    assert normalize_session_context_terms(
        ["U.S."],
        max_terms=24,
        max_chars=160,
    ) == ("U.S.",)
