from pathlib import Path

from tools.text_pool_trace_report import _group_rows, _parse_text_pool_rows, _summarize


def test_parse_text_pool_rows_filters_and_parses(tmp_path: Path):
    p = tmp_path / "stream.log"
    p.write_text(
        "\n".join(
            [
                'INFO text_pool {"topic":"text_pool","event":"pool_generating_set","phase":"generating","ws_id":"x","segment_id":1,"seq":1}',
                'INFO subtitle_trace {"topic":"subtitle_state","event":"x"}',
            ]
        ),
        encoding="utf-8",
    )
    rows = _parse_text_pool_rows(p)
    assert len(rows) == 1
    assert rows[0]["event"] == "pool_generating_set"


def test_summarize_groups_by_ws_and_segment():
    rows = [
        {"topic": "text_pool", "event": "pool_generating_set", "phase": "generating", "ws_id": "a", "segment_id": 1, "seq": 1, "text_chars": 8, "reason": "partial"},
        {"topic": "text_pool", "event": "pool_solidified_append", "phase": "solidified", "ws_id": "a", "segment_id": 1, "seq": 2, "text_chars": 12, "reason": "sentence_committed"},
    ]
    grouped = _group_rows(rows)
    out = _summarize(grouped)
    assert "groups=1" in out
    assert "segment=1" in out
    assert "generating=1" in out
    assert "solidified=1" in out

