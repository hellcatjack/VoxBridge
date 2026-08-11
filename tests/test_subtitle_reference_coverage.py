import json

from voxbridge.debug.subtitle_reference_coverage import analyze_reference_coverage


def _reference(tmp_path, cues):
    events = []
    for index, text in enumerate(cues):
        events.append(
            {
                "tStartMs": index * 3000,
                "dDurationMs": 3000,
                "segs": [{"utf8": text}],
            }
        )
    path = tmp_path / "reference.json3"
    path.write_text(json.dumps({"events": events}, ensure_ascii=False), encoding="utf-8")
    return path


def test_reference_coverage_flags_whole_missing_cue_without_requiring_exact_text(tmp_path):
    reference = _reference(
        tmp_path,
        [
            "有人因为工作和家庭已经喘不过气来了。",
            "但又听到别人不经意的一句批评。",
            "心里面的怨气慢慢开始起来。",
            "后来他们终于理解了彼此。",
        ],
    )
    events = [
        {"type": "sentence_committed", "sentence_id": "s1", "text": "有人因工作和家庭喘不过气了。"},
        {"type": "sentence_translation", "sentence_id": "s1", "translation": "First."},
        {"type": "sentence_committed", "sentence_id": "s2", "text": "后来他们逐渐理解了彼此。"},
        {"type": "sentence_translation", "sentence_id": "s2", "translation": "Later."},
    ]

    report = analyze_reference_coverage(reference, events, duration_sec=20)

    assert [item.text for item in report.likely_missing_cues] == [
        "但又听到别人不经意的一句批评。",
        "心里面的怨气慢慢开始起来。",
    ]
    assert report.translation_missing_sentence_ids == []


def test_reference_coverage_reports_aligned_sentence_final_suffix(tmp_path):
    reference = _reference(tmp_path, ["有人默默地最早来预备场地。"])
    events = [
        {
            "type": "sentence_committed",
            "sentence_id": "s1",
            "text": "有人搬椅子，有人默默地最早来预备场。",
        },
        {"type": "sentence_translation", "sentence_id": "s1", "translation": "Prepared."},
    ]

    report = analyze_reference_coverage(reference, events, duration_sec=10)

    assert len(report.suspected_final_suffix_gaps) == 1
    assert report.suspected_final_suffix_gaps[0].missing_suffix == "地"


def test_reference_coverage_tracks_duplicates_and_translation_id_gaps(tmp_path):
    reference = _reference(tmp_path, ["第一句话。", "第二句话。"])
    events = [
        {"type": "sentence_committed", "sentence_id": "s1", "text": "第一句话。"},
        {"type": "sentence_committed", "sentence_id": "s2", "text": "第一句话！"},
        {"type": "sentence_committed", "sentence_id": "s3", "text": "第二句话。"},
        {"type": "sentence_translation", "sentence_id": "s1", "translation": "One."},
        {"type": "sentence_translation", "sentence_id": "s3", "translation": "Two."},
    ]

    report = analyze_reference_coverage(reference, events, duration_sec=10)

    assert report.duplicate_normalized_texts == {"第一句话": 2}
    assert report.translation_missing_sentence_ids == ["s2"]


def test_reference_coverage_excludes_cue_cut_off_by_replay_duration(tmp_path):
    reference = _reference(tmp_path, ["完整播放的句子。", "只播放开头的边界句子。"])
    events = [
        {"type": "sentence_committed", "sentence_id": "s1", "text": "完整播放的句子。"},
        {"type": "sentence_translation", "sentence_id": "s1", "translation": "Complete."},
    ]

    report = analyze_reference_coverage(reference, events, duration_sec=3.5)

    assert report.reference_cue_count == 1
    assert report.likely_missing_cues == []
