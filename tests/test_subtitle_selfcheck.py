from voxbridge.debug.subtitle_selfcheck import analyze_subtitle_events


def test_analyze_subtitle_events_reports_clean_stream():
    events = [
        {"type": "ready"},
        {"type": "partial", "text": "今"},
        {"type": "partial", "text": "今天"},
        {"type": "partial", "text": "今天开会。"},
        {"type": "partial", "text": "今天开会。请开始。"},
        {"type": "final", "text": "今天开会。请开始。"},
    ]
    result = analyze_subtitle_events(events)
    assert result.partial_count == 4
    assert result.final_count == 1
    assert result.completed_sentence_drops == 0


def test_analyze_subtitle_events_detects_completed_sentence_drop():
    events = [
        {"type": "partial", "text": "这是第一句测试内容。这里是第二句测试内容。"},
        {"type": "partial", "text": "这是第一句测试内容。"},
    ]
    result = analyze_subtitle_events(events)
    assert result.completed_sentence_drops >= 1
    assert any(ex["kind"] == "completed_drop" for ex in result.examples)


def test_analyze_subtitle_events_detects_hard_rewrite():
    events = [
        {"type": "partial", "text": "这是一个比较长的句子前半部分正在说话没有停顿"},
        {"type": "partial", "text": "完全不同的内容开始了而且前缀不一致"},
    ]
    result = analyze_subtitle_events(events)
    assert result.hard_rewrites >= 1
    assert any(ex["kind"] == "hard_rewrite" for ex in result.examples)
