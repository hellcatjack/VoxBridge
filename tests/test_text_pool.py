from voxbridge.streaming.text_pool import (
    SolidifiedSentence,
    TextPool,
    dedup_segment_join,
    trim_prefix_overlap,
)


def test_text_pool_tracks_generating_and_solidified_separately():
    pool = TextPool()
    pool.set_generating("正在生成")
    assert pool.generating_text == "正在生成"
    assert pool.solidified == []

    pool.append_solidified(SolidifiedSentence(sentence_id="s1", text="第一句。", segment_id=1))
    assert len(pool.solidified) == 1
    assert pool.solidified[0].text == "第一句。"
    assert pool.generating_text == "正在生成"

    pool.reset_generating()
    assert pool.generating_text == ""


def test_text_pool_update_sentence_and_translation():
    pool = TextPool()
    pool.append_solidified(SolidifiedSentence(sentence_id="s1", text="第一句。", segment_id=1))
    assert pool.update_solidified("s1", "第一句更新。") is True
    assert pool.update_translation("s1", "First sentence.") is True
    assert pool.solidified[0].text == "第一句更新。"
    assert pool.solidified[0].translation == "First sentence."


def test_dedup_removes_overlap_between_segments():
    prev = "第一遍测试翻译，第二遍测试翻译。"
    nxt = "第二遍测试翻译。第三遍测试翻译。"
    merged = dedup_segment_join(prev, nxt)
    assert merged == "第一遍测试翻译，第二遍测试翻译。第三遍测试翻译。"


def test_trim_prefix_overlap_trims_small_boundary_overlap():
    ref = "女人对蛇说：那棵树上的果子"
    candidate = "树上的果子，神曾经说过。"
    trimmed, overlap = trim_prefix_overlap(ref, candidate, min_overlap=3, max_overlap=8)
    assert overlap == 5
    assert trimmed == "，神曾经说过。"


def test_trim_prefix_overlap_skips_when_overlap_exceeds_cap():
    ref = "第十一次测试翻译。"
    candidate = "第十一次测试翻译。第十二次测试翻译。"
    trimmed, overlap = trim_prefix_overlap(ref, candidate, min_overlap=4, max_overlap=6)
    assert overlap == 0
    assert trimmed == candidate


def test_trim_prefix_overlap_ignores_trailing_punctuation_in_reference():
    ref = "女人对蛇说：那一棵树上的果。"
    candidate = "树上的果子，神曾经说过。"
    trimmed, overlap = trim_prefix_overlap(ref, candidate, min_overlap=3, max_overlap=8)
    assert overlap == 4
    assert trimmed == "子，神曾经说过。"
