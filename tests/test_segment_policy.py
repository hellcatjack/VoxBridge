from voxbridge.streaming.segment_policy import SegmentPolicy


def test_segment_policy_triggers_hard_cut_at_limit():
    p = SegmentPolicy(vad_silence_ms=800, hard_cut_ms=30000, min_segment_ms=4000, min_active_ms=1200)
    d = p.evaluate(
        silence_ms=100.0,
        segment_age_ms=30000.0,
        segment_active_ms=12000.0,
        has_pending_text=True,
        vad_candidate=False,
        vad_force=False,
    )
    assert d.should_cut is True
    assert d.reason == "hard_cut"
    assert d.force_finalize is True


def test_segment_policy_triggers_vad_cut_when_ready():
    p = SegmentPolicy(vad_silence_ms=800, hard_cut_ms=30000, min_segment_ms=4000, min_active_ms=1200)
    d = p.evaluate(
        silence_ms=900.0,
        segment_age_ms=8000.0,
        segment_active_ms=3000.0,
        has_pending_text=True,
        vad_candidate=True,
        vad_force=False,
    )
    assert d.should_cut is True
    assert d.reason == "vad_silence"


def test_segment_policy_does_not_cut_when_no_pending_text():
    p = SegmentPolicy(vad_silence_ms=800, hard_cut_ms=30000, min_segment_ms=4000, min_active_ms=1200)
    d = p.evaluate(
        silence_ms=1200.0,
        segment_age_ms=12000.0,
        segment_active_ms=6000.0,
        has_pending_text=False,
        vad_candidate=True,
        vad_force=True,
    )
    assert d.should_cut is False
    assert d.reason == "none"

