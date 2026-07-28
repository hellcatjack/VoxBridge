import pytest

from voxbridge.tts.jobs import (
    RevisionStableTTSBuffer,
    TTSJobNotFound,
    TTSJobRegistry,
    TTSQueueFull,
)


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def create_job(registry: TTSJobRegistry, **overrides):
    values = {
        "owner_key": "owner-a",
        "client_id": "client-a-12345678",
        "sentence_id": "s1",
        "revision": 1,
        "source_order": 0,
        "target_language": "English",
        "text": "Stable translation.",
    }
    values.update(overrides)
    return registry.create(**values)


def test_registry_enforces_owner_and_acknowledgement():
    clock = FakeClock(100.0)
    registry = TTSJobRegistry(ttl_sec=30, max_client_jobs=4, clock=clock)
    job = create_job(registry)

    assert registry.get(job.job_id, "owner-a").text == "Stable translation."
    with pytest.raises(TTSJobNotFound):
        registry.get(job.job_id, "owner-b")

    assert registry.acknowledge(job.job_id, "owner-a") is True
    with pytest.raises(TTSJobNotFound):
        registry.get(job.job_id, "owner-a")


def test_registry_never_evicts_unread_job_when_full():
    registry = TTSJobRegistry(ttl_sec=30, max_client_jobs=1)
    first = create_job(registry)

    with pytest.raises(TTSQueueFull):
        create_job(registry, sentence_id="s2", source_order=1)

    assert registry.get(first.job_id, "owner-a").job_id == first.job_id


def test_registry_expires_jobs_without_exposing_them():
    clock = FakeClock(100.0)
    registry = TTSJobRegistry(ttl_sec=30, max_client_jobs=4, clock=clock)
    job = create_job(registry)

    clock.value = 130.01

    with pytest.raises(TTSJobNotFound):
        registry.get(job.job_id, "owner-a")
    assert registry.prune() == 0


def test_registry_cancels_only_matching_owner_and_client():
    registry = TTSJobRegistry(ttl_sec=30, max_client_jobs=4)
    first = create_job(registry, sentence_id="s1")
    second = create_job(registry, sentence_id="s2", client_id="client-b-12345678")
    third = create_job(
        registry,
        owner_key="owner-b",
        client_id="client-a-12345678",
        sentence_id="s3",
    )

    assert registry.cancel_client("owner-a", "client-a-12345678") == 1
    with pytest.raises(TTSJobNotFound):
        registry.get(first.job_id, "owner-a")
    assert registry.get(second.job_id, "owner-a").job_id == second.job_id
    assert registry.get(third.job_id, "owner-b").job_id == third.job_id


def test_registry_caches_audio_without_mutating_text_snapshot():
    registry = TTSJobRegistry(ttl_sec=30, max_client_jobs=4)
    job = create_job(registry)

    cached = registry.cache_audio(job.job_id, "owner-a", b"RIFF-audio")

    assert cached.audio_bytes == b"RIFF-audio"
    assert cached.text == "Stable translation."
    assert job.audio_bytes is None


def test_stability_buffer_withholds_ready_revision_until_quiet_window():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)

    result = buffer.register("s1", revision=1, source_order=0)
    assert result.accepted is True
    assert buffer.mark_ready("s1", 1, "first", "English") is True
    assert buffer.drain() == []
    assert buffer.next_deadline() == pytest.approx(103.0)

    clock.advance(2.999)
    assert buffer.drain() == []
    clock.advance(0.001)
    ready = buffer.drain()

    assert [(item.sentence_id, item.revision, item.text) for item in ready] == [
        ("s1", 1, "first")
    ]
    assert ready[0].release_reason == "quiet_window"
    assert ready[0].source_quiet_age_ms == 3000
    assert ready[0].translation_ready_age_ms == 3000


def test_revision_update_discards_old_translation_and_restarts_window():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    assert buffer.mark_ready("s1", 1, "old", "English") is True

    clock.advance(2.9)
    update = buffer.register("s1", 2, 0)

    assert update.reset is True
    assert update.previous_revision == 1
    assert update.previous_ready is True
    assert update.previous_quiet_age_ms == 2900
    assert buffer.mark_ready("s1", 1, "stale", "English") is False
    assert buffer.mark_ready("s1", 2, "new", "English") is True
    clock.advance(2.9)
    assert buffer.drain() == []
    clock.advance(0.1)
    assert [item.text for item in buffer.drain()] == ["new"]


def test_translation_finishing_after_source_deadline_releases_immediately():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)

    clock.advance(4.0)
    assert buffer.mark_ready("s1", 1, "late translation", "English") is True

    ready = buffer.drain()
    assert [item.text for item in ready] == ["late translation"]
    assert ready[0].source_quiet_age_ms == 4000
    assert ready[0].translation_ready_age_ms == 0


def test_release_age_preserves_zero_monotonic_timestamp():
    clock = FakeClock(0.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "translated", "English")

    clock.advance(3.0)
    ready = buffer.drain()

    assert ready[0].translation_ready_age_ms == 3000


def test_stability_buffer_preserves_order_and_skips_failed_head():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    assert buffer.mark_ready("s2", 1, "second", "English") is True

    clock.advance(3.0)
    assert buffer.drain() == []
    assert buffer.mark_failed("s1", 1) is True

    ready = buffer.drain()
    assert [item.sentence_id for item in ready] == ["s2"]


def test_wait_state_reports_quiet_time_and_order_blocking():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    buffer.mark_ready("s2", 1, "second", "English")

    clock.advance(1.25)
    wait = buffer.wait_state("s2")

    assert wait is not None
    assert wait.quiet_age_ms == 1250
    assert wait.required_quiet_ms == 3000
    assert wait.remaining_ms == 1750
    assert wait.blocked_by_earlier is True


def test_force_drain_releases_only_current_ready_revisions():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=60.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    buffer.register("s2", 2, 1)
    assert buffer.mark_ready("s1", 1, "first", "English") is True
    assert buffer.mark_ready("s2", 1, "stale", "English") is False
    assert buffer.mark_ready("s2", 2, "second", "English") is True

    ready = buffer.drain(force=True)

    assert [(item.revision, item.text) for item in ready] == [(1, "first"), (2, "second")]
    assert {item.release_reason for item in ready} == {"final_force"}


def test_revision_after_release_is_reported_and_never_emitted_twice():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=0.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "spoken", "English")
    assert len(buffer.drain()) == 1

    clock.advance(1.25)
    late = buffer.register("s1", 2, 0)

    assert late.accepted is False
    assert late.late_after_release is True
    assert late.released_revision == 1
    assert late.elapsed_since_release_ms == 1250
    assert buffer.mark_ready("s1", 2, "changed", "English") is False
    assert buffer.drain() == []


def test_stability_buffer_rejects_identity_changes():
    buffer = RevisionStableTTSBuffer(stable_sec=3.0)
    buffer.register("s1", 1, 0)

    with pytest.raises(ValueError, match="cannot change source_order"):
        buffer.register("s1", 2, 1)
    with pytest.raises(ValueError, match="already registered"):
        buffer.register("s2", 1, 0)


def test_stability_buffer_reset_discards_all_session_state():
    clock = FakeClock(100.0)
    buffer = RevisionStableTTSBuffer(stable_sec=3.0, clock=clock)
    buffer.register("s1", 1, 0)
    buffer.mark_ready("s1", 1, "old", "English")

    buffer.reset()
    buffer.register("s2", 1, 0)
    buffer.mark_ready("s2", 1, "new", "English")
    clock.advance(3.0)

    assert [item.sentence_id for item in buffer.drain()] == ["s2"]
