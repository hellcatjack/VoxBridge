import pytest

from voxbridge.tts.jobs import OrderedTTSBuffer, TTSJobNotFound, TTSJobRegistry, TTSQueueFull


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


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


def test_order_buffer_waits_for_earlier_translation():
    buffer = OrderedTTSBuffer()
    buffer.register("s1", revision=1, source_order=0)
    buffer.register("s2", revision=1, source_order=1)

    assert buffer.mark_ready("s2", 1, "second", "English") == []
    ready = buffer.mark_ready("s1", 1, "first", "English")

    assert [item.sentence_id for item in ready] == ["s1", "s2"]


def test_order_buffer_skips_failed_earlier_translation():
    buffer = OrderedTTSBuffer()
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)

    assert buffer.mark_ready("s2", 1, "second", "English") == []
    ready = buffer.mark_failed("s1", 1)

    assert [item.sentence_id for item in ready] == ["s2"]


def test_order_buffer_rejects_stale_revision_before_emit():
    buffer = OrderedTTSBuffer()
    buffer.register("s1", 1, 0)
    buffer.register("s1", 2, 0)

    assert buffer.mark_ready("s1", 1, "old", "English") == []
    assert buffer.mark_ready("s1", 2, "new", "English")[0].text == "new"


def test_order_buffer_never_emits_a_sentence_twice():
    buffer = OrderedTTSBuffer()
    buffer.register("s1", 1, 0)

    assert len(buffer.mark_ready("s1", 1, "first", "English")) == 1
    buffer.register("s1", 2, 0)
    assert buffer.mark_ready("s1", 2, "changed", "English") == []


def test_order_buffer_reset_discards_pending_items():
    buffer = OrderedTTSBuffer()
    buffer.register("s1", 1, 0)
    buffer.register("s2", 1, 1)
    assert buffer.mark_ready("s2", 1, "second", "English") == []

    buffer.reset()
    buffer.register("s3", 1, 0)

    assert [item.sentence_id for item in buffer.mark_ready("s3", 1, "third", "English")] == ["s3"]
