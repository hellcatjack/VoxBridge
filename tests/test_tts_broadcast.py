import asyncio

import pytest

from voxbridge.tts.broadcast import (
    TTSBroadcastHub,
    TTSBroadcastNotFound,
    TTSBroadcastQueueFull,
)
from voxbridge.tts.jobs import TTSReadyItem


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class TokenFactory:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"token-{self.value}"


def create_hub(
    *,
    clock=None,
    ttl_sec: float = 30.0,
    max_jobs: int = 8,
    listener_queue_size: int = 8,
) -> TTSBroadcastHub:
    return TTSBroadcastHub(
        ttl_sec=ttl_sec,
        max_jobs=max_jobs,
        listener_queue_size=listener_queue_size,
        clock=clock or FakeClock(),
        token_factory=TokenFactory(),
    )


def ready_item(
    text: str = "Stable translation.",
    *,
    sentence_id: str = "s1",
    revision: int = 1,
    source_order: int = 0,
    target_language: str = "English",
) -> TTSReadyItem:
    return TTSReadyItem(
        sentence_id=sentence_id,
        revision=revision,
        source_order=source_order,
        target_language=target_language,
        text=text,
    )


def test_publish_fans_one_job_to_current_listeners_only():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")

    job = hub.publish(ready_item())

    assert job is not None
    first_event = first.queue.get_nowait()
    second_event = second.queue.get_nowait()
    assert first_event == second_event
    assert first_event["type"] == "tts_job"
    assert first_event["job_id"] == job.job_id
    assert "text" not in first_event

    late = hub.register("owner-c")
    with pytest.raises(asyncio.QueueEmpty):
        late.queue.get_nowait()


def test_publish_without_listener_retains_nothing():
    hub = create_hub()

    assert hub.publish(ready_item()) is None
    assert hub.job_count == 0


def test_acknowledgement_is_per_listener_and_last_ack_deletes_job():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item())

    assert hub.acknowledge(job.job_id, first.listener_id, "owner-a") is True
    assert hub.job_count == 1
    assert hub.acknowledge(job.job_id, second.listener_id, "owner-b") is True
    assert hub.job_count == 0


def test_disconnect_releases_only_that_listener():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item())

    assert hub.unregister(first.listener_id, "owner-a") == 1
    claimed = hub.claim_audio(job.job_id, second.listener_id, "owner-b")
    assert claimed.job_id == job.job_id
    hub.release_audio(job.job_id)


def test_audio_lease_prevents_disconnect_from_deleting_inflight_job():
    hub = create_hub()
    listener = hub.register("owner-a")
    job = hub.publish(ready_item())

    hub.claim_audio(job.job_id, listener.listener_id, "owner-a")
    hub.unregister(listener.listener_id, "owner-a")

    assert hub.job_count == 1
    hub.release_audio(job.job_id)
    assert hub.job_count == 0


def test_audio_is_cached_once_and_visible_to_all_assigned_listeners():
    hub = create_hub()
    first = hub.register("owner-a")
    second = hub.register("owner-b")
    job = hub.publish(ready_item())

    assert hub.claim_audio(job.job_id, first.listener_id, "owner-a").audio_bytes is None
    cached = hub.cache_audio(
        job.job_id,
        b"RIFF-shared-audio",
        sample_rate=24000,
        duration_ms=750,
    )
    hub.release_audio(job.job_id)
    second_view = hub.claim_audio(job.job_id, second.listener_id, "owner-b")

    assert cached.audio_bytes == b"RIFF-shared-audio"
    assert second_view.audio_bytes == cached.audio_bytes
    assert second_view.sample_rate == 24000
    assert second_view.duration_ms == 750
    hub.release_audio(job.job_id)


def test_foreign_owner_cannot_claim_or_acknowledge_job():
    hub = create_hub()
    listener = hub.register("owner-a")
    job = hub.publish(ready_item(text="Private translation."))

    with pytest.raises(TTSBroadcastNotFound):
        hub.claim_audio(job.job_id, listener.listener_id, "owner-b")
    assert hub.acknowledge(job.job_id, listener.listener_id, "owner-b") is False


def test_expired_job_is_not_exposed():
    clock = FakeClock()
    hub = create_hub(clock=clock, ttl_sec=30.0)
    listener = hub.register("owner-a")
    job = hub.publish(ready_item(text="Expiring translation."))

    clock.value = 130.01

    with pytest.raises(TTSBroadcastNotFound):
        hub.claim_audio(job.job_id, listener.listener_id, "owner-a")
    assert hub.job_count == 0


def test_capacity_rejects_new_job_without_evicting_unread_job():
    hub = create_hub(max_jobs=1)
    listener = hub.register("owner-a")
    first = hub.publish(ready_item())

    with pytest.raises(TTSBroadcastQueueFull):
        hub.publish(ready_item("Second.", sentence_id="s2", source_order=1))

    claimed = hub.claim_audio(first.job_id, listener.listener_id, "owner-a")
    assert claimed.job_id == first.job_id
    hub.release_audio(first.job_id)


def test_overflow_disconnects_only_the_slow_listener():
    hub = create_hub(listener_queue_size=1)
    slow = hub.register("owner-a")
    fast = hub.register("owner-b")
    hub.publish(ready_item("First."))
    fast.queue.get_nowait()

    second = hub.publish(ready_item("Second.", sentence_id="s2", source_order=1))

    assert slow.overflowed.is_set()
    assert fast.queue.get_nowait()["job_id"] == second.job_id
    assert hub.listener_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ttl_sec": 0}, "ttl_sec"),
        ({"max_jobs": 0}, "max_jobs"),
        ({"listener_queue_size": 0}, "listener_queue_size"),
    ],
)
def test_invalid_hub_bounds_are_rejected(kwargs, message):
    values = {"ttl_sec": 30, "max_jobs": 8, "listener_queue_size": 8}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TTSBroadcastHub(**values)
