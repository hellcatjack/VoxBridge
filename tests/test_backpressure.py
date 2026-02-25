from voxbridge.streaming.backpressure import QueueBackpressureController


def test_backpressure_normal_under_target():
    c = QueueBackpressureController(sample_rate=16000, target_queue_sec=3.0, max_queue_sec=5.0)
    d = c.evaluate(queue_samples=int(1.0 * 16000))
    assert d.under_pressure is False
    assert d.reason == "normal"


def test_backpressure_soft_pressure_over_target():
    c = QueueBackpressureController(sample_rate=16000, target_queue_sec=3.0, max_queue_sec=5.0)
    d = c.evaluate(queue_samples=int(3.5 * 16000))
    assert d.under_pressure is True
    assert d.drop_oldest is False
    assert d.reason == "soft_pressure"
    assert d.suggested_batch_scale > 1.0


def test_backpressure_hard_overflow_drops_oldest():
    c = QueueBackpressureController(sample_rate=16000, target_queue_sec=3.0, max_queue_sec=5.0)
    d = c.evaluate(queue_samples=int(6.2 * 16000))
    assert d.under_pressure is True
    assert d.drop_oldest is True
    assert d.reason == "hard_overflow"

