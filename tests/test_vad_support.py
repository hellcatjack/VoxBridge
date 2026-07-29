import numpy as np
import pytest

from voxbridge.streaming.vad_support import (
    AudioPreRollBuffer,
    SileroShadowObserver,
    create_silero_onnx_observer,
)


def test_audio_preroll_keeps_newest_samples_and_replays_once():
    buffer = AudioPreRollBuffer(sample_rate=10, duration_sec=0.4)

    buffer.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    buffer.append(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    combined, replayed_samples = buffer.prepend_to(np.array([7.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(combined, np.array([3, 4, 5, 6, 7, 8], dtype=np.float32))
    assert replayed_samples == 4
    assert buffer.buffered_samples == 0

    untouched, replayed_samples = buffer.prepend_to(np.array([9.0], dtype=np.float32))
    np.testing.assert_array_equal(untouched, np.array([9.0], dtype=np.float32))
    assert replayed_samples == 0


def test_audio_preroll_copies_input_storage():
    buffer = AudioPreRollBuffer(sample_rate=10, duration_sec=0.4)
    source = np.array([1.0, 2.0], dtype=np.float32)

    buffer.append(source)
    source[:] = 9.0

    combined, _ = buffer.prepend_to(np.array([], dtype=np.float32))
    np.testing.assert_array_equal(combined, np.array([1.0, 2.0], dtype=np.float32))


def test_silero_shadow_buffers_fixed_frames_and_reports_transition():
    observer = SileroShadowObserver(
        runner=lambda frame: float(np.mean(frame)),
        sample_rate=16_000,
        frame_samples=4,
        threshold=0.5,
    )

    first = observer.feed(np.array([0.1, 0.1, 0.1], dtype=np.float32))
    assert first.frames == 0
    assert first.pending_samples == 3

    second = observer.feed(np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32))
    assert second.frames == 2
    assert second.pending_samples == 0
    assert second.last_probability == pytest.approx(0.9)
    assert second.mean_probability == pytest.approx(0.6)
    assert second.max_probability == pytest.approx(0.9)
    assert second.is_speech is True
    assert second.state_changed is True


def test_silero_shadow_inference_failure_disables_only_observer():
    calls = 0

    def broken_runner(_frame):
        nonlocal calls
        calls += 1
        raise RuntimeError("model failure")

    observer = SileroShadowObserver(
        runner=broken_runner,
        sample_rate=16_000,
        frame_samples=4,
    )

    failed = observer.feed(np.ones(4, dtype=np.float32))
    ignored = observer.feed(np.ones(4, dtype=np.float32))

    assert failed.available is False
    assert failed.error == "RuntimeError: model failure"
    assert ignored.available is False
    assert calls == 1


def test_create_silero_onnx_observer_loads_model_and_resets_state():
    class FakeModel:
        def __init__(self):
            self.reset_calls = 0
            self.sample_rates = []

        def reset_states(self):
            self.reset_calls += 1

        def __call__(self, tensor, sample_rate):
            self.sample_rates.append(sample_rate)
            return tensor.mean()

    model = FakeModel()
    observer = create_silero_onnx_observer(
        threshold=0.5,
        load_model=lambda **kwargs: model,
    )

    observation = observer.feed(np.full(512, 0.75, dtype=np.float32))

    assert model.reset_calls == 1
    assert model.sample_rates == [16_000]
    assert observation.last_probability == pytest.approx(0.75)
