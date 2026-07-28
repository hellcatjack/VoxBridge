import io
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from voxbridge.tts.kokoro_onnx import (
    KokoroOnnxSynthesizer,
    KokoroTTSConfig,
    TTSConfigurationError,
    TTSSynthesisError,
)


@dataclass
class FakeCreateCall:
    text: str
    voice: str
    speed: float
    lang: str
    is_phonemes: bool


class FakeKokoro:
    def __init__(self, samples=None, sample_rate: int = 24000) -> None:
        self.samples = np.asarray(
            samples if samples is not None else [0.0, 0.25, -0.25], dtype=np.float32
        )
        self.sample_rate = sample_rate
        self.calls: list[FakeCreateCall] = []

    def create(self, text, *, voice, speed, lang, is_phonemes):
        self.calls.append(FakeCreateCall(text, voice, speed, lang, is_phonemes))
        return self.samples, self.sample_rate


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.models: list[FakeKokoro] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        model = FakeKokoro()
        self.models.append(model)
        return model


class FakeChineseG2P:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str):
        self.calls.append(text)
        return "ni↓ xau↓", None


def make_config(tmp_path, **overrides) -> KokoroTTSConfig:
    paths = {}
    for name in (
        "english_model_path",
        "english_voices_path",
        "chinese_model_path",
        "chinese_voices_path",
        "chinese_config_path",
    ):
        path = tmp_path / name
        path.write_bytes(b"asset")
        paths[name] = path
    paths.update(overrides)
    return KokoroTTSConfig(**paths)


def test_english_synthesis_returns_pcm16_wav(tmp_path):
    factory = FakeFactory()
    synth = KokoroOnnxSynthesizer(config=make_config(tmp_path), kokoro_factory=factory)

    audio = synth.synthesize("The translation is stable.", "English")

    assert audio.wav_bytes[:4] == b"RIFF"
    assert audio.sample_rate == 24000
    assert audio.duration_ms == 0
    with wave.open(io.BytesIO(audio.wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 3
    call = factory.models[0].calls[0]
    assert call.voice == "af_heart"
    assert call.lang == "en-us"
    assert call.is_phonemes is False


def test_chinese_synthesis_uses_misaki_phonemes(tmp_path):
    factory = FakeFactory()
    g2p = FakeChineseG2P()
    synth = KokoroOnnxSynthesizer(
        config=make_config(tmp_path),
        kokoro_factory=factory,
        zh_g2p_factory=lambda: g2p,
    )

    synth.synthesize("稳定的译文。", "Chinese")

    assert g2p.calls == ["稳定的译文。"]
    call = factory.models[0].calls[0]
    assert call.text == "ni↓ xau↓"
    assert call.voice == "zf_001"
    assert call.lang == "cmn"
    assert call.is_phonemes is True
    assert factory.calls[0]["vocab_config"].name == "chinese_config_path"


def test_adapter_passes_cpu_only_runtime_configuration(tmp_path):
    factory = FakeFactory()
    synth = KokoroOnnxSynthesizer(
        config=make_config(tmp_path, cpu_threads=6), kokoro_factory=factory
    )

    synth.synthesize("Ready.", "English")

    assert factory.calls[0]["providers"] == ("CPUExecutionProvider",)
    assert factory.calls[0]["cpu_threads"] == 6


def test_adapter_rejects_missing_assets_before_runtime_import(tmp_path):
    config = make_config(tmp_path)
    config.english_model_path.unlink()

    with pytest.raises(TTSConfigurationError, match="English model"):
        KokoroOnnxSynthesizer(
            config=config,
            kokoro_factory=lambda **kwargs: pytest.fail("runtime must not be loaded"),
        )


def test_adapter_rejects_unsupported_language_and_oversized_text(tmp_path):
    synth = KokoroOnnxSynthesizer(config=make_config(tmp_path), kokoro_factory=FakeFactory())

    with pytest.raises(TTSSynthesisError, match="target language"):
        synth.synthesize("Stable.", "French")
    with pytest.raises(TTSSynthesisError, match="1000"):
        synth.synthesize("x" * 1001, "English")


def test_models_and_g2p_load_lazily_once(tmp_path):
    factory = FakeFactory()
    g2p_factory_calls = []

    def make_g2p():
        g2p_factory_calls.append(True)
        return FakeChineseG2P()

    synth = KokoroOnnxSynthesizer(
        config=make_config(tmp_path),
        kokoro_factory=factory,
        zh_g2p_factory=make_g2p,
    )

    assert factory.calls == []
    synth.synthesize("One.", "English")
    synth.synthesize("Two.", "English")
    synth.synthesize("一。", "Chinese")
    synth.synthesize("二。", "Chinese")

    assert len(factory.calls) == 2
    assert len(g2p_factory_calls) == 1
