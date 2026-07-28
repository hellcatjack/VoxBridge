"""Text-to-speech support for stable translated sentences."""

from .jobs import (
    OrderedTTSBuffer,
    TTSJob,
    TTSJobNotFound,
    TTSJobRegistry,
    TTSQueueFull,
    TTSReadyItem,
)
from .kokoro_onnx import (
    KokoroOnnxSynthesizer,
    KokoroTTSConfig,
    SynthesizedAudio,
    TTSConfigurationError,
    TTSSynthesisError,
)

__all__ = [
    "OrderedTTSBuffer",
    "TTSJob",
    "TTSJobNotFound",
    "TTSJobRegistry",
    "TTSQueueFull",
    "TTSReadyItem",
    "KokoroOnnxSynthesizer",
    "KokoroTTSConfig",
    "SynthesizedAudio",
    "TTSConfigurationError",
    "TTSSynthesisError",
]
