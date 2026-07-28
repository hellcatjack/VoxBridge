"""Text-to-speech support for stable translated sentences."""

from .broadcast import (
    BroadcastTTSJob,
    TTSBroadcastError,
    TTSBroadcastHub,
    TTSBroadcastNotFound,
    TTSBroadcastQueueFull,
    TTSListenerSubscription,
)
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
    "BroadcastTTSJob",
    "OrderedTTSBuffer",
    "TTSBroadcastError",
    "TTSBroadcastHub",
    "TTSBroadcastNotFound",
    "TTSBroadcastQueueFull",
    "TTSListenerSubscription",
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
