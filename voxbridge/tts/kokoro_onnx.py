from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


class TTSConfigurationError(RuntimeError):
    """Raised when TTS assets or runtime settings are invalid."""


class TTSSynthesisError(RuntimeError):
    """Raised when a synthesis request cannot be completed."""


@dataclass(frozen=True, slots=True)
class KokoroTTSConfig:
    english_model_path: Path
    english_voices_path: Path
    chinese_model_path: Path
    chinese_voices_path: Path
    chinese_config_path: Path
    english_voice: str = "am_michael"
    chinese_voice: str = "zf_001"
    speed: float = 1.05
    cpu_threads: int = 4
    max_chars: int = 1000


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    wav_bytes: bytes
    sample_rate: int
    duration_ms: int


KokoroFactory = Callable[..., Any]
ChineseG2PFactory = Callable[[], Any]


class KokoroOnnxSynthesizer:
    def __init__(
        self,
        *,
        config: KokoroTTSConfig,
        kokoro_factory: KokoroFactory | None = None,
        zh_g2p_factory: ChineseG2PFactory | None = None,
    ) -> None:
        self.config = config
        self._validate_config()
        self._kokoro_factory = kokoro_factory or self._create_cpu_kokoro
        self._zh_g2p_factory = zh_g2p_factory or self._create_chinese_g2p
        self._models: dict[str, Any] = {}
        self._zh_g2p: Any | None = None
        self._inference_lock = threading.Lock()

    def _validate_config(self) -> None:
        assets = (
            ("English model", self.config.english_model_path),
            ("English voices", self.config.english_voices_path),
            ("Chinese model", self.config.chinese_model_path),
            ("Chinese voices", self.config.chinese_voices_path),
            ("Chinese vocabulary config", self.config.chinese_config_path),
        )
        for label, value in assets:
            if not Path(value).is_file():
                raise TTSConfigurationError(f"{label} asset does not exist: {value}")
        if not 0.5 <= self.config.speed <= 2.0:
            raise TTSConfigurationError("TTS speed must be between 0.5 and 2.0")
        if self.config.cpu_threads <= 0:
            raise TTSConfigurationError("TTS CPU threads must be positive")
        if self.config.max_chars <= 0:
            raise TTSConfigurationError("TTS maximum characters must be positive")
        if not self.config.english_voice.strip() or not self.config.chinese_voice.strip():
            raise TTSConfigurationError("TTS voices must be non-empty")

    @staticmethod
    def _create_cpu_kokoro(
        *,
        model_path: Path,
        voices_path: Path,
        vocab_config: Path | None,
        cpu_threads: int,
        providers: tuple[str, ...],
    ) -> Any:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=list(providers),
        )
        return Kokoro.from_session(
            session,
            str(voices_path),
            vocab_config=str(vocab_config) if vocab_config is not None else None,
        )

    @staticmethod
    def _create_chinese_g2p() -> Any:
        from misaki.zh import ZHG2P

        return ZHG2P(version="1.1")

    def _model(self, language: str) -> Any:
        model = self._models.get(language)
        if model is not None:
            return model
        if language == "English":
            model_path = self.config.english_model_path
            voices_path = self.config.english_voices_path
            vocab_config = None
        else:
            model_path = self.config.chinese_model_path
            voices_path = self.config.chinese_voices_path
            vocab_config = self.config.chinese_config_path
        model = self._kokoro_factory(
            model_path=Path(model_path),
            voices_path=Path(voices_path),
            vocab_config=Path(vocab_config) if vocab_config is not None else None,
            cpu_threads=self.config.cpu_threads,
            providers=("CPUExecutionProvider",),
        )
        self._models[language] = model
        return model

    @staticmethod
    def _normalize_language(target_language: str) -> str:
        normalized = target_language.strip().lower()
        if normalized in {"english", "en"}:
            return "English"
        if normalized in {"chinese", "zh"}:
            return "Chinese"
        raise TTSSynthesisError(f"unsupported target language: {target_language}")

    def synthesize(
        self,
        text: str,
        target_language: str,
        *,
        speed: float | None = None,
    ) -> SynthesizedAudio:
        if not isinstance(text, str) or not text.strip():
            raise TTSSynthesisError("TTS text must be non-empty")
        if len(text) > self.config.max_chars:
            raise TTSSynthesisError(
                f"TTS text exceeds the {self.config.max_chars} character limit"
            )
        language = self._normalize_language(target_language)
        effective_speed = self.config.speed if speed is None else float(speed)
        if not math.isfinite(effective_speed) or not 0.5 <= effective_speed <= 2.0:
            raise TTSSynthesisError("TTS speed must be between 0.5 and 2.0")

        try:
            with self._inference_lock:
                model = self._model(language)
                if language == "English":
                    model_input = text
                    voice = self.config.english_voice
                    lang = "en-us"
                    is_phonemes = False
                else:
                    if self._zh_g2p is None:
                        self._zh_g2p = self._zh_g2p_factory()
                    phoneme_result = self._zh_g2p(text)
                    model_input = (
                        phoneme_result[0]
                        if isinstance(phoneme_result, tuple)
                        else phoneme_result
                    )
                    if not isinstance(model_input, str) or not model_input:
                        raise TTSSynthesisError("Chinese G2P returned invalid phonemes")
                    voice = self.config.chinese_voice
                    lang = "cmn"
                    is_phonemes = True
                samples, sample_rate = model.create(
                    model_input,
                    voice=voice,
                    speed=effective_speed,
                    lang=lang,
                    is_phonemes=is_phonemes,
                )
        except TTSSynthesisError:
            raise
        except Exception as exc:
            raise TTSSynthesisError("Kokoro synthesis failed") from exc

        samples_array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples_array.size == 0 or int(sample_rate) <= 0:
            raise TTSSynthesisError("Kokoro returned empty audio")
        import soundfile as sf

        output = io.BytesIO()
        sf.write(output, np.clip(samples_array, -1.0, 1.0), int(sample_rate), format="WAV", subtype="PCM_16")
        duration_ms = round(samples_array.size * 1000 / int(sample_rate))
        return SynthesizedAudio(output.getvalue(), int(sample_rate), duration_ms)
