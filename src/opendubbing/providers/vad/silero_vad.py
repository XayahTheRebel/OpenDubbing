"""Silero VAD provider."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError
from opendubbing.utils import media


class SileroVADProvider(Provider):
    """Voice activity detection using Silero VAD."""

    name = "silero_vad"
    kind = "vad"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})
        self._model = None

    def load_model(self) -> None:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ProviderModelLoadError(
                "silero-vad not installed; install opendubbing[heavy]"
            ) from exc
        try:
            self._model = load_silero_vad()
        except Exception as exc:
            raise ProviderModelLoadError("Failed to load Silero VAD model") from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        from silero_vad import get_speech_timestamps

        audio_path = inputs["audio"]
        threshold = inputs.get("threshold", 0.5)
        min_silence_ms = inputs.get("min_silence_ms", 500)

        wav, sr = media.read_audio(audio_path, sample_rate=16000)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_tensor = self._model[1].tensor(wav)

        timestamps = get_speech_timestamps(
            wav_tensor,
            self._model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            sampling_rate=16000,
        )
        segments = [
            {"start": ts["start"] / 16000, "end": ts["end"] / 16000}
            for ts in timestamps
        ]
        return {"segments": segments}

    def release(self) -> None:
        self._model = None
        import gc

        gc.collect()
