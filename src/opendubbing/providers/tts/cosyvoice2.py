"""CosyVoice2 TTS provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError
from opendubbing.utils import media


class CosyVoice2Provider(Provider):
    """Text-to-speech using CosyVoice2.

    Supports zero-shot inference. Voice cloning requires ``inputs["reference_audio"]``.
    """

    name = "cosyvoice2"
    kind = "tts"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "iic/CosyVoice2-0.5B")
        self.options = config.get("options", {})
        self._model = None

    def load_model(self) -> None:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError as exc:
            raise ProviderModelLoadError(
                "cosyvoice not installed; install opendubbing[heavy]"
            ) from exc
        try:
            self._model = CosyVoice2(self.model)
        except Exception as exc:
            raise ProviderModelLoadError(
                f"Failed to load CosyVoice2 model {self.model}"
            ) from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        text = inputs["text"]
        out_path = Path(inputs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        reference_audio = inputs.get("reference_audio")
        if reference_audio:
            ref_samples, ref_sr = media.read_audio(reference_audio, sample_rate=16000)
            result = list(
                self._model.inference_zero_shot(text, "", ref_samples.tolist(), ref_sr)
            )
        else:
            result = list(self._model.inference_sft(text))

        audio = result[0]["tts_speech"].numpy().squeeze()
        sample_rate = 22050
        media.write_audio(audio, sample_rate, out_path)
        duration = len(audio) / sample_rate
        return {"duration": duration, "path": str(out_path)}

    def release(self) -> None:
        self._model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
