"""Demucs audio separation provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError
from opendubbing.utils import media


class DemucsProvider(Provider):
    """Separate audio sources using Demucs."""

    name = "demucs"
    kind = "audio_separation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "htdemucs")
        self.options = config.get("options", {})
        self._model = None

    def load_model(self) -> None:
        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise ProviderModelLoadError(
                "demucs not installed; install opendubbing[heavy]"
            ) from exc
        try:
            device = self.options.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = get_model(self.model)
            self._model.to(device)
            self._model.eval()
        except Exception as exc:
            raise ProviderModelLoadError(f"Failed to load Demucs model {self.model}") from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        audio_path = Path(inputs["audio"])
        out_dir = Path(inputs.get("out_dir", audio_path.parent))
        out_dir.mkdir(parents=True, exist_ok=True)

        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        device = self.options.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        wav = wav.to(device)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        with torch.no_grad():
            separated = self._model.forward(wav[None])

        sources = separated[0]
        source_names = self._model.sources
        stems = {}
        for source_name in source_names:
            idx = source_names.index(source_name)
            stem = sources[idx].mean(dim=0).cpu().numpy()
            stem_path = out_dir / f"{source_name}.wav"
            media.write_audio(stem, sr, stem_path)
            stems[source_name] = str(stem_path)

        return {"stems": stems}

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
