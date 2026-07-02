"""DeepFilterNet3 noise reduction provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torchaudio

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class DeepFilterNet3Provider(Provider):
    """Reduce noise using DeepFilterNet3."""

    name = "deep_filter_net3"
    kind = "noise_reduction"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})
        self._model = None
        self._state = None

    def load_model(self) -> None:
        try:
            from df import init_df
        except ImportError as exc:
            raise ProviderModelLoadError(
                "deepfilternet not installed; install opendubbing[heavy]"
            ) from exc
        try:
            self._model, self._state, _ = init_df()
        except Exception as exc:
            raise ProviderModelLoadError("Failed to load DeepFilterNet3 model") from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None or self._state is None:
            self.load_model()
        assert self._model is not None and self._state is not None

        from df import enhance

        audio_path = Path(inputs["audio"])
        out_path = Path(inputs.get("out_path", audio_path.parent / "enhanced.wav"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = torchaudio.load(str(audio_path))
        enhanced = enhance(self._model, self._state, wav)
        torchaudio.save(str(out_path), enhanced, sample_rate=sr)
        return {"enhanced": str(out_path)}

    def release(self) -> None:
        self._model = None
        self._state = None
        import gc

        gc.collect()
