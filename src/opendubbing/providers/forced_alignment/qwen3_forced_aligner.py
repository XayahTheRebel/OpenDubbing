"""Qwen3 Forced Aligner provider (fallback to FunASR for V1)."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class Qwen3ForcedAlignerProvider(Provider):
    """Forced alignment using Qwen3-ForcedAligner backend.

    V1 uses FunASR as the runnable backend; the Qwen3-ForcedAligner model can
    be swapped in later without changing Engine code.
    """

    name = "qwen3_forced_aligner"
    kind = "forced_alignment"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model") or "fa-zh"
        self.options = config.get("options", {})
        self._model = None

    def load_model(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ProviderModelLoadError(
                "funasr not installed; install opendubbing[heavy]"
            ) from exc
        try:
            self._model = AutoModel(model=self.model)
        except Exception as exc:
            raise ProviderModelLoadError(
                f"Failed to load forced alignment model {self.model}"
            ) from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        audio_path = inputs["audio"]
        text = inputs["text"]
        result = self._model.generate(input=audio_path, text=text, batch_size=1)

        words = []
        confidence = 1.0
        for item in result:
            confidence = item.get("confidence", confidence)
            tokens = item.get("tokens", [])
            ts = item.get("timestamp", [])
            for token, (start_ms, end_ms) in zip(tokens, ts, strict=False):
                words.append(
                    {
                        "text": token,
                        "start": start_ms / 1000,
                        "end": end_ms / 1000,
                        "phonemes": [],
                    }
                )

        return {"words": words, "confidence": confidence}

    def release(self) -> None:
        self._model = None
        import gc

        gc.collect()
