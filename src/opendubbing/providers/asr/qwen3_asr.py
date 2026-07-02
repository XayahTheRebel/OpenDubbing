"""Qwen3 ASR provider (fallback to FunASR for V1)."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class Qwen3ASRProvider(Provider):
    """Automatic speech recognition using Qwen3-ASR backend.

    V1 uses FunASR as the runnable backend; the Qwen3-ASR model can be
    swapped in later without changing Engine code.
    """

    name = "qwen3_asr"
    kind = "asr"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model") or "paraformer-zh"
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
            raise ProviderModelLoadError(f"Failed to load ASR model {self.model}") from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        audio_path = inputs["audio"]
        language = inputs.get("language", "auto")
        result = self._model.generate(input=audio_path, batch_size_s=300)

        segments = []
        for item in result:
            text = item.get("text", "").strip()
            if not text:
                continue
            ts = item.get("timestamp", [])
            start = ts[0][0] / 1000 if ts else 0.0
            end = ts[-1][1] / 1000 if ts else 0.0
            words = [
                {"text": w, "start": s / 1000, "end": e / 1000}
                for w, (s, e) in zip(item.get("sentence_words", []), ts, strict=False)
            ]
            segments.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "confidence": item.get("confidence", 1.0),
                    "words": words,
                }
            )

        return {"language": language if language != "auto" else "zh", "segments": segments}

    def release(self) -> None:
        self._model = None
        import gc

        gc.collect()
