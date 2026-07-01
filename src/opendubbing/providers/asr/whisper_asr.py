"""Whisper ASR provider using openai-whisper."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class WhisperASRProvider(Provider):
    """Automatic speech recognition using OpenAI Whisper.

    The provider auto-detects the source language and returns both segment-level
    and approximate word-level timestamps, so the timeline engine can use them
    directly when forced alignment is disabled.
    """

    name = "whisper"
    kind = "asr"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "base")
        self.options = config.get("options", {})
        self._model = None

    def load_model(self) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise ProviderModelLoadError(
                "openai-whisper not installed; run: pip install openai-whisper"
            ) from exc
        try:
            self._model = whisper.load_model(self.model)
        except Exception as exc:
            raise ProviderModelLoadError(
                f"Failed to load Whisper model {self.model}"
            ) from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        audio_path = inputs["audio"]
        language = inputs.get("language", "auto")

        kwargs: dict[str, Any] = {"word_timestamps": True}
        if language != "auto":
            kwargs["language"] = language

        result = self._model.transcribe(audio_path, **kwargs)

        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append(
                    {
                        "text": w.get("word", "").strip(),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                    }
                )
            if not words and seg.get("text", "").strip():
                text = seg.get("text", "").strip()
                raw_words = text.split()
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                duration = end - start
                for i, word in enumerate(raw_words):
                    ws = start + (i / len(raw_words)) * duration
                    we = start + ((i + 1) / len(raw_words)) * duration
                    words.append({"text": word, "start": ws, "end": we})

            segments.append(
                {
                    "text": seg.get("text", "").strip(),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "confidence": seg.get("avg_logprob", 0.0),
                    "words": words,
                }
            )

        detected = result.get("language", language if language != "auto" else "en")
        return {"language": detected, "segments": segments}

    def release(self) -> None:
        self._model = None
        import gc

        gc.collect()
