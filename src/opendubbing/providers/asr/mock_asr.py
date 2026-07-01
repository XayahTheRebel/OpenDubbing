"""Mock ASR provider for smoke testing without heavy dependencies."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider


class MockASRProvider(Provider):
    """Return a hard-coded transcription for pipeline smoke tests."""

    name = "mock_asr"
    kind = "asr"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})

    def load_model(self) -> None:
        pass

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "language": self.options.get("language", "en"),
            "segments": [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": 1.0,
                    "words": [
                        {"text": "hello", "start": 0.0, "end": 0.5},
                        {"text": "world", "start": 0.5, "end": 1.0},
                    ],
                }
            ],
        }

    def release(self) -> None:
        pass
