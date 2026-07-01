"""Mock forced alignment provider for smoke testing."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider


class MockForcedAlignerProvider(Provider):
    """Return word timings unchanged for pipeline smoke tests."""

    name = "mock_forced_aligner"
    kind = "forced_alignment"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    def load_model(self) -> None:
        pass

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        words = inputs.get("words", [])
        aligned = [
            {"text": text, "start": 0.0, "end": 0.0, "phonemes": []}
            for text in words
        ]
        return {"words": aligned, "confidence": 1.0}

    def release(self) -> None:
        pass
