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
        start = inputs.get("start", 0.0)
        end = inputs.get("end", 0.0)
        duration = max(0.0, end - start)

        aligned = []
        if not words:
            return {"words": aligned, "confidence": 1.0}

        if isinstance(words[0], dict):
            # Already contains timings; preserve them and add empty phonemes.
            aligned = [
                {
                    "text": w.get("text", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                    "phonemes": [],
                }
                for w in words
            ]
        else:
            # Strings only: spread evenly across [start, end].
            word_duration = duration / len(words)
            for i, text in enumerate(words):
                word_start = start + i * word_duration
                word_end = word_start + word_duration
                aligned.append(
                    {"text": text, "start": word_start, "end": word_end, "phonemes": []}
                )

        return {"words": aligned, "confidence": 1.0}

    def release(self) -> None:
        pass
