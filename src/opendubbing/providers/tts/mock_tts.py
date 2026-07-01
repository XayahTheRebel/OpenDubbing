"""Mock TTS provider for smoke testing."""

from __future__ import annotations

from typing import Any

import numpy as np

from opendubbing.core.interfaces import Provider
from opendubbing.utils import media


class MockTTSProvider(Provider):
    """Write a short silent wav for pipeline smoke tests."""

    name = "mock_tts"
    kind = "tts"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})

    def load_model(self) -> None:
        pass

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        out_path = inputs["out_path"]
        duration = self.options.get("duration", 1.0)
        sample_rate = 16000
        samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
        media.write_audio(samples, sample_rate, out_path)
        return {"duration": duration, "path": out_path}

    def release(self) -> None:
        pass
