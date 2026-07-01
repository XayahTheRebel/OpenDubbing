"""Mock face animation provider for smoke testing."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider
from opendubbing.utils import media


class MockFaceProvider(Provider):
    """Mux audio directly onto video for pipeline smoke tests."""

    name = "mock_face"
    kind = "face"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    def load_model(self) -> None:
        pass

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        media.mux_audio_video(
            inputs["video"], inputs["audio"], inputs["out_path"], codec="libx264"
        )
        return {"path": inputs["out_path"], "fallback": True}

    def release(self) -> None:
        pass
