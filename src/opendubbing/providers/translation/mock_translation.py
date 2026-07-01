"""Mock translation provider for smoke testing."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider


class MockTranslationProvider(Provider):
    """Return a hard-coded translation for pipeline smoke tests."""

    name = "mock_translation"
    kind = "translation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})

    def load_model(self) -> None:
        pass

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text", "")
        target = inputs.get("target_language", "zh")
        translation = f"[{text}] 的中文翻译" if target == "zh" else f"translation of {text}"
        return {"translation": translation}

    def release(self) -> None:
        pass
