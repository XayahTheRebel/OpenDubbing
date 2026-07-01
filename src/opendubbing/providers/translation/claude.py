"""Claude translation provider."""

from __future__ import annotations

import os
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class ClaudeProvider(Provider):
    """Translation using Anthropic Claude."""

    name = "claude"
    kind = "translation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "claude-3-5-sonnet-latest")
        self.options = config.get("options", {})
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def load_model(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderModelLoadError(
                "anthropic not installed; install opendubbing"
            ) from exc
        if not self.api_key:
            raise ProviderModelLoadError(
                "Anthropic API key required; set ANTHROPIC_API_KEY or providers.translation.api_key"
            )
        self._client = Anthropic(api_key=self.api_key)

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            self.load_model()

        text = inputs["text"]
        target = inputs.get("target_language", "zh")
        prompt = (
            f"Translate the following text to {target}. "
            f"Return only the translation, no explanation.\n\n{text}"
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        translation = response.content[0].text.strip()
        return {"translation": translation}

    def release(self) -> None:
        self._client = None
