"""GPT translation provider."""

from __future__ import annotations

import os
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class GPTProvider(Provider):
    """Translation using OpenAI GPT."""

    name = "gpt"
    kind = "translation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "gpt-4o-mini")
        self.options = config.get("options", {})
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self._client = None

    def load_model(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderModelLoadError(
                "openai not installed; install opendubbing"
            ) from exc
        if not self.api_key:
            raise ProviderModelLoadError(
                "OpenAI API key required; set OPENAI_API_KEY or providers.translation.api_key"
            )
        self._client = OpenAI(api_key=self.api_key)

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            self.load_model()

        text = inputs["text"]
        target = inputs.get("target_language", "zh")
        system = (
            f"You are a translator. Translate the user's text to {target}. "
            "Return only the translation, no explanation."
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        translation = response.choices[0].message.content.strip()
        return {"translation": translation}

    def release(self) -> None:
        self._client = None
