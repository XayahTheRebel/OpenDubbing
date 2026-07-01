"""Gemini translation provider."""

from __future__ import annotations

import os
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class GeminiProvider(Provider):
    """Translation using Google Gemini."""

    name = "gemini"
    kind = "translation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "gemini-1.5-flash")
        self.options = config.get("options", {})
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self._model = None

    def load_model(self) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderModelLoadError(
                "google-generativeai not installed; install opendubbing"
            ) from exc
        if not self.api_key:
            raise ProviderModelLoadError(
                "Google API key required; set GOOGLE_API_KEY or providers.translation.api_key"
            )
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model)

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            self.load_model()

        text = inputs["text"]
        target = inputs.get("target_language", "zh")
        prompt = (
            f"Translate the following text to {target}. "
            f"Return only the translation, no explanation.\n\n{text}"
        )
        response = self._model.generate_content(prompt)
        translation = response.text.strip()
        return {"translation": translation}

    def release(self) -> None:
        self._model = None
