"""Tests for GeminiProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.translation.gemini import GeminiProvider


class TestGeminiProvider:
    def test_load_model_raises_when_genai_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "google", None)
        provider = GeminiProvider()
        provider.initialize({"name": "gemini", "api_key": "test"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GeminiProvider()
        provider.initialize({"name": "gemini"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_extracts_text(self, monkeypatch):
        fake_model_instance = MagicMock()
        fake_response = MagicMock()
        fake_response.text = " 你好世界 "
        fake_model_instance.generate_content.return_value = fake_response

        fake_genai = MagicMock()
        fake_genai.GenerativeModel = MagicMock(return_value=fake_model_instance)
        fake_google = SimpleNamespace(generativeai=fake_genai)
        monkeypatch.setitem(__import__("sys").modules, "google", fake_google)

        provider = GeminiProvider()
        provider.initialize(
            {"name": "gemini", "api_key": "test", "model": "gemini-test"}
        )
        provider.load_model()

        result = provider.infer(
            {"text": "hello world", "source_language": "en", "target_language": "zh"}
        )

        assert result["translation"] == "你好世界"
        fake_genai.GenerativeModel.assert_called_once_with("gemini-test")
        fake_model_instance.generate_content.assert_called_once()
        prompt = fake_model_instance.generate_content.call_args[0][0]
        assert "zh" in prompt
