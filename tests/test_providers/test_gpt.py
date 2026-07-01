"""Tests for GPTProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.translation.gpt import GPTProvider


class TestGPTProvider:
    def test_load_model_raises_when_openai_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "openai", None)
        provider = GPTProvider()
        provider.initialize({"name": "gpt", "api_key": "test"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = GPTProvider()
        provider.initialize({"name": "gpt"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_calls_chat_completions(self, monkeypatch):
        fake_message = MagicMock()
        fake_message.content = " 你好世界 "
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        fake_openai = SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))
        monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

        provider = GPTProvider()
        provider.initialize({"name": "gpt", "api_key": "test", "model": "gpt-4o-mini"})
        provider.load_model()

        result = provider.infer(
            {"text": "hello world", "source_language": "en", "target_language": "zh"}
        )

        assert result["translation"] == "你好世界"
        call_args = fake_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert any("zh" in msg["content"] for msg in call_args.kwargs["messages"])
