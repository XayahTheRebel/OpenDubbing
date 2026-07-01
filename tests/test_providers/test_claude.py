"""Tests for ClaudeProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.translation.claude import ClaudeProvider


class TestClaudeProvider:
    def test_load_model_raises_when_anthropic_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
        provider = ClaudeProvider()
        provider.initialize({"name": "claude", "api_key": "test"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = ClaudeProvider()
        provider.initialize({"name": "claude"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_extracts_translation_text(self, monkeypatch):
        fake_content = MagicMock()
        fake_content.text = " 你好世界 "
        fake_response = MagicMock()
        fake_response.content = [fake_content]

        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        fake_anthropic = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

        provider = ClaudeProvider()
        provider.initialize(
            {"name": "claude", "api_key": "test", "model": "claude-test"}
        )
        provider.load_model()

        result = provider.infer(
            {"text": "hello world", "source_language": "en", "target_language": "zh"}
        )

        assert result["translation"] == "你好世界"
        call_args = fake_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-test"
        assert "zh" in call_args.kwargs["messages"][0]["content"]
