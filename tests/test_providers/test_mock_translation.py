"""Tests for MockTranslationProvider."""

from __future__ import annotations

from opendubbing.providers.translation.mock_translation import MockTranslationProvider


class TestMockTranslationProvider:
    def test_infer_returns_chinese_translation(self):
        provider = MockTranslationProvider()
        provider.initialize({})
        result = provider.infer(
            {"text": "hello world", "source_language": "en", "target_language": "zh"}
        )
        assert result["translation"] == "[hello world] 的中文翻译"

    def test_infer_returns_default_for_non_chinese(self):
        provider = MockTranslationProvider()
        provider.initialize({})
        result = provider.infer(
            {"text": "hello", "source_language": "en", "target_language": "fr"}
        )
        assert result["translation"] == "translation of hello"
