"""Tests for MockASRProvider."""

from __future__ import annotations

from opendubbing.providers.asr.mock_asr import MockASRProvider


class TestMockASRProvider:
    def test_infer_returns_expected_segment(self):
        provider = MockASRProvider()
        provider.initialize({"name": "mock_asr", "options": {"language": "en"}})
        provider.load_model()

        result = provider.infer({"audio": "ignored.wav", "language": "auto"})

        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["text"] == "hello world"
        assert seg["start"] == 0.0
        assert seg["end"] == 1.0
        assert len(seg["words"]) == 2
        assert seg["words"][0]["text"] == "hello"
        assert seg["words"][1]["text"] == "world"

    def test_load_model_is_noop(self):
        provider = MockASRProvider()
        provider.initialize({})
        provider.load_model()
        assert provider.config == {}
