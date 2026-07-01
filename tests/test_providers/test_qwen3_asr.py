"""Tests for Qwen3ASRProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.asr.qwen3_asr import Qwen3ASRProvider


class TestQwen3ASRProvider:
    def test_load_model_raises_when_funasr_missing(self, monkeypatch):
        monkeypatch.setitem(
            __import__("sys").modules, "funasr", None
        )
        provider = Qwen3ASRProvider()
        provider.initialize({"name": "qwen3_asr"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_converts_timestamps_to_seconds(self, monkeypatch, tmp_path):
        fake_model = MagicMock()
        fake_model.generate.return_value = [
            {
                "text": "你好世界",
                "timestamp": [[100, 500], [500, 900]],
                "sentence_words": ["你好", "世界"],
                "confidence": 0.95,
            }
        ]
        fake_automodel = MagicMock(return_value=fake_model)

        fake_funasr = SimpleNamespace(AutoModel=fake_automodel)
        monkeypatch.setitem(__import__("sys").modules, "funasr", fake_funasr)

        provider = Qwen3ASRProvider()
        provider.initialize({"name": "qwen3_asr"})
        provider.load_model()

        audio = tmp_path / "a.wav"
        audio.write_text("dummy")
        result = provider.infer({"audio": str(audio), "language": "zh"})

        assert result["language"] == "zh"
        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["text"] == "你好世界"
        assert seg["start"] == pytest.approx(0.1)
        assert seg["end"] == pytest.approx(0.9)
        assert len(seg["words"]) == 2
        assert seg["words"][0]["start"] == pytest.approx(0.1)
        assert seg["words"][1]["end"] == pytest.approx(0.9)
        assert seg["confidence"] == 0.95
