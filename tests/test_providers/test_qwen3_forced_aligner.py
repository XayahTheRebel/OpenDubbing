"""Tests for Qwen3ForcedAlignerProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.forced_alignment.qwen3_forced_aligner import (
    Qwen3ForcedAlignerProvider,
)


class TestQwen3ForcedAlignerProvider:
    def test_load_model_raises_when_funasr_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "funasr", None)
        provider = Qwen3ForcedAlignerProvider()
        provider.initialize({"name": "qwen3_forced_aligner"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_converts_ms_to_seconds(self, monkeypatch, tmp_path):
        fake_model = MagicMock()
        fake_model.generate.return_value = [
            {
                "tokens": ["hello", "world"],
                "timestamp": [[100, 400], [500, 900]],
                "confidence": 0.97,
            }
        ]
        fake_automodel = MagicMock(return_value=fake_model)

        fake_funasr = SimpleNamespace(AutoModel=fake_automodel)
        monkeypatch.setitem(__import__("sys").modules, "funasr", fake_funasr)

        provider = Qwen3ForcedAlignerProvider()
        provider.initialize({"name": "qwen3_forced_aligner"})
        provider.load_model()

        audio = tmp_path / "a.wav"
        audio.write_text("dummy")
        result = provider.infer(
            {"audio": str(audio), "text": "hello world", "words": ["hello", "world"]}
        )

        assert result["confidence"] == 0.97
        words = result["words"]
        assert len(words) == 2
        assert words[0]["text"] == "hello"
        assert words[0]["start"] == pytest.approx(0.1)
        assert words[0]["end"] == pytest.approx(0.4)
        assert words[1]["start"] == pytest.approx(0.5)
        assert words[1]["end"] == pytest.approx(0.9)
        assert words[0]["phonemes"] == []
