"""Tests for WhisperASRProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.asr.whisper_asr import WhisperASRProvider


class TestWhisperASRProvider:
    def test_load_model_raises_when_whisper_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "whisper", None)
        provider = WhisperASRProvider()
        provider.initialize({"name": "whisper"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_returns_segments_and_words(self, monkeypatch, tmp_path):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "avg_logprob": -0.2,
                    "words": [
                        {"word": "hello ", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                }
            ],
        }

        fake_whisper = SimpleNamespace(load_model=MagicMock(return_value=fake_model))
        monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

        provider = WhisperASRProvider()
        provider.initialize({"name": "whisper", "model": "base"})
        provider.load_model()

        audio = tmp_path / "a.wav"
        audio.write_text("dummy")
        result = provider.infer({"audio": str(audio), "language": "auto"})

        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["text"] == "hello world"
        assert seg["confidence"] == -0.2
        assert len(seg["words"]) == 2
        assert seg["words"][0]["text"] == "hello"
        assert seg["words"][1]["text"] == "world"
        fake_whisper.load_model.assert_called_once_with("base")
