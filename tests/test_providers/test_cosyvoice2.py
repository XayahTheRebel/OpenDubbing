"""Tests for CosyVoice2Provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.tts.cosyvoice2 import CosyVoice2Provider


class TestCosyVoice2Provider:
    def test_load_model_raises_when_cosyvoice_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "cosyvoice", None)
        provider = CosyVoice2Provider()
        provider.initialize({"name": "cosyvoice2"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_sft_writes_wav(self, monkeypatch, tmp_path):
        fake_speech = np.zeros(22050, dtype=np.float32)
        fake_tensor = MagicMock()
        fake_tensor.numpy.return_value = fake_speech.reshape(1, -1)
        fake_model = MagicMock()
        fake_model.inference_sft.return_value = [{"tts_speech": fake_tensor}]

        fake_cosyvoice = SimpleNamespace(
            cli=SimpleNamespace(cosyvoice=SimpleNamespace(CosyVoice2=MagicMock(return_value=fake_model)))
        )
        monkeypatch.setitem(__import__("sys").modules, "cosyvoice", fake_cosyvoice)
        monkeypatch.setitem(
            __import__("sys").modules,
            "cosyvoice.cli",
            fake_cosyvoice.cli,
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "cosyvoice.cli.cosyvoice",
            fake_cosyvoice.cli.cosyvoice,
        )

        out = tmp_path / "out.wav"
        provider = CosyVoice2Provider()
        provider.initialize({"name": "cosyvoice2"})
        provider.load_model()

        result = provider.infer(
            {
                "text": "你好",
                "language": "zh",
                "speech_rate": 1.0,
                "emotion": "neutral",
                "out_path": str(out),
            }
        )

        assert result["path"] == str(out)
        assert result["duration"] == pytest.approx(1.0)
        fake_model.inference_sft.assert_called_once_with("你好")

    def test_infer_zero_shot_with_reference(self, monkeypatch, tmp_path, silent_wav):
        ref = silent_wav(duration=1.0)
        fake_speech = np.zeros(22050, dtype=np.float32)
        fake_tensor = MagicMock()
        fake_tensor.numpy.return_value = fake_speech.reshape(1, -1)
        fake_model = MagicMock()
        fake_model.inference_zero_shot.return_value = [{"tts_speech": fake_tensor}]

        fake_cosyvoice = SimpleNamespace(
            cli=SimpleNamespace(cosyvoice=SimpleNamespace(CosyVoice2=MagicMock(return_value=fake_model)))
        )
        monkeypatch.setitem(__import__("sys").modules, "cosyvoice", fake_cosyvoice)
        monkeypatch.setitem(
            __import__("sys").modules,
            "cosyvoice.cli",
            fake_cosyvoice.cli,
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "cosyvoice.cli.cosyvoice",
            fake_cosyvoice.cli.cosyvoice,
        )

        out = tmp_path / "out.wav"
        provider = CosyVoice2Provider()
        provider.initialize({"name": "cosyvoice2"})
        provider.load_model()

        result = provider.infer(
            {
                "text": "你好",
                "language": "zh",
                "speech_rate": 1.0,
                "emotion": "neutral",
                "out_path": str(out),
                "reference_audio": ref,
            }
        )

        assert result["path"] == str(out)
        fake_model.inference_zero_shot.assert_called_once()
