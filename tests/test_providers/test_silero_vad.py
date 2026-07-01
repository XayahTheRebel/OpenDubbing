"""Tests for SileroVADProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.vad.silero_vad import SileroVADProvider


class TestSileroVADProvider:
    def test_load_model_raises_when_silero_vad_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "silero_vad", None)
        provider = SileroVADProvider()
        provider.initialize({"name": "silero_vad"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_converts_samples_to_seconds(self, monkeypatch, silent_wav):
        audio = silent_wav(duration=1.0)

        fake_tensor_fn = MagicMock(return_value="tensor")
        fake_model = (object(), SimpleNamespace(tensor=fake_tensor_fn))
        fake_load = MagicMock(return_value=fake_model)
        fake_get_ts = MagicMock(
            return_value=[
                {"start": 1600, "end": 8000},
                {"start": 9600, "end": 14400},
            ]
        )

        fake_silero = SimpleNamespace(
            load_silero_vad=fake_load, get_speech_timestamps=fake_get_ts
        )
        monkeypatch.setitem(__import__("sys").modules, "silero_vad", fake_silero)

        provider = SileroVADProvider()
        provider.initialize({"name": "silero_vad"})
        provider.load_model()

        result = provider.infer(
            {"audio": audio, "threshold": 0.5, "min_silence_ms": 300}
        )

        segments = result["segments"]
        assert len(segments) == 2
        assert segments[0]["start"] == pytest.approx(0.1)
        assert segments[0]["end"] == pytest.approx(0.5)
        assert segments[1]["start"] == pytest.approx(0.6)
        assert segments[1]["end"] == pytest.approx(0.9)
        fake_get_ts.assert_called_once()
