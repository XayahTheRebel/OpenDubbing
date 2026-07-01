"""Tests for EdgeTTSProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.tts.edge_tts import EdgeTTSProvider


class TestEdgeTTSProvider:
    def test_load_model_raises_when_edge_tts_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "edge_tts", None)
        provider = EdgeTTSProvider()
        provider.initialize({"name": "edge_tts"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_writes_audio(self, monkeypatch, tmp_path):
        out = tmp_path / "out.mp3"

        fake_communicate = MagicMock()
        fake_communicate.save = MagicMock()

        fake_edge_tts = SimpleNamespace(Communicate=MagicMock(return_value=fake_communicate))
        monkeypatch.setitem(__import__("sys").modules, "edge_tts", fake_edge_tts)

        provider = EdgeTTSProvider()
        provider.initialize({"name": "edge_tts", "options": {"voice": "test-voice"}})
        provider.load_model()

        with patch.object(
            provider, "_read_audio_with_fallback", return_value=(np.zeros(16000, dtype=np.float32), 16000)
        ), patch.object(provider, "_run_async") as mock_run_async:
            result = provider.infer(
                {
                    "text": "hello",
                    "language": "en",
                    "speech_rate": 1.5,
                    "out_path": str(out),
                }
            )

        assert result["path"] == str(out)
        assert result["duration"] == pytest.approx(1.0)
        fake_edge_tts.Communicate.assert_called_once_with("hello", "test-voice", rate="+50%")
        mock_run_async.assert_called_once()
