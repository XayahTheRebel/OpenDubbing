"""Tests for DeepFilterNet3Provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.noise_reduction.deep_filter_net3 import (
    DeepFilterNet3Provider,
)


class TestDeepFilterNet3Provider:
    def test_load_model_raises_when_df_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "df", None)
        provider = DeepFilterNet3Provider()
        provider.initialize({"name": "deep_filter_net3"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_writes_enhanced_audio(self, monkeypatch, tmp_path, silent_wav):
        audio = silent_wav(duration=1.0)
        out_path = tmp_path / "enhanced.wav"

        fake_model = object()
        fake_state = object()
        fake_init_df = MagicMock(return_value=(fake_model, fake_state, None))
        fake_enhance = MagicMock()

        fake_df = SimpleNamespace(init_df=fake_init_df, enhance=fake_enhance)
        monkeypatch.setitem(__import__("sys").modules, "df", fake_df)

        provider = DeepFilterNet3Provider()
        provider.initialize({"name": "deep_filter_net3"})
        provider.load_model()

        result = provider.infer({"audio": audio, "out_path": str(out_path)})

        assert result["enhanced"] == str(out_path)
        fake_enhance.assert_called_once()
        _, args, kwargs = fake_enhance.mock_calls[0]
        assert kwargs.get("output_file") == str(out_path)
