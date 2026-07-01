"""Tests for DemucsProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.audio_separation.demucs import DemucsProvider


class TestDemucsProvider:
    def test_load_model_raises_when_deps_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        provider = DemucsProvider()
        provider.initialize({"name": "demucs"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_writes_stems_and_returns_paths(self, monkeypatch, tmp_path, silent_wav):
        audio = silent_wav(duration=1.0)
        out_dir = tmp_path / "stems"

        # Build a fake tensor chain that survives .repeat, .to, indexing, .mean, .cpu, .numpy
        fake_source = np.zeros(16000, dtype=np.float32)
        stem_tensor = MagicMock()
        stem_tensor.mean.return_value.cpu.return_value.numpy.return_value = fake_source

        fake_sources = MagicMock()
        fake_sources.__getitem__ = lambda _self, idx: stem_tensor

        fake_model = MagicMock()
        fake_model.sources = ["vocals", "drums"]
        fake_model.forward.return_value = [fake_sources]
        fake_model.to.return_value = fake_model

        fake_get_model = MagicMock(return_value=fake_model)

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.no_grad.return_value.__enter__ = MagicMock()
        fake_torch.no_grad.return_value.__exit__ = MagicMock()

        fake_wav = MagicMock()
        fake_wav.shape = (1, 16000)
        fake_wav.repeat.return_value = fake_wav
        fake_wav.to.return_value = fake_wav
        fake_wav.__getitem__ = lambda _self, idx: fake_wav

        fake_torchaudio = SimpleNamespace(load=MagicMock(return_value=(fake_wav, 16000)))

        fake_demucs = SimpleNamespace(pretrained=SimpleNamespace(get_model=fake_get_model))

        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        monkeypatch.setitem(__import__("sys").modules, "torchaudio", fake_torchaudio)
        monkeypatch.setitem(__import__("sys").modules, "demucs", MagicMock())
        monkeypatch.setitem(
            __import__("sys").modules, "demucs.pretrained", fake_demucs.pretrained
        )

        provider = DemucsProvider()
        provider.initialize({"name": "demucs", "options": {"device": "cpu"}})
        provider.load_model()

        with patch(
            "opendubbing.providers.audio_separation.demucs.media.write_audio"
        ) as mock_write:
            result = provider.infer({"audio": audio, "out_dir": str(out_dir)})

        assert "stems" in result
        assert "vocals" in result["stems"]
        assert "drums" in result["stems"]
        assert mock_write.call_count == 2
