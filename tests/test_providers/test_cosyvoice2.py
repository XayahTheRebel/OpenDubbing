"""Tests for CosyVoice2Provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.tts.cosyvoice2 import CosyVoice2Provider


class TestCosyVoice2Provider:
    def test_load_model_raises_when_model_dir_missing(self, tmp_path):
        provider = CosyVoice2Provider()
        provider.initialize(
            {"name": "cosyvoice2", "model": str(tmp_path / "missing")}
        )
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_when_conda_env_missing(self, tmp_path):
        (tmp_path / "cosyvoice2.yaml").write_text("dummy")
        provider = CosyVoice2Provider()
        provider.initialize(
            {
                "name": "cosyvoice2",
                "model": str(tmp_path),
                "options": {"conda_env": "CosyVoiceMissing"},
            }
        )
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_release_clears_checked_flag(self):
        provider = CosyVoice2Provider()
        provider.initialize({"name": "cosyvoice2"})
        provider._checked = True
        provider.release()
        assert provider._checked is False

    def test_infer_runs_subprocess_and_returns_duration(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice2.yaml").write_text("dummy")
        out = tmp_path / "out.wav"

        # Generate a silent wav so soundfile.info can read duration.
        samples = np.zeros(22050, dtype=np.float32)
        sf.write(out, samples, 22050)

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = f"Wrote {out}, duration=1.000s"
            result.stderr = ""
            return result

        provider = CosyVoice2Provider()
        provider.initialize(
            {
                "name": "cosyvoice2",
                "model": str(model_dir),
                "options": {"conda_env": "CosyVoice"},
            }
        )
        provider._checked = True

        with patch(
            "opendubbing.providers.tts.cosyvoice2.subprocess.run",
            side_effect=fake_run,
        ):
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

    def test_infer_passes_reference_audio(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "cosyvoice2.yaml").write_text("dummy")
        out = tmp_path / "out.wav"
        ref = tmp_path / "ref.wav"
        sf.write(ref, np.zeros(16000, dtype=np.float32), 16000)
        sf.write(out, np.zeros(22050, dtype=np.float32), 22050)

        captured_cmd = None

        def fake_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = "ok"
            result.stderr = ""
            return result

        provider = CosyVoice2Provider()
        provider.initialize(
            {
                "name": "cosyvoice2",
                "model": str(model_dir),
                "options": {"conda_env": "CosyVoice"},
            }
        )
        provider._checked = True

        with patch(
            "opendubbing.providers.tts.cosyvoice2.subprocess.run",
            side_effect=fake_run,
        ):
            provider.infer(
                {
                    "text": "你好",
                    "language": "zh",
                    "speech_rate": 1.0,
                    "emotion": "neutral",
                    "out_path": str(out),
                    "reference_audio": str(ref),
                }
            )

        assert captured_cmd is not None
        assert "--reference_audio" in captured_cmd
        assert str(ref) in captured_cmd
