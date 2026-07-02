"""Tests for MuseTalkProvider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.face.musetalk import MuseTalkProvider


def _create_fake_musetalk_root(tmp_path: Any) -> Any:
    root = tmp_path / "MuseTalk"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "inference.py").write_text("dummy")
    (root / "models" / "musetalkV15").mkdir(parents=True)
    (root / "models" / "musetalkV15" / "unet.pth").write_text("dummy")
    (root / "models" / "musetalkV15" / "musetalk.json").write_text("dummy")
    (root / "models" / "whisper").mkdir(parents=True)
    (root / "models" / "whisper" / "pytorch_model.bin").write_text("dummy")
    (root / "models" / "sd-vae-ft-mse").mkdir(parents=True)
    (root / "models" / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin").write_text(
        "dummy"
    )
    (root / "models" / "dwpose").mkdir(parents=True)
    (root / "models" / "dwpose" / "dw-ll_ucoco_384.pth").write_text("dummy")
    (root / "models" / "syncnet").mkdir(parents=True)
    (root / "models" / "syncnet" / "latentsync_syncnet.pt").write_text("dummy")
    (root / "models" / "face-parse-bisent").mkdir(parents=True)
    (root / "models" / "face-parse-bisent" / "79999_iter.pth").write_text("dummy")
    return root


class TestMuseTalkProvider:
    def test_release_clears_checked_flag(self):
        provider = MuseTalkProvider()
        provider.initialize({"name": "musetalk"})
        provider._checked = True
        provider.release()
        assert provider._checked is False

    def test_load_model_raises_when_root_missing(self, tmp_path):
        provider = MuseTalkProvider()
        provider.initialize(
            {"name": "musetalk", "options": {"musetalk_root": str(tmp_path)}}
        )
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_when_weights_missing(self, tmp_path):
        root = tmp_path / "MuseTalk"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "inference.py").write_text("dummy")

        provider = MuseTalkProvider()
        provider.initialize(
            {"name": "musetalk", "options": {"musetalk_root": str(root)}}
        )
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_load_model_raises_when_conda_env_missing(self, tmp_path):
        root = _create_fake_musetalk_root(tmp_path)

        provider = MuseTalkProvider()
        provider.initialize(
            {
                "name": "musetalk",
                "options": {
                    "musetalk_root": str(root),
                    "conda_env": "MuseTalkMissingEnv",
                },
            }
        )

        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    @patch("opendubbing.providers.face.musetalk.shutil.rmtree")
    @patch("opendubbing.providers.face.musetalk.tempfile.mkdtemp")
    @patch("opendubbing.providers.face.musetalk.subprocess.run")
    def test_infer_builds_expected_command(
        self, mock_run, mock_mkdtemp, mock_rmtree, tmp_path
    ):
        root = _create_fake_musetalk_root(tmp_path)

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "results" / "v15").mkdir(parents=True)
        (task_dir / "results" / "v15" / "out.mp4").write_text("dummy")
        mock_mkdtemp.return_value = str(task_dir)

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "ok"
            result.stderr = ""
            return result

        mock_run.side_effect = fake_run

        provider = MuseTalkProvider()
        provider.initialize(
            {
                "name": "musetalk",
                "options": {
                    "musetalk_root": str(root),
                    "conda_env": "MuseTalk",
                    "version": "v15",
                    "vae_type": "sd-vae-ft-mse",
                },
            }
        )
        provider._checked = True

        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        video.write_text("dummy")
        audio.write_text("dummy")

        result = provider.infer(
            {"video": str(video), "audio": str(audio), "out_path": str(out)}
        )

        assert result["path"] == str(out)
        assert out.exists()
        mock_run.assert_called()
        conda_call = mock_run.call_args_list[-1][0][0]
        assert "conda" in conda_call
        assert "scripts.inference" in conda_call
        mock_rmtree.assert_called_once()
