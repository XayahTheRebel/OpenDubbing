"""Hallo3 face animation provider.

This provider runs the official Hallo3 inference inside the WSL Ubuntu
environment (where the heavy dependencies and pretrained models are installed)
and returns the generated lip-sync video.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class Hallo3Provider(Provider):
    """Real Hallo3 inference via WSL subprocess.

    Expects the following WSL setup:
        - Ubuntu distribution installed and reachable as ``Ubuntu``
        - Hallo3 cloned to ``/root/hallo3``
        - Conda environment ``hallo3`` with all Hallo3 dependencies
        - Pretrained models under ``/root/hallo3/pretrained_models``
        - ``nvidia-cuda-toolkit`` installed and ``CUDA_HOME=/usr`` works
    """

    name = "hallo3"
    kind = "face"

    _WSL_DISTRO = "Ubuntu"
    _WSL_USER = "root"
    _HALLO3_ROOT = "/root/hallo3"
    _CONDA_ENV = "hallo3"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "")
        self.options = config.get("options", {})
        self.prompt = self.options.get("prompt", "A person speaking naturally")
        self._checked = False

    def _wsl(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a shell command inside WSL."""
        return subprocess.run(
            ["wsl", "-d", self._WSL_DISTRO, "-u", self._WSL_USER, "-e", "bash", "-c", cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=check,
        )

    def _to_wsl_path(self, path: str) -> str:
        """Convert a Windows absolute path to a WSL ``/mnt/X/...`` path."""
        p = Path(path).resolve()
        if p.drive:
            drive = p.drive.replace(":", "").lower()
            posix = p.as_posix().split(":", 1)[1]
            return f"/mnt/{drive}{posix}"
        return str(p).replace("\\", "/")

    def load_model(self) -> None:
        """Verify that Hallo3 environment and models are present in WSL."""
        if self._checked:
            return

        probe = self._wsl(
            f"test -d {self._HALLO3_ROOT} && "
            f"test -f {self._HALLO3_ROOT}/hallo3/sample_video.py && "
            f"test -d {self._HALLO3_ROOT}/pretrained_models && "
            f"source /root/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self._CONDA_ENV} && "
            f"python -c 'import torch; print(torch.cuda.is_available())'",
            check=False,
        )
        if probe.returncode != 0:
            raise ProviderModelLoadError(
                "Hallo3 is not ready in WSL. Check that /root/hallo3, "
                "pretrained_models, and the conda env exist.\n" + probe.stderr
            )
        self._checked = True

    def _extract_reference_image(self, video_wsl: str, tmp_dir: str) -> str:
        """Extract the first frame of the video as a reference image."""
        ref_path = f"{tmp_dir}/ref.jpg"
        self._wsl(
            f"ffmpeg -y -i '{video_wsl}' -ss 00:00:00 -vframes 1 -q:v 2 '{ref_path}'"
        )
        if not self._wsl(f"test -f '{ref_path}'", check=False).returncode == 0:
            raise RuntimeError(f"Failed to extract reference image to {ref_path}")
        return ref_path

    def _find_output_video(self, output_dir: str) -> str:
        """Locate the generated ``*_with_audio.mp4`` file."""
        find_cmd = f"find '{output_dir}' -type f -name '*_with_audio.mp4' | head -n 1"
        result = self._wsl(find_cmd, check=False)
        path = result.stdout.strip()
        if not path:
            raise RuntimeError(
                f"Hallo3 did not produce a '*_with_audio.mp4' file under {output_dir}"
            )
        return path

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self._checked:
            self.load_model()

        video = str(inputs["video"])
        audio = str(inputs["audio"])
        out_path = Path(inputs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        video_wsl = self._to_wsl_path(video)
        audio_wsl = self._to_wsl_path(str(out_path.parent / "dub_audio_for_hallo3.wav"))
        out_path_wsl = self._to_wsl_path(str(out_path))

        # Hallo3 expects 16kHz mono WAV; convert the dub audio if necessary.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio,
                "-ar",
                "16000",
                "-ac",
                "1",
                out_path.parent / "dub_audio_for_hallo3.wav",
            ],
            check=True,
            capture_output=True,
        )

        tmp_dir = f"/tmp/hallo3_run_{uuid.uuid4().hex}"
        self._wsl(f"mkdir -p '{tmp_dir}'")

        try:
            ref_image = self._extract_reference_image(video_wsl, tmp_dir)
            input_txt = f"{tmp_dir}/input.txt"
            self._wsl(
                f"printf '%s@@%s@@%s\\n' '{self.prompt}' '{ref_image}' '{audio_wsl}' > '{input_txt}'"
            )

            output_dir = f"{tmp_dir}/out"
            self._wsl(f"mkdir -p '{output_dir}'")

            inference_cmd = (
                f"cd {self._HALLO3_ROOT} && "
                f"source /root/miniconda3/etc/profile.d/conda.sh && "
                f"conda activate {self._CONDA_ENV} && "
                f"export CUDA_HOME=/usr && "
                f"python hallo3/sample_video.py "
                f"--base configs/inference.yaml "
                f"--image2video "
                f"--input-type txt "
                f"--input-file '{input_txt}' "
                f"--output-dir '{output_dir}' "
                f"--sampling-fps 25"
            )
            result = self._wsl(inference_cmd, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Hallo3 inference failed:\n{result.stderr}\n{result.stdout}"
                )

            generated = self._find_output_video(output_dir)
            self._wsl(f"cp '{generated}' '{out_path_wsl}'")
        finally:
            self._wsl(f"rm -rf '{tmp_dir}'", check=False)
            (out_path.parent / "dub_audio_for_hallo3.wav").unlink(missing_ok=True)

        if not out_path.exists():
            raise RuntimeError(f"Hallo3 output was not copied to {out_path}")

        return {"path": str(out_path)}

    def release(self) -> None:
        self._checked = False
        import gc

        gc.collect()
