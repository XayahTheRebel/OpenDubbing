"""MuseTalk 1.5 face-animation provider for OpenDubbing.

MuseTalk is executed inside a dedicated Conda environment to avoid dependency
conflicts with OpenDubbing. It takes the original video and the dubbed audio,
performs latent-space lip-sync inpainting, and writes the result to the
requested output path.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class MuseTalkProvider(Provider):
    """MuseTalk 1.5 inference via a dedicated Conda environment."""

    name = "musetalk"
    kind = "face"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})

        self.musetalk_root = Path(
            self.options.get("musetalk_root", r"C:\MuseTalk")
        ).resolve()
        self.conda_env = self.options.get("conda_env", "MuseTalk")
        self.ffmpeg_path = self.options.get("ffmpeg_path", "")
        self.gpu_id = self.options.get("gpu_id", 0)
        self.version = self.options.get("version", "v15")
        self.vae_type = self.options.get("vae_type", "sd-vae-ft-mse")
        self.batch_size = self.options.get("batch_size", 8)
        self.use_float16 = self.options.get("use_float16", False)
        self.bbox_shift = self.options.get("bbox_shift", 0)
        self.extra_margin = self.options.get("extra_margin", 10)
        self.parsing_mode = self.options.get("parsing_mode", "jaw")
        self.left_cheek_width = self.options.get("left_cheek_width", 90)
        self.right_cheek_width = self.options.get("right_cheek_width", 90)
        self.fps = self.options.get("fps", 25)

        self._checked = False

    def _run_conda(
        self, args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the MuseTalk Conda environment."""
        cmd = ["conda", "run", "-n", self.conda_env, "--no-capture-output", *args]
        return subprocess.run(
            cmd,
            cwd=self.musetalk_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )

    def load_model(self) -> None:
        """Verify MuseTalk repo, Conda env, and model weights are ready."""
        if self._checked:
            return

        inference_script = self.musetalk_root / "scripts" / "inference.py"
        if not inference_script.exists():
            raise ProviderModelLoadError(
                f"MuseTalk inference script not found: {inference_script}"
            )

        if self.version == "v15":
            unet_dir = self.musetalk_root / "models" / "musetalkV15"
            unet_model = unet_dir / "unet.pth"
            unet_config = unet_dir / "musetalk.json"
        else:
            unet_dir = self.musetalk_root / "models" / "musetalk"
            unet_model = unet_dir / "pytorch_model.bin"
            unet_config = unet_dir / "musetalk.json"

        required = [
            unet_model,
            unet_config,
            self.musetalk_root / "models" / "whisper" / "pytorch_model.bin",
            self.musetalk_root / "models" / self.vae_type / "diffusion_pytorch_model.bin",
            self.musetalk_root / "models" / "dwpose" / "dw-ll_ucoco_384.pth",
            self.musetalk_root / "models" / "syncnet" / "latentsync_syncnet.pt",
            self.musetalk_root / "models" / "face-parse-bisent" / "79999_iter.pth",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise ProviderModelLoadError(
                f"MuseTalk missing model files: {missing}"
            )

        probe = self._run_conda(["python", "-c", "import torch; print(torch.__version__)"])
        if probe.returncode != 0:
            raise ProviderModelLoadError(
                f"MuseTalk Conda env '{self.conda_env}' is not ready:\n{probe.stderr}"
            )

        self._checked = True

    def _to_wav(self, audio: str, out: Path) -> None:
        """Convert any audio to 16 kHz mono WAV for MuseTalk."""
        cmd = ["ffmpeg", "-y", "-i", audio, "-ar", "16000", "-ac", "1", str(out)]
        if self.ffmpeg_path:
            cmd[0] = str(Path(self.ffmpeg_path) / "ffmpeg.exe")
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio conversion failed: {result.stderr}")

    def _write_config(self, video: str, audio: str, task_dir: Path) -> Path:
        """Write MuseTalk inference YAML for a single task."""
        task: dict[str, Any] = {
            "video_path": str(Path(video).resolve().as_posix()),
            "audio_path": str(Path(audio).resolve().as_posix()),
        }
        if self.version != "v15":
            task["bbox_shift"] = self.bbox_shift

        cfg = {"task_0": task}
        cfg_path = task_dir / "infer.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return cfg_path

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run MuseTalk inference and copy the output to the requested path."""
        if not self._checked:
            self.load_model()

        video = str(inputs["video"])
        audio = str(inputs["audio"])
        out_path = Path(inputs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        task_dir = Path(tempfile.mkdtemp(prefix="musetalk_"))
        try:
            wav_path = task_dir / "audio_16k.wav"
            self._to_wav(audio, wav_path)

            cfg_path = self._write_config(video, str(wav_path), task_dir)
            result_dir = task_dir / "results"

            cmd = [
                "python",
                "-m",
                "scripts.inference",
                "--inference_config",
                str(cfg_path),
                "--result_dir",
                str(result_dir),
                "--unet_model_path",
                str(self._unet_model_path()),
                "--unet_config",
                str(self._unet_config_path()),
                "--vae_type",
                self.vae_type,
                "--version",
                self.version,
                "--batch_size",
                str(self.batch_size),
                "--fps",
                str(self.fps),
                "--output_vid_name",
                "out.mp4",
                "--extra_margin",
                str(self.extra_margin),
                "--parsing_mode",
                self.parsing_mode,
                "--left_cheek_width",
                str(self.left_cheek_width),
                "--right_cheek_width",
                str(self.right_cheek_width),
                "--gpu_id",
                str(self.gpu_id),
            ]
            if self.ffmpeg_path:
                cmd.extend(["--ffmpeg_path", self.ffmpeg_path])
            if self.use_float16:
                cmd.append("--use_float16")

            proc = self._run_conda(cmd)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"MuseTalk inference failed:\n{proc.stderr}\n{proc.stdout}"
                )

            generated = result_dir / self.version / "out.mp4"
            if not generated.exists():
                raise RuntimeError(
                    f"MuseTalk did not produce expected output: {generated}"
                )

            shutil.copy(str(generated), str(out_path))
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        if not out_path.exists():
            raise RuntimeError(f"MuseTalk output was not copied to {out_path}")

        return {"path": str(out_path)}

    def _unet_model_path(self) -> Path:
        if self.version == "v15":
            return self.musetalk_root / "models" / "musetalkV15" / "unet.pth"
        return self.musetalk_root / "models" / "musetalk" / "pytorch_model.bin"

    def _unet_config_path(self) -> Path:
        if self.version == "v15":
            return self.musetalk_root / "models" / "musetalkV15" / "musetalk.json"
        return self.musetalk_root / "models" / "musetalk" / "musetalk.json"

    def release(self) -> None:
        self._checked = False
