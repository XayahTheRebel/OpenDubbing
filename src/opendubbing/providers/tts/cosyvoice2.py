"""CosyVoice2 TTS provider.

CosyVoice2 is executed inside a dedicated Conda environment to avoid dependency
conflicts with OpenDubbing (it pins older torch/transformers versions).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class CosyVoice2Provider(Provider):
    """Text-to-speech using CosyVoice2 via a dedicated Conda environment."""

    name = "cosyvoice2"
    kind = "tts"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model") or r"C:\CosyVoice\pretrained_models\CosyVoice2-0.5B"
        self.options = config.get("options", {})
        self.conda_env = self.options.get("conda_env", "CosyVoice")
        self.sample_rate = self.options.get("sample_rate", 22050)
        self._checked = False

    def _run_conda(
        self, args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the CosyVoice Conda environment."""
        cmd = ["conda", "run", "-n", self.conda_env, "--no-capture-output", *args]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )

    def load_model(self) -> None:
        if self._checked:
            return

        model_dir = Path(self.model)
        if not (model_dir / "cosyvoice2.yaml").exists():
            raise ProviderModelLoadError(
                f"CosyVoice2 model dir not found or incomplete: {model_dir}"
            )

        probe = self._run_conda(
            ["python", "-c", "import torch; print(torch.__version__)"]
        )
        if probe.returncode != 0:
            raise ProviderModelLoadError(
                f"CosyVoice Conda env '{self.conda_env}' is not ready:\n{probe.stderr}"
            )

        self._checked = True

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self._checked:
            self.load_model()

        text = inputs["text"]
        out_path = Path(inputs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        script = Path(__file__).resolve().parents[4] / "scripts" / "cosyvoice2_infer.py"
        cmd = [
            "python",
            str(script),
            "--model_dir",
            str(self.model),
            "--text",
            text,
            "--out_path",
            str(out_path),
            "--speech_rate",
            str(inputs.get("speech_rate", 1.0)),
            "--sample_rate",
            str(self.sample_rate),
        ]
        reference_audio = inputs.get("reference_audio")
        reference_text = inputs.get("reference_text", "")
        if reference_audio:
            cmd.extend(["--reference_audio", str(reference_audio)])
            cmd.extend(["--reference_text", str(reference_text)])

        proc = self._run_conda(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CosyVoice2 inference failed:\n{proc.stderr}\n{proc.stdout}"
            )

        import soundfile as sf

        info = sf.info(out_path)
        duration = info.duration
        return {"duration": duration, "path": str(out_path)}

    def release(self) -> None:
        self._checked = False
