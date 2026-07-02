"""TTS engine: synthesize dubbed audio segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ffmpeg

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class TTSEngine(Engine):
    """Synthesize speech from translated text using a TTS provider."""

    name = "tts"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get("tts", {})
        self.provider_name = self.provider_config.get("name")
        self.registry = config.get("registry", ProviderRegistry())

    def _extract_reference(
        self, audio_path: Path, start: float, end: float, out_path: Path
    ) -> Path:
        """Extract a reference audio slice for zero-shot voice cloning."""
        duration = max(0.0, end - start)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stream = (
            ffmpeg.input(str(audio_path), ss=start, t=duration)
            .output(str(out_path), ar=16000, ac=1)
            .overwrite_output()
        )
        try:
            ffmpeg.run(stream, quiet=True)
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"extract_reference failed at {start}-{end}: {exc.stderr}"
            ) from exc
        return out_path

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.provider_name:
            raise ValueError("TTS provider name is required")

        input_audio = timeline.metadata.get("input_audio")
        input_audio_abs = (
            workspace.root / input_audio
            if input_audio and not Path(input_audio).is_absolute()
            else Path(input_audio) if input_audio else None
        )

        provider = self.registry.build(
            "tts", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            for sentence in timeline.sentences:
                if not sentence.translation:
                    continue
                out = workspace.path_for("tts", f"{sentence.id}.wav")
                infer_inputs: dict[str, Any] = {
                    "text": sentence.translation,
                    "language": timeline.target_language,
                    "speech_rate": sentence.speech_rate,
                    "emotion": sentence.emotion,
                    "out_path": str(out),
                }
                if input_audio_abs and input_audio_abs.exists():
                    ref = workspace.path_for("tts", f"ref_{sentence.id}.wav")
                    self._extract_reference(
                        input_audio_abs, sentence.start, sentence.end, ref
                    )
                    infer_inputs["reference_audio"] = str(ref)
                    infer_inputs["reference_text"] = sentence.text
                result = provider.infer(infer_inputs)
                sentence.metadata["tts_audio"] = str(workspace.relative(out))
                sentence.metadata["tts_duration"] = result.get("duration", 0.0)
        finally:
            provider.release()
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "tts.jsonl"))

    def release(self) -> None:
        pass
