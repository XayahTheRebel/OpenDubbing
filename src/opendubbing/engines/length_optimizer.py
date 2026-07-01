"""Length optimizer: adjust translated text to fit target duration."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class LengthOptimizer(Engine):
    """Optimize translation length to match original speech duration."""

    name = "length_optimizer"

    _RATES = {
        "zh": 0.28,
        "en": 0.18,
    }

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get(
            "translation", {}
        )
        self.target_language = config.get("target_language", "zh")
        self.target_language = (
            self.provider_config.get("options", {}).get("target_language")
            or self.target_language
        )

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        rate = self._RATES.get(self.target_language, 0.25)
        for sentence in timeline.sentences:
            if not sentence.translation:
                continue
            target_duration = sentence.duration
            estimated = len(sentence.translation) * rate
            ratio = estimated / target_duration if target_duration > 0 else 1.0
            if estimated > target_duration * 1.15:
                sentence.speech_rate = max(0.8, min(1.3, ratio))
                sentence.metadata["length_warning"] = "too_long"
                sentence.metadata["length_estimated"] = estimated
                sentence.metadata["length_target"] = target_duration
            else:
                sentence.speech_rate = 1.0
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "length_optimized.jsonl"))

    def release(self) -> None:
        pass
