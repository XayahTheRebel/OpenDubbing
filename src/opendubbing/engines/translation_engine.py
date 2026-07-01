"""Translation engine: translate timeline sentences."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class TranslationEngine(Engine):
    """Translate source text into target language using a translation provider."""

    name = "translation"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get(
            "translation", {}
        )
        self.provider_name = self.provider_config.get("name")
        self.target_language = config.get("target_language", "zh")
        self.target_language = (
            self.provider_config.get("options", {}).get("target_language")
            or self.target_language
        )
        self.registry = config.get("registry", ProviderRegistry())

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.provider_name:
            raise ValueError("Translation provider name is required")

        provider = self.registry.build(
            "translation", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            for sentence in timeline.sentences:
                if not sentence.text:
                    continue
                result = provider.infer(
                    {
                        "text": sentence.text,
                        "source_language": timeline.source_language or "auto",
                        "target_language": self.target_language,
                    }
                )
                sentence.translation = result.get("translation", "").strip()
        finally:
            provider.release()

        timeline.target_language = self.target_language
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("translation", "translation.jsonl"))

    def release(self) -> None:
        pass
