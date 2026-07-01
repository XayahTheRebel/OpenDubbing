"""Prosody engine: annotate emotion, speech rate and pauses."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class ProsodyEngine(Engine):
    """Annotate prosody metadata on timeline sentences and words."""

    name = "prosody"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _detect_emotion(text: str) -> str:
        text = text.strip()
        if text.endswith("!"):
            return "surprise"
        if text.endswith("？") or text.endswith("?"):
            return "question"
        if "…" in text or "..." in text:
            return "calm"
        return "neutral"

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        sentences = timeline.sentences
        for i, sentence in enumerate(sentences):
            sentence.emotion = self._detect_emotion(sentence.translation or sentence.text)
            if i + 1 < len(sentences):
                sentence.pause = max(0.0, sentences[i + 1].start - sentence.end)
            else:
                sentence.pause = 0.0

            words = sentence.words
            for j, word in enumerate(words):
                if j + 1 < len(words):
                    word.pause = max(0.0, words[j + 1].start - word.end)
                else:
                    word.pause = 0.0

            sentence.metadata["prosody_done"] = True
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "prosody.jsonl"))

    def release(self) -> None:
        pass
