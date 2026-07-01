"""Timeline data model and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Phoneme:
    """A single phoneme inside a word."""

    symbol: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0


@dataclass
class Word:
    """A single word with timing and optional phonemes."""

    text: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    pause: float = 0.0
    speech_rate: float = 1.0
    confidence: float = 1.0
    phonemes: list[Phoneme] = field(default_factory=list)


@dataclass
class Sentence:
    """A sentence containing words and prosody metadata."""

    id: str = ""
    text: str = ""
    translation: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    pause: float = 0.0
    speech_rate: float = 1.0
    emotion: str = "neutral"
    confidence: float = 1.0
    words: list[Word] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Timeline:
    """The central time-axis of the dubbing pipeline."""

    version: str = "1.0"
    source_language: str = ""
    target_language: str = ""
    sentences: list[Sentence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize timeline to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Timeline:
        """Deserialize timeline from a dictionary."""
        sentences_data = data.pop("sentences", [])
        sentences = []
        for sentence_data in sentences_data:
            words_data = sentence_data.pop("words", [])
            words = []
            for word_data in words_data:
                phonemes_data = word_data.pop("phonemes", [])
                phonemes = [Phoneme(**p) for p in phonemes_data]
                words.append(Word(phonemes=phonemes, **word_data))
            sentences.append(Sentence(words=words, **sentence_data))
        return cls(
            version=data.get("version", "1.0"),
            source_language=data.get("source_language", ""),
            target_language=data.get("target_language", ""),
            sentences=sentences,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save timeline as JSONL: one sentence per line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": self.version,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "metadata": self.metadata,
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for sentence in self.sentences:
                f.write(json.dumps(asdict(sentence), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> Timeline:
        """Load timeline from JSONL."""
        path = Path(path)
        sentences = []
        version = "1.0"
        source_language = ""
        target_language = ""
        metadata: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if index == 0 and "sentences" not in data and "metadata" in data:
                    version = data.get("version", "1.0")
                    source_language = data.get("source_language", "")
                    target_language = data.get("target_language", "")
                    metadata = data.get("metadata", {})
                    continue
                words_data = data.pop("words", [])
                words = []
                for word_data in words_data:
                    phonemes_data = word_data.pop("phonemes", [])
                    phonemes = [Phoneme(**p) for p in phonemes_data]
                    words.append(Word(phonemes=phonemes, **word_data))
                sentences.append(Sentence(words=words, **data))
        return cls(
            version=version,
            source_language=source_language,
            target_language=target_language,
            sentences=sentences,
            metadata=metadata,
        )

    def append(self, sentence: Sentence) -> None:
        """Append a sentence to the timeline."""
        self.sentences.append(sentence)

    def update_sentence(self, sentence_id: str, **kwargs: Any) -> None:
        """Update fields of a sentence by id."""
        for sentence in self.sentences:
            if sentence.id == sentence_id:
                for key, value in kwargs.items():
                    setattr(sentence, key, value)
                return
        raise KeyError(f"Sentence {sentence_id} not found in timeline")

    def __len__(self) -> int:
        return len(self.sentences)
