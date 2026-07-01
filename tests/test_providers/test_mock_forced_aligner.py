"""Tests for MockForcedAlignerProvider."""

from __future__ import annotations

import pytest

from opendubbing.providers.forced_alignment.mock_forced_aligner import (
    MockForcedAlignerProvider,
)


class TestMockForcedAlignerProvider:
    def test_infer_spreads_string_words_evenly(self):
        provider = MockForcedAlignerProvider()
        provider.initialize({})

        result = provider.infer(
            {
                "audio": "ignored.wav",
                "text": "hello world",
                "words": ["hello", "world"],
                "start": 0.0,
                "end": 2.0,
            }
        )

        assert result["confidence"] == 1.0
        words = result["words"]
        assert len(words) == 2
        assert words[0]["text"] == "hello"
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[0]["end"] == pytest.approx(1.0)
        assert words[1]["text"] == "world"
        assert words[1]["start"] == pytest.approx(1.0)
        assert words[1]["end"] == pytest.approx(2.0)
        assert words[0]["phonemes"] == []

    def test_infer_preserves_dict_words(self):
        provider = MockForcedAlignerProvider()
        provider.initialize({})

        result = provider.infer(
            {
                "audio": "ignored.wav",
                "text": "hello world",
                "words": [
                    {"text": "hello", "start": 0.1, "end": 0.4},
                    {"text": "world", "start": 0.5, "end": 0.9},
                ],
            }
        )

        words = result["words"]
        assert words[0]["start"] == pytest.approx(0.1)
        assert words[0]["end"] == pytest.approx(0.4)
        assert words[1]["start"] == pytest.approx(0.5)
        assert words[1]["end"] == pytest.approx(0.9)
