"""Tests for MockTTSProvider."""

from __future__ import annotations

from pathlib import Path

from opendubbing.providers.tts.mock_tts import MockTTSProvider


class TestMockTTSProvider:
    def test_infer_writes_silent_wav(self, tmp_path):
        out = tmp_path / "out.wav"
        provider = MockTTSProvider()
        provider.initialize({"name": "mock_tts", "options": {"duration": 1.5}})
        provider.load_model()

        result = provider.infer({"out_path": str(out)})

        assert Path(out).exists()
        assert result["duration"] == 1.5
        assert result["path"] == str(out)
