"""Shared fixtures for provider tests."""

from __future__ import annotations

import numpy as np
import pytest

from opendubbing.utils import media


@pytest.fixture
def silent_wav(tmp_path):
    """Return a path to a short silent WAV file."""

    def _make(duration: float = 1.0, sample_rate: int = 16000) -> str:
        path = tmp_path / f"silent_{duration}s.wav"
        samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
        media.write_audio(samples, sample_rate, path)
        return str(path)

    return _make
