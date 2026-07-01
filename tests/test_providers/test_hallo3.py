"""Tests for Hallo3Provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.face.hallo3 import Hallo3Provider


class TestHallo3Provider:
    def test_load_model_raises_when_wsl_probe_fails(self, monkeypatch):
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "mock wsl failure"
        monkeypatch.setattr(
            "opendubbing.providers.face.hallo3.subprocess.run",
            MagicMock(return_value=fake_result),
        )

        provider = Hallo3Provider()
        provider.initialize({"name": "hallo3"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_release_clears_checked_flag(self):
        provider = Hallo3Provider()
        provider.initialize({})
        provider._checked = True
        provider.release()
        assert provider._checked is False
