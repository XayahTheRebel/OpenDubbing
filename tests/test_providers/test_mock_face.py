"""Tests for MockFaceProvider."""

from __future__ import annotations

import ffmpeg
import numpy as np

from opendubbing.providers.face.mock_face import MockFaceProvider
from opendubbing.utils import media


class TestMockFaceProvider:
    def test_infer_muxes_audio_into_video(self, tmp_path):
        video = tmp_path / "video.mp4"
        audio = media.write_audio(
            np.zeros(16000 * 2, dtype=np.float32), 16000, tmp_path / "audio.wav"
        )
        out = tmp_path / "out.mp4"

        # Create a minimal blank video.
        (
            ffmpeg
            .input("testsrc=size=320x240:rate=1", f="lavfi", t=2)
            .output(str(video), vcodec="libx264", pix_fmt="yuv420p")
            .overwrite_output()
            .run(quiet=True)
        )

        provider = MockFaceProvider()
        provider.initialize({})
        provider.load_model()

        result = provider.infer(
            {"video": str(video), "audio": str(audio), "out_path": str(out)}
        )

        assert out.exists()
        assert result["path"] == str(out)
        assert result["fallback"] is True
