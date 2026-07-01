import numpy as np
import pytest

from opendubbing.utils import media


class TestMedia:
    def test_write_read_audio(self, tmp_path):
        sr = 16000
        samples = np.zeros(sr * 2, dtype=np.float32)
        path = tmp_path / "silence.wav"
        media.write_audio(samples, sr, path)
        read, read_sr = media.read_audio(path)
        assert read_sr == sr
        assert len(read) == len(samples)

    def test_probe_duration(self, tmp_path):
        sr = 16000
        samples = np.zeros(sr * 2, dtype=np.float32)
        path = tmp_path / "silence.wav"
        media.write_audio(samples, sr, path)
        duration = media.probe_duration(path)
        assert abs(duration - 2.0) < 0.1

    def test_concat_audio(self, tmp_path):
        sr = 16000
        path1 = tmp_path / "a.wav"
        path2 = tmp_path / "b.wav"
        media.write_audio(np.zeros(sr * 2, dtype=np.float32), sr, path1)
        media.write_audio(np.zeros(sr * 2, dtype=np.float32), sr, path2)
        out = tmp_path / "out.wav"
        media.concat_audio([path1, path2], out)
        read, _ = media.read_audio(out)
        assert len(read) == sr * 4

    def test_segment_audio(self, tmp_path):
        pytest.skip("segment_audio requires a real audio fixture")

    def test_mux_audio_video(self, tmp_path):
        pytest.skip("mux_audio_video requires a real video fixture")

    def test_extract_audio(self, tmp_path):
        pytest.skip("extract_audio requires a real video fixture")
