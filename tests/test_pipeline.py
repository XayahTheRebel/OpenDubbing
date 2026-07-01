import pytest

from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.pipeline.cache import PipelineCache


def _make_test_video(tmp_path):
    """Create a short test video with silence audio."""
    import ffmpeg
    import numpy as np

    from opendubbing.utils import media

    audio = media.write_audio(
        np.zeros(16000 * 2, dtype=np.float32), 16000, tmp_path / "silence.wav"
    )
    blank = tmp_path / "blank.mp4"
    (
        ffmpeg.input("testsrc=size=320x240:rate=1", f="lavfi", t=2)
        .output(str(blank), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    video = tmp_path / "video.mp4"
    media.mux_audio_video(blank, audio, video)
    return video


class TestPipelineCache:
    def test_cache_miss(self, tmp_path):
        workspace = Workspace(tmp_path)
        cache = PipelineCache(workspace)
        timeline = Timeline()
        assert not cache.hit("input", timeline)

    def test_commit_and_hit(self, tmp_path):
        workspace = Workspace(tmp_path)
        cache = PipelineCache(workspace)
        timeline = Timeline()
        cache.commit("input", timeline)
        assert cache.hit("input", timeline)

    def test_load_after_commit(self, tmp_path):
        workspace = Workspace(tmp_path)
        cache = PipelineCache(workspace)
        timeline = Timeline()
        timeline.metadata["key"] = "value"
        cache.commit("input", timeline)
        loaded = cache.load("input")
        assert loaded.metadata["key"] == "value"

    def test_should_skip_with_resume(self, tmp_path):
        workspace = Workspace(tmp_path)
        cache = PipelineCache(workspace)
        timeline = Timeline()
        cache.commit("input", timeline)
        assert cache.should_skip("input", resume=True)
        assert not cache.should_skip("asr", resume=True)

    def test_reset(self, tmp_path):
        workspace = Workspace(tmp_path)
        cache = PipelineCache(workspace)
        timeline = Timeline()
        cache.commit("input", timeline)
        cache.reset()
        assert not cache.hit("input", timeline)
        assert cache.last_completed() is None


class TestPipelineOrchestrator:
    def test_run_default_steps(self, tmp_path):
        from opendubbing.engines.base import create_default_engine_registry
        from opendubbing.pipeline.orchestrator import PipelineOrchestrator

        video = _make_test_video(tmp_path)
        workspace = Workspace(tmp_path / "ws")
        registry = create_default_engine_registry()
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config={"input_path": str(video), "providers": {}},
            engine_registry=registry,
        )
        timeline = orchestrator.run(steps=["input"])
        assert isinstance(timeline, Timeline)
        assert workspace.timeline_path.exists()

    def test_progress_callback(self, tmp_path):
        from opendubbing.engines.base import create_default_engine_registry
        from opendubbing.pipeline.orchestrator import PipelineOrchestrator

        video = _make_test_video(tmp_path)
        workspace = Workspace(tmp_path / "ws")
        registry = create_default_engine_registry()
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config={"input_path": str(video), "providers": {}},
            engine_registry=registry,
        )
        events = []
        orchestrator.add_progress_callback(
            lambda step, payload: events.append((step, payload))
        )
        orchestrator.run(steps=["input"])
        assert any(step == "input" for step, _ in events)

    def test_resume_skips_completed(self, tmp_path):
        from opendubbing.engines.base import create_default_engine_registry
        from opendubbing.pipeline.orchestrator import PipelineOrchestrator

        video = _make_test_video(tmp_path)
        workspace = Workspace(tmp_path / "ws")
        registry = create_default_engine_registry()
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config={"input_path": str(video), "providers": {}},
            engine_registry=registry,
        )
        orchestrator.run(steps=["input"])
        events = []
        orchestrator.add_progress_callback(
            lambda step, payload: events.append((step, payload))
        )
        orchestrator.run(steps=["input"], resume=True)
        assert any(payload["status"] == "skipped" for _, payload in events)

    def test_run_requires_input_file(self, tmp_path):
        from opendubbing.engines.base import create_default_engine_registry
        from opendubbing.pipeline.errors import PipelineError
        from opendubbing.pipeline.orchestrator import PipelineOrchestrator

        workspace = Workspace(tmp_path / "ws")
        registry = create_default_engine_registry()
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config={"input_path": str(tmp_path / "missing.mp4"), "providers": {}},
            engine_registry=registry,
        )
        with pytest.raises(PipelineError):
            orchestrator.run(steps=["input"])
