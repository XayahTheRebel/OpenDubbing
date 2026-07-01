import pytest

from opendubbing.core.interfaces import Engine, EngineNotFoundError, Provider
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.engines.asr_engine import ASREngine
from opendubbing.engines.base import EngineRegistry, create_default_engine_registry
from opendubbing.engines.input_engine import InputEngine
from opendubbing.engines.translation_engine import TranslationEngine
from opendubbing.pipeline.orchestrator import PipelineOrchestrator


class FakeEngine(Engine):
    name = "fake"

    def initialize(self, config):
        pass

    def process(self, timeline, workspace):
        timeline.metadata["fake"] = True
        return timeline

    def save(self, timeline, workspace):
        timeline.save(workspace.timeline_path)

    def release(self):
        pass


class FakeAsrProvider(Provider):
    name = "fake_asr"
    kind = "asr"

    def initialize(self, config):
        pass

    def load_model(self):
        pass

    def infer(self, inputs):
        return {
            "language": "en",
            "segments": [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": 0.95,
                    "words": [
                        {"text": "hello", "start": 0.0, "end": 0.5},
                        {"text": "world", "start": 0.5, "end": 1.0},
                    ],
                }
            ],
        }

    def release(self):
        pass


class FakeTranslationProvider(Provider):
    name = "fake_translation"
    kind = "translation"

    def initialize(self, config):
        pass

    def load_model(self):
        pass

    def infer(self, inputs):
        return {"translation": "你好世界"}

    def release(self):
        pass


class FakeTtsProvider(Provider):
    name = "fake_tts"
    kind = "tts"

    def initialize(self, config):
        pass

    def load_model(self):
        pass

    def infer(self, inputs):
        import numpy as np

        from opendubbing.utils import media

        out = inputs["out_path"]
        media.write_audio(np.zeros(16000, dtype=np.float32), 16000, out)
        return {"duration": 1.0, "path": out}

    def release(self):
        pass


class FakeFaceProvider(Provider):
    name = "fake_face"
    kind = "face"

    def initialize(self, config):
        pass

    def load_model(self):
        pass

    def infer(self, inputs):
        from opendubbing.utils import media

        media.mux_audio_video(inputs["video"], inputs["audio"], inputs["out_path"])
        return {"path": inputs["out_path"], "fallback": True}

    def release(self):
        pass


class TestEngineRegistry:
    def test_register_and_get(self):
        registry = EngineRegistry()
        registry.register("fake", FakeEngine)
        assert registry.get("fake") is FakeEngine

    def test_get_unknown_raises(self):
        registry = EngineRegistry()
        with pytest.raises(EngineNotFoundError):
            registry.get("missing")

    def test_default_registry(self):
        registry = create_default_engine_registry()
        names = registry.list_names()
        assert "input" in names
        assert "asr" in names
        assert "tts" in names
        assert "output" in names

    def test_build_engine(self):
        registry = EngineRegistry()
        registry.register("fake", FakeEngine)
        engine = registry.build("fake", {"key": "value"})
        assert isinstance(engine, FakeEngine)


class TestEngines:
    def test_input_engine_requires_existing_file(self, tmp_path):
        engine = InputEngine()
        engine.initialize({"input_path": str(tmp_path / "missing.mp4")})
        workspace = Workspace(tmp_path / "ws")
        with pytest.raises(FileNotFoundError):
            engine.process(Timeline(), workspace)

    def test_asr_engine_with_fake_provider(self, tmp_path):
        registry = ProviderRegistry()
        registry.register("asr", "fake_asr", FakeAsrProvider)
        engine = ASREngine()
        engine.initialize(
            {"providers": {"asr": {"name": "fake_asr"}}, "registry": registry}
        )
        workspace = Workspace(tmp_path / "ws")
        timeline = Timeline(metadata={"input_audio": str(tmp_path / "audio.wav")})
        timeline = engine.process(timeline, workspace)
        assert len(timeline.sentences) == 1
        assert timeline.sentences[0].text == "hello world"

    def test_translation_engine_with_fake_provider(self, tmp_path):
        registry = ProviderRegistry()
        registry.register("translation", "fake_translation", FakeTranslationProvider)
        engine = TranslationEngine()
        engine.initialize(
            {
                "providers": {"translation": {"name": "fake_translation"}},
                "registry": registry,
                "target_language": "zh",
            }
        )
        workspace = Workspace(tmp_path / "ws")
        timeline = Timeline()
        timeline.append_sentence = True
        from opendubbing.core.timeline import Sentence

        timeline.append(Sentence(id="s0000", text="hello world"))
        timeline = engine.process(timeline, workspace)
        assert timeline.sentences[0].translation == "你好世界"


class TestIntegration:
    def test_full_pipeline_with_fake_providers(self, tmp_path):
        import numpy as np

        from opendubbing.utils import media

        registry = ProviderRegistry()
        registry.register("asr", "fake_asr", FakeAsrProvider)
        registry.register("translation", "fake_translation", FakeTranslationProvider)
        registry.register("tts", "fake_tts", FakeTtsProvider)
        registry.register("face", "fake_face", FakeFaceProvider)

        engine_registry = create_default_engine_registry()

        video = tmp_path / "test.mp4"
        audio = media.write_audio(
            np.zeros(16000 * 2, dtype=np.float32), 16000, tmp_path / "a.wav"
        )
        # Generate a 2-second blank video with ffmpeg testsrc, then mux audio.
        import ffmpeg as _ffmpeg

        blank_video = tmp_path / "blank.mp4"
        (
            _ffmpeg
            .input("testsrc=size=320x240:rate=1", f="lavfi", t=2)
            .output(str(blank_video), vcodec="libx264", pix_fmt="yuv420p")
            .overwrite_output()
            .run(quiet=True)
        )
        media.mux_audio_video(blank_video, audio, video)

        config = {
            "input_path": str(video),
            "target_language": "zh",
            "sample_rate": 16000,
            "providers": {
                "asr": {"name": "fake_asr"},
                "forced_alignment": {"name": ""},
                "translation": {"name": "fake_translation"},
                "tts": {"name": "fake_tts"},
                "face": {"name": "fake_face"},
            },
            "output": {"filename": "output.mp4", "codec": "libx264"},
        }
        workspace = Workspace(tmp_path / "ws")
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config=config,
            engine_registry=engine_registry,
            provider_registry=registry,
        )
        orchestrator.run()

        output_path = workspace.path_for("output", "output.mp4")
        assert output_path.exists()
        assert orchestrator.cache.last_completed() == "output"
