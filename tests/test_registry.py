import pytest

from opendubbing.core.interfaces import Provider, ProviderNotFoundError
from opendubbing.core.registry import ProviderRegistry, create_default_registry


class FakeProvider(Provider):
    name = "fake"
    kind = "test"

    def initialize(self, config):
        self.config = config

    def load_model(self):
        pass

    def infer(self, inputs):
        return inputs

    def release(self):
        pass


class TestProviderRegistry:
    def test_register_and_get(self):
        registry = ProviderRegistry()
        registry.register("test", "fake", FakeProvider)
        assert registry.get("test", "fake") is FakeProvider

    def test_get_unknown_raises(self):
        registry = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            registry.get("test", "missing")

    def test_build(self):
        registry = ProviderRegistry()
        registry.register("test", "fake", FakeProvider)
        provider = registry.build("test", "fake", {"key": "value"})
        assert isinstance(provider, FakeProvider)
        assert provider.config == {"key": "value"}

    def test_decorator(self):
        registry = ProviderRegistry()

        @registry.decorator("test", "decorated")
        class DecoratedProvider(FakeProvider):
            pass

        assert registry.get("test", "decorated") is DecoratedProvider

    def test_default_registry(self):
        registry = create_default_registry()
        kinds = registry.list_kinds()
        assert "asr" in kinds
        assert "tts" in kinds
        assert "translation" in kinds
        assert "qwen3_asr" in registry.list_names("asr")
        assert "cosyvoice2" in registry.list_names("tts")
