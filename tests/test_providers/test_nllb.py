"""Tests for NLLBProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opendubbing.core.interfaces import ProviderModelLoadError
from opendubbing.providers.translation.nllb import NLLBProvider


class TestNLLBProvider:
    def test_load_model_raises_when_transformers_missing(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "transformers", None)
        provider = NLLBProvider()
        provider.initialize({"name": "nllb"})
        with pytest.raises(ProviderModelLoadError):
            provider.load_model()

    def test_infer_maps_language_codes_and_decodes(self, monkeypatch):
        fake_tokenizer = MagicMock()
        fake_tokenizer.src_lang = None
        fake_tokenizer.convert_tokens_to_ids.return_value = 12345
        fake_tokenizer.return_value = {"input_ids": "encoded"}
        fake_tokenizer.batch_decode.return_value = ["你好世界"]

        fake_model = MagicMock()
        fake_model.generate.return_value = "generated_ids"
        fake_model.to.return_value = fake_model

        fake_tokenizer_cls = MagicMock()
        fake_tokenizer_cls.from_pretrained.return_value = fake_tokenizer
        fake_model_cls = MagicMock()
        fake_model_cls.from_pretrained.return_value = fake_model

        fake_transformers = SimpleNamespace(
            AutoTokenizer=fake_tokenizer_cls, AutoModelForSeq2SeqLM=fake_model_cls
        )
        monkeypatch.setitem(
            __import__("sys").modules, "transformers", fake_transformers
        )

        provider = NLLBProvider()
        provider.initialize({"name": "nllb", "model": "facebook/nllb-test"})
        provider.load_model()

        result = provider.infer(
            {"text": "hello world", "source_language": "en", "target_language": "zh"}
        )

        assert result["translation"] == "你好世界"
        assert fake_tokenizer.src_lang == "eng_Latn"
        fake_tokenizer.convert_tokens_to_ids.assert_called_once_with("zho_Hans")
        fake_model.generate.assert_called_once_with(
            input_ids="encoded",
            forced_bos_token_id=12345,
            max_length=256,
        )
