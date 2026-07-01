"""NLLB translation provider."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError


class NLLBProvider(Provider):
    """Translation using NLLB."""

    name = "nllb"
    kind = "translation"

    # NLLB 200 language codes for common languages
    _LANG_MAP = {
        "zh": "zho_Hans",
        "en": "eng_Latn",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "es": "spa_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "ru": "rus_Cyrl",
        "pt": "por_Latn",
        "it": "ita_Latn",
        "ar": "arb_Arab",
        "hi": "hin_Deva",
    }

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = config.get("model", "facebook/nllb-200-distilled-600M")
        self.options = config.get("options", {})
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderModelLoadError(
                "transformers not installed; install opendubbing[heavy]"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
            device = self.options.get("device", "cpu")
            self._model.to(device)
        except Exception as exc:
            raise ProviderModelLoadError(f"Failed to load NLLB model {self.model}") from exc

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None or self._tokenizer is None:
            self.load_model()

        text = inputs["text"]
        target = inputs.get("target_language", "zh")
        src = inputs.get("source_language", "en")
        target_code = self._LANG_MAP.get(target, target)
        src_code = self._LANG_MAP.get(src, src)

        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(text, return_tensors="pt")
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(target_code)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
        )
        translation = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return {"translation": translation}

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
