"""Edge TTS provider using Microsoft Edge online text-to-speech."""

from __future__ import annotations

import asyncio
import subprocess
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Provider, ProviderModelLoadError
from opendubbing.utils import media


class EdgeTTSProvider(Provider):
    """Text-to-speech via the ``edge-tts`` package.

    Uses Microsoft Edge's online TTS service, which requires an internet
    connection but no API key. Supports multiple languages via voice selection.
    """

    name = "edge_tts"
    kind = "tts"

    _VOICE_MAP = {
        "zh": "zh-CN-XiaoxiaoNeural",
        "en": "en-US-AriaNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
    }

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = config.get("options", {})
        self._ready = False

    def load_model(self) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise ProviderModelLoadError(
                "edge-tts not installed; run: pip install edge-tts"
            ) from exc
        self._ready = True

    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run an async coroutine from synchronous code safely.

        The API server runs inside an asyncio event loop, so ``asyncio.run``
        would fail. When a loop is already running we offload the coroutine to
        a dedicated thread with its own event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = None
        exception: BaseException | None = None

        def target() -> None:
            nonlocal result, exception
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
            except BaseException as exc:  # noqa: BLE001
                exception = exc
            finally:
                new_loop.close()

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
        if exception is not None:
            raise exception
        return result

    def _read_audio_with_fallback(self, path: Path, sample_rate: int) -> tuple[Any, int]:
        """Read audio, converting to WAV via ffmpeg if soundfile cannot read it."""
        try:
            return media.read_audio(path, sample_rate=sample_rate)
        except Exception:
            wav_path = path.with_suffix(".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ar", str(sample_rate), "-ac", "1", str(wav_path)],
                check=True,
                capture_output=True,
            )
            try:
                return media.read_audio(wav_path, sample_rate=sample_rate)
            finally:
                wav_path.unlink(missing_ok=True)

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self._ready:
            self.load_model()

        import edge_tts

        text = inputs["text"]
        out_path = Path(inputs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        language = inputs.get("language", "en")
        voice = self.options.get("voice") or self._VOICE_MAP.get(
            language, "en-US-AriaNeural"
        )
        rate = inputs.get("speech_rate", 1.0)
        rate_str = f"{int((rate - 1.0) * 100):+d}%"

        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        self._run_async(communicate.save(str(out_path)))

        samples, sr = self._read_audio_with_fallback(out_path, sample_rate=16000)
        duration = len(samples) / sr

        return {"duration": duration, "path": str(out_path)}

    def release(self) -> None:
        self._ready = False
