"""Standalone CosyVoice2 inference script for OpenDubbing subprocess TTS.

This script is executed inside the dedicated CosyVoice Conda environment to avoid
dependency conflicts with OpenDubbing. It expects ``C:\CosyVoice`` and
``C:\CosyVoice\third_party\Matcha-TTS`` to be on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice2 TTS inference")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--reference_audio", default="")
    parser.add_argument("--reference_text", default="")
    parser.add_argument("--speech_rate", type=float, default=1.0)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    sys.path.insert(0, r"C:\CosyVoice")
    sys.path.insert(0, r"C:\CosyVoice\third_party\Matcha-TTS")

    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosyvoice = CosyVoice2(args.model_dir)

    if args.reference_audio:
        ref_path = Path(args.reference_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        # CosyVoice2's zero-shot frontend cannot handle reference audio > 30s.
        # Load the reference, truncate to the first 30s, and write a temp file.
        ref_audio, ref_sr = sf.read(str(ref_path), dtype=np.float32)
        max_ref_samples = int(30 * ref_sr)
        if ref_audio.shape[0] > max_ref_samples:
            ref_audio = ref_audio[:max_ref_samples]
            truncated_ref = ref_path.with_name(
                f"{ref_path.stem}_truncated{ref_path.suffix}"
            )
            sf.write(str(truncated_ref), ref_audio, ref_sr)
            ref_path = truncated_ref

        result = list(
            cosyvoice.inference_zero_shot(
                args.text,
                args.reference_text,
                str(ref_path),
                zero_shot_spk_id="",
            )
        )
    else:
        # CosyVoice2-0.5B ships without spk2info.pt, so SFT mode requires a
        # registered speaker. Fail loudly rather than producing bad output.
        raise SystemExit(
            "CosyVoice2 inference requires --reference_audio for zero-shot cloning."
        )

    audio = result[0]["tts_speech"].numpy().squeeze()

    # Apply simple speech-rate adjustment via resampling if needed.
    if args.speech_rate != 1.0 and args.speech_rate > 0:
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=args.sample_rate,
            target_sr=int(args.sample_rate / args.speech_rate),
        )

    sf.write(args.out_path, audio, args.sample_rate)
    duration = len(audio) / args.sample_rate
    print(f"Wrote {args.out_path}, duration={duration:.3f}s")


if __name__ == "__main__":
    main()
