"""FFmpeg-based media I/O utilities shared by engines and providers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import soundfile as sf


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def probe_duration(path: Path | str) -> float:
    """Return media duration in seconds via ffprobe.

    Args:
        path: Path to media file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails or duration cannot be determined.
    """
    path = _resolve(path)
    try:
        info = ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr}") from exc

    duration = info.get("format", {}).get("duration")
    if duration:
        return float(duration)

    for stream in info.get("streams", []):
        duration = stream.get("duration")
        if duration:
            return float(duration)

    raise RuntimeError(f"Could not determine duration for {path}")


def extract_audio(
    source: Path | str,
    out_path: Path | str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio track to 16-bit PCM wav at sample_rate.

    Args:
        source: Input video or audio file.
        out_path: Destination wav path.
        sample_rate: Target sample rate.
        mono: Whether to downmix to mono.

    Returns:
        Resolved output path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    source = _resolve(source)
    out_path = _resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(str(source))
    kwargs: dict[str, Any] = {"ar": sample_rate, "ac": 1 if mono else None}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    stream = stream.output(str(out_path), **kwargs).overwrite_output()

    try:
        ffmpeg.run(stream, quiet=True)
    except ffmpeg.Error as exc:
        raise RuntimeError(f"extract_audio failed: {exc.stderr}") from exc
    return out_path


def read_audio(
    path: Path | str,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read audio as float32 numpy array.

    Args:
        path: Audio file path.
        sample_rate: If given, resample to this rate via librosa.

    Returns:
        Tuple of (samples, sample_rate).
    """
    path = _resolve(path)
    samples, sr = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate is not None and sr != sample_rate:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return samples, sr


def write_audio(
    samples: np.ndarray,
    sample_rate: int,
    path: Path | str,
) -> Path:
    """Write float32 samples to wav.

    Args:
        samples: Audio samples.
        sample_rate: Sample rate.
        path: Destination path.

    Returns:
        Resolved output path.
    """
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def concat_audio(segments: list[Path], out_path: Path | str) -> Path:
    """Concatenate wav segments in order using ffmpeg concat demuxer.

    Args:
        segments: Ordered list of wav file paths.
        out_path: Destination path.

    Returns:
        Resolved output path.
    """
    if not segments:
        raise ValueError("segments must not be empty")

    out_path = _resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for seg in segments:
            f.write(f"file '{_resolve(seg).as_posix()}\n")
        concat_list = f.name

    try:
        stream = (
            ffmpeg.input(concat_list, f="concat", safe=0)
            .output(str(out_path), c="copy")
            .overwrite_output()
        )
        try:
            ffmpeg.run(stream, quiet=True)
        except ffmpeg.Error as exc:
            raise RuntimeError(f"concat_audio failed: {exc.stderr}") from exc
    finally:
        Path(concat_list).unlink(missing_ok=True)

    return out_path


def mux_audio_video(
    video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    codec: str = "libx264",
) -> Path:
    """Mux audio into video, replacing original audio track.

    Args:
        video: Source video path.
        audio: Replacement audio path.
        out_path: Destination path.
        codec: Video codec.

    Returns:
        Resolved output path.
    """
    video = _resolve(video)
    audio = _resolve(audio)
    out_path = _resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v = ffmpeg.input(str(video))
    a = ffmpeg.input(str(audio))
    stream = ffmpeg.output(
        v["v"],
        a["a"],
        str(out_path),
        vcodec=codec,
        acodec="aac",
        audio_bitrate="192k",
    ).overwrite_output()

    try:
        ffmpeg.run(stream, quiet=True)
    except ffmpeg.Error as exc:
        raise RuntimeError(f"mux_audio_video failed: {exc.stderr}") from exc
    return out_path


def segment_audio(
    audio: Path | str,
    segments: list[tuple[float, float]],
    out_dir: Path | str,
    prefix: str = "seg",
) -> list[Path]:
    """Cut audio into [start,end) second segments.

    Args:
        audio: Source audio path.
        segments: List of (start, end) in seconds.
        out_dir: Output directory.
        prefix: Output filename prefix.

    Returns:
        Ordered list of output paths.
    """
    audio = _resolve(audio)
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        out = out_dir / f"{prefix}_{i:04d}.wav"
        stream = (
            ffmpeg.input(str(audio), ss=start, t=duration)
            .output(str(out), ar=16000, ac=1)
            .overwrite_output()
        )
        try:
            ffmpeg.run(stream, quiet=True)
        except ffmpeg.Error as exc:
            raise RuntimeError(f"segment_audio failed at {start}-{end}: {exc.stderr}") from exc
        paths.append(out)
    return paths
