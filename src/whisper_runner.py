"""Whisper-based transcript fallback using yt-dlp and faster-whisper."""

import os
import tempfile
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

from config import config
from .exceptions import WhisperError


# Model cache for reusing WhisperModel instances
_model_cache: dict = {}
_cache_lock = threading.Lock()


@dataclass
class WhisperSegment:
    """A segment from Whisper transcription."""
    start: float  # seconds
    end: float    # seconds
    text: str


@dataclass
class WhisperResult:
    """Complete Whisper transcription result."""
    video_id: str
    language: str
    segments: list[WhisperSegment]
    text: str  # Full concatenated text


def get_audio_path() -> Path:
    """Get the temp directory for audio files."""
    tmp_dir = Path(config.TEMP_DIR)
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def _get_cuda_device() -> tuple[str, str]:
    """Detect GPU availability and return (device, compute_type)."""
    if config.WHISPER_USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                return ("cuda", "float16")
        except Exception:
            pass
    return ("cpu", "int8")


def _get_cached_model(model_size: str):
    """Get or create cached WhisperModel with BatchedInferencePipeline for GPU.

    Returns:
        Tuple of (WhisperModel, BatchedInferencePipeline or None)
    """
    device, compute_type = _get_cuda_device()
    cache_key = f"{model_size}:{device}"

    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    # Import here to avoid hard dependency
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise WhisperError(
            "faster-whisper is required for Whisper fallback. "
            "Install with: pip install faster-whisper"
        )

    # Try GPU model first if GPU is enabled
    if device == "cuda":
        try:
            model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16",
            )
            # Try to create batched pipeline for GPU
            batched = None
            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=model)
            except ImportError:
                pass
            _model_cache[cache_key] = (model, batched)
            return (model, batched)
        except Exception as e:
            if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                import warnings
                warnings.warn(f"CUDA model creation failed ({e}), using CPU fallback")
            else:
                raise

    # Fall back to CPU with int8
    cache_key = f"{model_size}:cpu"
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )
    _model_cache[cache_key] = (model, None)
    return (model, None)


def download_audio(video_id: str) -> tuple[str, str]:
    """Download audio from a YouTube video using yt-dlp.

    Args:
        video_id: The YouTube video ID.

    Returns:
        Tuple of (audio_path, temp_dir) - caller must clean up temp_dir.

    Raises:
        WhisperError: If audio download fails.
    """
    temp_dir = tempfile.mkdtemp(prefix="yt_transcript_")
    audio_path = os.path.join(temp_dir, "audio")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'quiet': True,
        'no_warnings': True,
        'extractaudio': True,
        'audioformat': 'mp3',
        'audioquality': '0',
        'retries': 3,
        'fragment_retries': 3,
    }

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the actual file (yt-dlp may add extension)
        files = os.listdir(temp_dir)
        if not files:
            raise WhisperError(f"No audio file generated for {video_id}")

        actual_path = os.path.join(temp_dir, files[0])
        return actual_path, temp_dir

    except yt_dlp.utils.DownloadError as e:
        raise WhisperError(f"Failed to download audio for {video_id}: {e}")


def transcribe_audio(audio_path: str, model_size: str = "base") -> WhisperResult:
    """Transcribe audio using faster-whisper with cached model.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size (tiny/base/small/medium/large).

    Returns:
        WhisperResult with segments and full text.

    Raises:
        WhisperError: If transcription fails.
    """
    tried_cpu_fallback = False
    try:
        model, batched_model = _get_cached_model(model_size)

        if batched_model is not None:
            # GPU: use BatchedInferencePipeline for efficient batched inference
            try:
                segments, info = batched_model.transcribe(
                    audio_path,
                    language=None,
                    batch_size=config.WHISPER_BATCH_SIZE,
                )
            except Exception as e:
                if "cublas" in str(e).lower():
                    # Cublas error during transcription - fall back to CPU
                    import warnings
                    warnings.warn(f"GPU transcription failed (cublas: {e}), falling back to CPU")
                    tried_cpu_fallback = True
                    # Clear GPU model from cache
                    with _cache_lock:
                        gpu_key = f"{model_size}:cuda"
                        if gpu_key in _model_cache:
                            del _model_cache[gpu_key]
                    model, batched_model = _get_cached_model(model_size)
                    segments, info = model.transcribe(
                        audio_path,
                        language=None,
                        beam_size=5,
                        word_timestamps=True,
                    )
                else:
                    raise
        else:
            # CPU: use standard model
            segments, info = model.transcribe(
                audio_path,
                language=None,
                beam_size=5,
                word_timestamps=True,
            )

        # Convert to our segment type
        result_segments = []
        full_text_parts = []

        for seg in segments:
            result_segments.append(WhisperSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))
            full_text_parts.append(seg.text.strip())

        return WhisperResult(
            video_id="",  # Will be set by caller
            language=info.language or "en",
            segments=result_segments,
            text=" ".join(full_text_parts),
        )

    except Exception as e:
        raise WhisperError(f"Whisper transcription failed: {e}")


def cleanup_temp_dir(temp_dir: str) -> None:
    """Remove a temporary directory and its contents."""
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


def fetch_transcript_whisper(video_id: str) -> WhisperResult:
    """Fetch transcript using Whisper fallback.

    This method is used when Innertube fails (rate limited or no captions).
    It downloads the audio and transcribes it using faster-whisper.

    Args:
        video_id: The YouTube video ID.

    Returns:
        WhisperResult with transcribed text and timestamps.

    Raises:
        WhisperError: If Whisper fallback fails.
    """
    audio_path = None
    temp_dir = None

    try:
        # Download audio
        audio_path, temp_dir = download_audio(video_id)

        # Transcribe
        result = transcribe_audio(audio_path, config.WHISPER_MODEL)
        result.video_id = video_id

        return result

    finally:
        # Always clean up temp files
        if temp_dir:
            cleanup_temp_dir(temp_dir)
        elif audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass