"""Configuration constants for the YouTube transcript fetcher."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Innertube client settings
    INNERTUBE_CLIENT: str = "ANDROID"
    INNERTUBE_CLIENT_VERSION: str = "20.10.38"
    INNERTUBE_API_URL: str = "https://www.youtube.com/youtubei/v1/player"
    VIDEO_PAGE_URL: str = "https://www.youtube.com/watch"

    # Rate limiting
    RATE_LIMIT_RATE: float = 0.5  # tokens per second
    RATE_LIMIT_BURST: int = 5    # max tokens (bucket size)

    # Cache settings
    CACHE_TTL_DAYS: int = 7
    CACHE_DB_PATH: str = ".transcript_cache.db"

    # Whisper settings
    WHISPER_MODEL: str = "base"  # tiny/base/small/medium/large
    WHISPER_FALLBACK_ENABLED: bool = True
    WHISPER_USE_GPU: bool = True   # Enable GPU acceleration
    WHISPER_BATCH_SIZE: int = 16  # For GPU batched inference

    # Batch processing
    MAX_BATCH_SIZE: int = 50
    MAX_CONCURRENT_WORKERS: int = 4  # Max parallel transcription tasks

    # Output formats
    DEFAULT_LANGUAGE: str = "en"
    DEFAULT_FORMAT: str = "text"  # text/json/srt/vtt

    # Temp directory for Whisper audio files
    TEMP_DIR: str = ".tmp"

    # Search settings
    SEARCH_CACHE_TTL_HOURS: int = 24
    SEARCH_DEFAULT_LIMIT: int = 5
    SEARCH_MAX_LIMIT: int = 20

    # CUDA settings
    # Automatically download CUDA/cuBLAS when GPU is needed (avoids bundling)
    AUTO_DOWNLOAD_CUDA: bool = False


# Global config instance
config = Config()
