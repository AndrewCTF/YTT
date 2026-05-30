"""Configuration for the YouTube transcript fetcher.

All values can be overridden with ``YTT_*`` environment variables, which makes
the library easy to tune in deployments that hit rate limits (e.g. point it at
a proxy or supply a cookies file) without touching code.
"""

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# The public Innertube API key has been stable for years and is embedded in
# every youtube.com page. Hardcoding it lets us skip a ~1 MB watch-page fetch
# (which is slow and a prime bot-detection trigger) on every transcript call.
DEFAULT_INNERTUBE_API_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"


# Caption tracks are fetched through the Innertube ``player`` endpoint. YouTube
# has locked down the plain ANDROID/IOS clients (they now return
# FAILED_PRECONDITION without an attestation/PoToken), so we lead with the
# ANDROID_VR client, which still returns signed caption URLs without a token,
# then fall back to WEB / MWEB which work from many residential IPs. Each entry
# carries the client context plus a matching User-Agent — a realistic UA is the
# single biggest factor in avoiding 429/empty responses. The fetcher tries each
# in order until one yields caption tracks, then a watch-page scrape as a last
# resort.
INNERTUBE_CLIENTS: list[dict] = [
    {
        "name": "ANDROID_VR",
        "context": {
            "clientName": "ANDROID_VR",
            "clientVersion": "1.57.29",
            "deviceModel": "Quest 3",
            "androidSdkVersion": 32,
            "hl": "en",
            "gl": "US",
        },
        "headers": {
            "User-Agent": (
                "com.google.android.apps.youtube.vr.oculus/1.57.29 "
                "(Linux; U; Android 12L; Quest 3) gzip"
            ),
            "X-YouTube-Client-Name": "28",
            "X-YouTube-Client-Version": "1.57.29",
        },
    },
    {
        "name": "WEB",
        "context": {
            "clientName": "WEB",
            "clientVersion": "2.20240304.00.00",
            "hl": "en",
            "gl": "US",
        },
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "X-YouTube-Client-Name": "1",
            "X-YouTube-Client-Version": "2.20240304.00.00",
            "Origin": "https://www.youtube.com",
            "Referer": "https://www.youtube.com/",
        },
    },
    {
        "name": "MWEB",
        "context": {
            "clientName": "MWEB",
            "clientVersion": "2.20240304.08.00",
            "hl": "en",
            "gl": "US",
        },
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
            ),
            "X-YouTube-Client-Name": "2",
            "X-YouTube-Client-Version": "2.20240304.08.00",
        },
    },
]

# Realistic browser UA for the search results endpoint / generic requests.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class Config:
    """Application configuration (env-overridable)."""

    # Innertube endpoints
    INNERTUBE_API_KEY: str = field(
        default_factory=lambda: _env_str("YTT_INNERTUBE_API_KEY", DEFAULT_INNERTUBE_API_KEY)
    )
    INNERTUBE_PLAYER_URL: str = "https://www.youtube.com/youtubei/v1/player"
    INNERTUBE_SEARCH_URL: str = "https://www.youtube.com/youtubei/v1/search"

    # Networking / anti-rate-limit
    REQUEST_TIMEOUT: int = field(default_factory=lambda: _env_int("YTT_TIMEOUT", 15))
    MAX_RETRIES: int = field(default_factory=lambda: _env_int("YTT_MAX_RETRIES", 4))
    BACKOFF_BASE: float = field(default_factory=lambda: _env_float("YTT_BACKOFF_BASE", 1.5))
    BACKOFF_MAX: float = field(default_factory=lambda: _env_float("YTT_BACKOFF_MAX", 30.0))
    # Optional outbound proxy, e.g. "http://user:pass@host:port". Routes every
    # request through it — the real escape hatch for IP-level rate limits.
    PROXY: str = field(default_factory=lambda: _env_str("YTT_PROXY", ""))
    # Optional Netscape cookies.txt path (passed to yt-dlp) for age/region
    # restricted videos.
    COOKIES_FILE: str = field(default_factory=lambda: _env_str("YTT_COOKIES_FILE", ""))

    # Client-side rate limiting (token bucket, applied to all network calls)
    RATE_LIMIT_RATE: float = field(default_factory=lambda: _env_float("YTT_RATE", 1.0))
    RATE_LIMIT_BURST: int = field(default_factory=lambda: _env_int("YTT_BURST", 5))

    # Cache settings
    CACHE_TTL_DAYS: int = field(default_factory=lambda: _env_int("YTT_CACHE_TTL_DAYS", 7))
    CACHE_DB_PATH: str = field(
        default_factory=lambda: _env_str("YTT_CACHE_DB", ".transcript_cache.db")
    )

    # Whisper settings (optional fallback; requires the ``whisper`` extra)
    WHISPER_MODEL: str = field(default_factory=lambda: _env_str("YTT_WHISPER_MODEL", "base"))
    WHISPER_FALLBACK_ENABLED: bool = field(
        default_factory=lambda: _env_bool("YTT_WHISPER_FALLBACK", True)
    )
    WHISPER_USE_GPU: bool = field(default_factory=lambda: _env_bool("YTT_WHISPER_GPU", True))
    WHISPER_BATCH_SIZE: int = field(default_factory=lambda: _env_int("YTT_WHISPER_BATCH", 16))

    # Batch processing
    MAX_BATCH_SIZE: int = 50
    MAX_CONCURRENT_WORKERS: int = field(default_factory=lambda: _env_int("YTT_WORKERS", 4))

    # Defaults
    DEFAULT_LANGUAGE: str = "en"
    DEFAULT_FORMAT: str = "text"

    # Temp dir for Whisper audio
    TEMP_DIR: str = ".tmp"

    # Search settings
    SEARCH_CACHE_TTL_HOURS: int = 24
    SEARCH_DEFAULT_LIMIT: int = 5
    SEARCH_MAX_LIMIT: int = 20

    # CUDA auto-download (opt-in, avoids bundling NVIDIA binaries)
    AUTO_DOWNLOAD_CUDA: bool = field(
        default_factory=lambda: _env_bool("YTT_AUTO_DOWNLOAD_CUDA", False)
    )

    # ---- Local-LLM summarization (token-saving, all opt-in) ----
    # Provider: "ollama" (default, http://localhost:11434) or "openai" (any
    # OpenAI-compatible /v1/chat/completions endpoint, e.g. llama.cpp / LM Studio).
    SUMMARY_PROVIDER: str = field(
        default_factory=lambda: _env_str("YTT_SUMMARY_PROVIDER", "ollama")
    )
    SUMMARY_MODEL: str = field(default_factory=lambda: _env_str("YTT_SUMMARY_MODEL", "qwen3.6:27b"))
    OLLAMA_URL: str = field(
        default_factory=lambda: _env_str("YTT_OLLAMA_URL", "http://localhost:11434")
    )
    # OpenAI-compatible endpoint (used when SUMMARY_PROVIDER == "openai").
    SUMMARY_OPENAI_BASE: str = field(
        default_factory=lambda: _env_str("YTT_SUMMARY_OPENAI_BASE", "http://localhost:8080/v1")
    )
    SUMMARY_API_KEY: str = field(default_factory=lambda: _env_str("YTT_SUMMARY_API_KEY", ""))
    SUMMARY_TIMEOUT: int = field(default_factory=lambda: _env_int("YTT_SUMMARY_TIMEOUT", 300))
    SUMMARY_TEMPERATURE: float = field(
        default_factory=lambda: _env_float("YTT_SUMMARY_TEMPERATURE", 0.2)
    )
    # Keep the model resident between calls ("hot loading"). Ollama keep_alive
    # syntax: "5m", "30m", "1h", "-1" (forever), "0" (unload immediately).
    SUMMARY_KEEP_ALIVE: str = field(
        default_factory=lambda: _env_str("YTT_SUMMARY_KEEP_ALIVE", "5m")
    )
    # Pull the model on demand if missing (only when the user opts in).
    SUMMARY_AUTO_PULL: bool = field(
        default_factory=lambda: _env_bool("YTT_SUMMARY_AUTO_PULL", False)
    )
    # Map-reduce chunk size (chars) for long transcripts.
    SUMMARY_MAX_INPUT_CHARS: int = field(
        default_factory=lambda: _env_int("YTT_SUMMARY_MAX_INPUT_CHARS", 12000)
    )
    SUMMARY_NUM_CTX: int = field(default_factory=lambda: _env_int("YTT_SUMMARY_NUM_CTX", 8192))

    # ---- Semantic search / embeddings (fully local, all opt-in) ----
    # Provider: "auto" (sentence-transformers if installed, else dependency-free
    # hashing embedder), "hash", "sentence-transformers", "ollama", or "openai".
    EMBED_PROVIDER: str = field(default_factory=lambda: _env_str("YTT_EMBED_PROVIDER", "auto"))
    EMBED_MODEL: str = field(default_factory=lambda: _env_str("YTT_EMBED_MODEL", ""))
    EMBED_HASH_DIM: int = field(default_factory=lambda: _env_int("YTT_EMBED_HASH_DIM", 256))
    # Chunking for retrieval (chars per window, with overlap for context bleed).
    CHUNK_TARGET_CHARS: int = field(default_factory=lambda: _env_int("YTT_CHUNK_CHARS", 480))
    CHUNK_OVERLAP_CHARS: int = field(default_factory=lambda: _env_int("YTT_CHUNK_OVERLAP", 80))
    # Local corpus index (cross-video semantic search).
    INDEX_DB_PATH: str = field(default_factory=lambda: _env_str("YTT_INDEX_DB", ".ytt_index.db"))

    @property
    def proxies(self) -> dict | None:
        """requests-style proxies dict, or None when no proxy is configured."""
        if not self.PROXY:
            return None
        return {"http": self.PROXY, "https": self.PROXY}


# Global config instance
config = Config()
