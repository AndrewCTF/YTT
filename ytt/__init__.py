"""yttranscript-mcp — fetch, clean, and search YouTube transcripts.

Captions-first (fast, low-bandwidth, rate-limit resistant) with an optional
local Whisper fallback, plus LLM-friendly cleaning that strips rolling-caption
duplication and timestamps.
"""

from .cleaner import clean_segments, estimate_tokens, merge_overlapping
from .search_service import search, search_and_get_transcripts
from .searcher import VideoSearchResult
from .service import ServiceResult, get_transcript, get_transcripts_batch
from .summarizer import preload, summarize_text

__version__ = "0.3.0"

__all__ = [
    "get_transcript",
    "get_transcripts_batch",
    "ServiceResult",
    "VideoSearchResult",
    "search",
    "search_and_get_transcripts",
    "clean_segments",
    "merge_overlapping",
    "estimate_tokens",
    "summarize_text",
    "preload",
    "__version__",
]
