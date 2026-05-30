"""yttranscript-mcp — fetch, clean, and search YouTube transcripts.

Captions-first (fast, low-bandwidth, rate-limit resistant) with an optional
local Whisper fallback, plus LLM-friendly cleaning that strips rolling-caption
duplication and timestamps.
"""

from .cleaner import clean_segments, estimate_tokens, merge_overlapping
from .embeddings import get_embedder
from .index import CorpusIndex
from .search_service import search, search_and_get_transcripts, search_ranked
from .searcher import VideoSearchResult
from .semantic import Chunk, Passage, search_transcript, seconds_to_clock
from .service import (
    AskResult,
    SearchInVideoResult,
    ServiceResult,
    ask_video,
    corpus_stats,
    find_in_corpus,
    get_transcript,
    get_transcript_data,
    get_transcripts_batch,
    get_video_info,
    index_videos,
    list_languages,
    search_in_video,
)
from .summarizer import preload, summarize_text

__version__ = "0.4.0"

__all__ = [
    # transcripts
    "get_transcript",
    "get_transcript_data",
    "get_transcripts_batch",
    "ServiceResult",
    "get_video_info",
    "list_languages",
    # search (keyword + neural)
    "VideoSearchResult",
    "search",
    "search_ranked",
    "search_and_get_transcripts",
    # semantic search / RAG (fully local)
    "search_transcript",
    "search_in_video",
    "ask_video",
    "AskResult",
    "SearchInVideoResult",
    "Passage",
    "Chunk",
    "seconds_to_clock",
    "get_embedder",
    # corpus index (cross-video semantic search)
    "CorpusIndex",
    "index_videos",
    "find_in_corpus",
    "corpus_stats",
    # cleaning / summarization
    "clean_segments",
    "merge_overlapping",
    "estimate_tokens",
    "summarize_text",
    "preload",
    "__version__",
]
