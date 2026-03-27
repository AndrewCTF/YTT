"""YouTube Transcript Fetcher - Core library."""

from .service import get_transcript, get_transcripts_batch, ServiceResult
from .searcher import VideoSearchResult
from .search_service import search, search_and_get_transcripts

__all__ = [
    "get_transcript",
    "get_transcripts_batch",
    "ServiceResult",
    "VideoSearchResult",
    "search",
    "search_and_get_transcripts",
]
