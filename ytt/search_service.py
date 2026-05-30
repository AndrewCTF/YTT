"""High-level search service orchestrator."""

import asyncio
from functools import partial
from typing import Optional

from .embeddings import get_embedder
from .http import new_session
from .rate_limiter import rate_limiter
from .search_cache import search_cache, CachedSearchResult
from .searcher import VideoSearchResult, search_videos_innertube
from .semantic import relevance_score
from .service import get_transcript, get_transcript_data, ServiceResult


async def search(
    query: str,
    max_results: int = 5,
    use_cache: bool = True,
) -> list[VideoSearchResult]:
    """Search YouTube for videos.

    This is the main entry point for search. It:
    1. Checks cache first
    2. Calls Innertube Search API (with rate limiting)
    3. Caches results

    Args:
        query: Search query string.
        max_results: Maximum results to return (default 5).
        use_cache: Whether to use cache (default True).

    Returns:
        List of VideoSearchResult objects.

    Raises:
        SearchError: If search fails after retries.
    """
    # Step 1: Check cache
    if use_cache:
        cached = await search_cache.get(query)
        if cached:
            return [_cached_to_search_result(c) for c in cached[:max_results]]

    # Step 2: Fetch from API (rate-limited, retry/backoff inside the session).
    await rate_limiter.acquire()
    loop = asyncio.get_event_loop()

    def _run() -> list[VideoSearchResult]:
        session = new_session()
        try:
            return search_videos_innertube(query, max_results, session)
        finally:
            session.close()

    results = await loop.run_in_executor(None, _run)

    # Step 3: Cache results
    if use_cache and results:
        await search_cache.set(query, results)

    return results


async def search_ranked(
    query: str,
    max_results: int = 5,
    pool: int | None = None,
    language: str = "en",
    use_cache: bool = True,
    embed_provider: str | None = None,
    embed_model: str | None = None,
) -> list[tuple[VideoSearchResult, float]]:
    """Neural re-ranking of search results by *transcript* relevance (exa-style).

    YouTube's keyword search returns a candidate ``pool``; we fetch each
    transcript and score how well its *content* (not just its title) answers the
    query using a shared local embedder, then return the best ``max_results``
    with their relevance scores. Fully local. Videos without transcripts sink to
    the bottom (score 0) rather than being dropped.
    """
    pool = pool or max(max_results * 3, max_results)
    results = await search(query, max_results=pool, use_cache=use_cache)
    if not results:
        return []

    try:
        embedder = get_embedder(embed_provider, embed_model)
    except Exception:
        embedder = get_embedder("hash")

    loop = asyncio.get_event_loop()

    async def score_one(r: VideoSearchResult) -> float:
        try:
            transcript, _ = await get_transcript_data(r.video_id, language, None, use_cache)
        except Exception:
            return 0.0
        try:
            return await loop.run_in_executor(
                None, partial(relevance_score, transcript.segments, query, embedder)
            )
        except Exception:
            return 0.0

    scores = await asyncio.gather(*[score_one(r) for r in results])
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return ranked[:max_results]


def _cached_to_search_result(cached: CachedSearchResult) -> VideoSearchResult:
    """Convert cached search result to VideoSearchResult."""
    return VideoSearchResult(
        video_id=cached.video_id,
        title=cached.title,
        channel_name=cached.channel_name,
        duration=cached.duration,
        view_count=cached.view_count,
    )


async def search_and_get_transcripts(
    query: str,
    max_results: int = 5,
    language: str = "en",
    use_cache: bool = True,
    output_format: str = "clean",
    summary_model: str | None = None,
) -> list[tuple[VideoSearchResult, Optional[ServiceResult]]]:
    """Search YouTube and fetch transcripts for results.

    Args:
        query: Search query string.
        max_results: Maximum number of search results.
        language: Language for transcripts.
        use_cache: Whether to use cache.
        output_format: 'clean' (default), 'text', 'json', 'srt', 'vtt', or 'summary'.
        summary_model: Override the local summary model (only for 'summary').

    Returns:
        List of (search_result, transcript_or_none) tuples.
    """
    results = await search(query, max_results=max_results, use_cache=use_cache)

    # Fetch transcripts in parallel
    async def fetch_one(video_result: VideoSearchResult):
        try:
            transcript = await get_transcript(
                video_result.video_id,
                language=language,
                output_format=output_format,
                use_cache=use_cache,
                summary_model=summary_model,
            )
            return transcript
        except Exception:
            return None

    transcripts = await asyncio.gather(*[fetch_one(r) for r in results])

    return list(zip(results, transcripts))
