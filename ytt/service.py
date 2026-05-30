"""High-level service orchestrator for transcript fetching.

Strategy (captions-first — fast, light, rate-limit resistant):

1. Check the cache.
2. Fetch YouTube's own captions via the Innertube API (a few KB, no audio
   download) with retry/backoff and multi-client fallback.
3. Only if captions are genuinely unavailable, fall back to local Whisper
   transcription (downloads audio; requires the ``whisper`` extra).
4. Format (``clean`` strips duplication/timestamps for LLM ingestion) and cache.
"""

import asyncio
from dataclasses import dataclass, field
from functools import partial

from .cache import cache, CachedTranscript
from .config import config
from .embeddings import get_embedder
from .exceptions import NoTranscriptFound, YouTubeTranscriptError
from .fetcher import (
    CaptionListing,
    TranscriptData,
    fetch_transcript_innertube,
    fetch_video_metadata,
    list_caption_tracks,
)
from .formatters import format_transcript
from .http import new_session
from .index import CorpusIndex
from .parser import VideoMetadata, extract_video_id
from .rate_limiter import rate_limiter
from .semantic import Passage, search_transcript as _semantic_search
from .summarizer import generate as _llm_generate, summarize_text
from .whisper_runner import WhisperResult, fetch_transcript_whisper


@dataclass
class ServiceResult:
    """Result from the transcript service."""

    video_id: str
    title: str
    language: str
    source: str  # 'innertube' or 'whisper'
    is_generated: bool
    content: str  # Formatted transcript content


@dataclass
class SearchInVideoResult:
    """Ranked passages from semantic search within one video."""

    video_id: str
    title: str
    query: str
    passages: list[Passage] = field(default_factory=list)


@dataclass
class AskResult:
    """A grounded answer plus the cited transcript passages it drew from."""

    video_id: str
    title: str
    question: str
    answer: str | None
    passages: list[Passage] = field(default_factory=list)
    llm_used: bool = False
    note: str = ""


async def _render(
    transcript: TranscriptData,
    output_format: str,
    summary_model: str | None = None,
    summary_provider: str | None = None,
) -> str:
    """Format a transcript, handling the network-backed ``summary`` format.

    ``summary`` cleans the transcript then runs a local LLM (off-thread) to
    produce a short summary — the token-saving path. All other formats are pure.
    The model/provider are passed as arguments (never via shared global state),
    so concurrent requests with different overrides don't interfere.
    """
    if output_format == "summary":
        clean = format_transcript(transcript, "clean")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(summarize_text, clean, model=summary_model, provider=summary_provider),
        )
    return format_transcript(transcript, output_format)


async def _fetch_captions(
    video_id: str, language: str, translate: str | None = None
) -> TranscriptData:
    """Fetch captions in a worker thread, rate-limited and session-pooled."""
    await rate_limiter.acquire()
    loop = asyncio.get_event_loop()

    def _run() -> TranscriptData:
        session = new_session()
        try:
            return fetch_transcript_innertube(video_id, language, session, translate=translate)
        finally:
            session.close()

    return await loop.run_in_executor(None, _run)


async def get_transcript_data(
    video_id_or_url: str,
    language: str = "en",
    translate: str | None = None,
    use_cache: bool = True,
    use_whisper_fallback: bool = True,
) -> tuple[TranscriptData, str]:
    """Fetch raw transcript data (segments + metadata) with caching.

    Shared by :func:`get_transcript` and the semantic-search helpers so the
    captions-first / Whisper-fallback / cache logic lives in exactly one place.
    Returns ``(transcript, source)`` where source is ``'innertube'``/``'whisper'``.
    """
    video_id = extract_video_id(video_id_or_url)
    if video_id is None:
        raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")

    # Translated tracks are cached under the target language so re-fetch hits.
    cache_lang = translate or language

    if use_cache:
        cached = await cache.get(video_id, cache_lang)
        if cached:
            return _cached_to_transcript_data(cached), cached.source

    transcript: TranscriptData | None = None
    source = "innertube"
    captions_error: Exception | None = None
    try:
        transcript = await _fetch_captions(video_id, language, translate)
    except Exception as err:
        captions_error = err

    if transcript is None:
        if not use_whisper_fallback:
            raise captions_error or NoTranscriptFound(f"No captions found for {video_id}")
        try:
            loop = asyncio.get_event_loop()
            whisper_result = await loop.run_in_executor(None, fetch_transcript_whisper, video_id)
            transcript = _whisper_to_transcript_data(whisper_result)
            source = "whisper"
        except Exception as whisper_err:
            raise YouTubeTranscriptError(
                f"Captions failed ({captions_error}); Whisper fallback failed ({whisper_err})"
            )

    if use_cache:
        await cache.set(video_id, cache_lang, _transcript_to_cache_dict(transcript), source=source)
    return transcript, source


async def get_transcript(
    video_id_or_url: str,
    language: str = "en",
    output_format: str = "text",
    use_cache: bool = True,
    use_whisper_fallback: bool = True,
    summary_model: str | None = None,
    summary_provider: str | None = None,
    translate: str | None = None,
) -> ServiceResult:
    """Get a YouTube video transcript.

    Args:
        video_id_or_url: YouTube video ID or URL.
        language: Preferred language code (e.g., 'en', 'es').
        output_format: 'clean', 'text', 'json', 'srt', 'vtt', or 'summary'.
            'clean' is recommended for LLM ingestion (deduplicated, no timestamps).
            'summary' runs a local LLM (requires a reachable provider; raises
            SummarizerError otherwise).
        use_cache: Whether to use the cache (default: True).
        use_whisper_fallback: Fall back to local Whisper if no captions exist
            (default: True; requires the ``whisper`` extra).
        summary_model: Override the local summary model (only for 'summary').
        summary_provider: Override the summary provider (only for 'summary').
        translate: Optional target language to machine-translate captions into
            (server-side, no local model needed).

    Returns:
        ServiceResult with formatted transcript content.

    Raises:
        YouTubeTranscriptError: If both captions and Whisper fail.
        ValueError: If the video ID/URL is invalid.
    """
    transcript, source = await get_transcript_data(
        video_id_or_url, language, translate, use_cache, use_whisper_fallback
    )
    return ServiceResult(
        video_id=transcript.video_id,
        title=transcript.title,
        language=transcript.language,
        source=source,
        is_generated=transcript.is_generated,
        content=await _render(transcript, output_format, summary_model, summary_provider),
    )


async def get_transcripts_batch(
    video_ids: list[str],
    language: str = "en",
    output_format: str = "text",
    max_workers: int | None = None,
    summary_model: str | None = None,
    summary_provider: str | None = None,
) -> list[ServiceResult | Exception]:
    """Get transcripts for multiple videos concurrently.

    Args:
        video_ids: List of YouTube video IDs or URLs.
        language: Preferred language code (e.g., 'en', 'es').
        output_format: 'clean', 'text', 'json', 'srt', 'vtt', or 'summary'.
        max_workers: Max concurrent transcriptions. Defaults to config.MAX_CONCURRENT_WORKERS.
        summary_model: Override the local summary model (only for 'summary').
        summary_provider: Override the summary provider (only for 'summary').

    Returns:
        List of (ServiceResult | Exception) - one per video_id.
    """
    if max_workers is None:
        max_workers = config.MAX_CONCURRENT_WORKERS

    semaphore = asyncio.Semaphore(max_workers)

    async def fetch_one(vid: str) -> ServiceResult | Exception:
        async with semaphore:
            try:
                return await get_transcript(
                    vid,
                    language=language,
                    output_format=output_format,
                    summary_model=summary_model,
                    summary_provider=summary_provider,
                )
            except Exception as e:
                return e

    return await asyncio.gather(*[fetch_one(vid) for vid in video_ids])


_RAG_PROMPT = (
    "You are answering a question about a YouTube video using ONLY the transcript "
    "excerpts below. Each excerpt is tagged with its timestamp. Answer concisely and "
    "cite the timestamps you rely on, like [12:34]. If the excerpts do not contain the "
    "answer, say so plainly instead of guessing.\n\n"
    "Question: {question}\n\nTranscript excerpts:\n{context}\n\nAnswer:"
)


async def get_video_info(video_id_or_url: str) -> VideoMetadata:
    """Fetch rich, audio-free video metadata (title, channel, views, chapters)."""
    video_id = extract_video_id(video_id_or_url)
    if video_id is None:
        raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")
    await rate_limiter.acquire()
    loop = asyncio.get_event_loop()

    def _run() -> VideoMetadata:
        session = new_session()
        try:
            return fetch_video_metadata(video_id, session)
        finally:
            session.close()

    return await loop.run_in_executor(None, _run)


async def list_languages(video_id_or_url: str) -> CaptionListing:
    """List all caption tracks + translation targets (like ``yt-dlp --list-subs``)."""
    video_id = extract_video_id(video_id_or_url)
    if video_id is None:
        raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")
    await rate_limiter.acquire()
    loop = asyncio.get_event_loop()

    def _run() -> CaptionListing:
        session = new_session()
        try:
            return list_caption_tracks(video_id, session)
        finally:
            session.close()

    return await loop.run_in_executor(None, _run)


async def search_in_video(
    video_id_or_url: str,
    query: str,
    top_k: int = 5,
    language: str = "en",
    translate: str | None = None,
    use_cache: bool = True,
    hybrid: bool = True,
    embed_provider: str | None = None,
    embed_model: str | None = None,
) -> SearchInVideoResult:
    """Semantic search *inside* a single video → ranked timestamped passages."""
    transcript, _ = await get_transcript_data(video_id_or_url, language, translate, use_cache)
    loop = asyncio.get_event_loop()

    def _run() -> list[Passage]:
        embedder = None
        if hybrid:
            try:
                embedder = get_embedder(embed_provider, embed_model)
            except Exception:
                embedder = None
        return _semantic_search(
            transcript.segments,
            query,
            top_k=top_k,
            embedder=embedder,
            hybrid=hybrid,
            mmr=True,
            video_id=transcript.video_id,
            title=transcript.title,
        )

    passages = await loop.run_in_executor(None, _run)
    return SearchInVideoResult(
        video_id=transcript.video_id, title=transcript.title, query=query, passages=passages
    )


async def ask_video(
    video_id_or_url: str,
    question: str,
    top_k: int = 6,
    language: str = "en",
    use_cache: bool = True,
    answer: bool = True,
    model: str | None = None,
    provider: str | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
) -> AskResult:
    """Retrieval-augmented Q&A over one video (fully local).

    Retrieves the most relevant timestamped passages, then — if a local LLM is
    reachable and ``answer`` is set — generates a grounded, citation-bearing
    answer. If no LLM is available, the cited passages are still returned, so the
    feature degrades to exa-style "here are the relevant moments".
    """
    res = await search_in_video(
        video_id_or_url,
        question,
        top_k=top_k,
        language=language,
        use_cache=use_cache,
        hybrid=True,
        embed_provider=embed_provider,
        embed_model=embed_model,
    )
    passages = res.passages
    if not passages:
        return AskResult(
            video_id=res.video_id,
            title=res.title,
            question=question,
            answer=None,
            passages=[],
            llm_used=False,
            note="No relevant passages found in the transcript.",
        )
    if not answer:
        return AskResult(
            video_id=res.video_id,
            title=res.title,
            question=question,
            answer=None,
            passages=passages,
            llm_used=False,
            note="Answer generation disabled; returning passages only.",
        )

    context = "\n".join(f"[{p.timestamp}] {p.text}" for p in passages)
    prompt = _RAG_PROMPT.format(question=question, context=context)
    loop = asyncio.get_event_loop()
    try:
        ans = await loop.run_in_executor(None, partial(_llm_generate, prompt, model, provider))
        return AskResult(
            video_id=res.video_id,
            title=res.title,
            question=question,
            answer=ans,
            passages=passages,
            llm_used=True,
        )
    except Exception as e:
        return AskResult(
            video_id=res.video_id,
            title=res.title,
            question=question,
            answer=None,
            passages=passages,
            llm_used=False,
            note=f"Local LLM unavailable ({e}); returning cited passages only.",
        )


async def index_videos(
    video_ids: list[str],
    language: str = "en",
    use_cache: bool = True,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Add videos to the local corpus index for cross-video semantic search."""
    indexed: list[dict] = []
    failed: dict[str, str] = {}
    loop = asyncio.get_event_loop()

    for vid in video_ids:
        try:
            transcript, _ = await get_transcript_data(vid, language, None, use_cache)

            def _run(t: TranscriptData) -> tuple[int, str]:
                embedder = get_embedder(embed_provider, embed_model)
                with CorpusIndex(db_path) as idx:
                    n = idx.add_video(
                        t.video_id, t.title, t.segments, language=language, embedder=embedder
                    )
                return n, embedder.name

            n_chunks, ename = await loop.run_in_executor(None, partial(_run, transcript))
            indexed.append(
                {
                    "video_id": transcript.video_id,
                    "title": transcript.title,
                    "chunks": n_chunks,
                    "embedder": ename,
                }
            )
        except Exception as e:
            failed[vid] = str(e)

    return {"indexed": indexed, "failed": failed}


async def find_in_corpus(
    query: str,
    top_k: int = 8,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    db_path: str | None = None,
    hybrid: bool = True,
) -> list[Passage]:
    """Semantic search across every video in the local corpus index."""
    loop = asyncio.get_event_loop()

    def _run() -> list[Passage]:
        embedder = None
        if hybrid:
            try:
                embedder = get_embedder(embed_provider, embed_model)
            except Exception:
                embedder = None
        with CorpusIndex(db_path) as idx:
            return idx.search(query, top_k=top_k, embedder=embedder, hybrid=hybrid)

    return await loop.run_in_executor(None, _run)


async def corpus_stats(db_path: str | None = None) -> dict:
    """Stats for the local corpus index (videos, chunks, embedders)."""
    loop = asyncio.get_event_loop()

    def _run() -> dict:
        with CorpusIndex(db_path) as idx:
            stats = idx.stats()
            stats["video_list"] = [
                {"video_id": v.video_id, "title": v.title, "chunks": v.n_chunks}
                for v in idx.list_videos()
            ]
            return stats

    return await loop.run_in_executor(None, _run)


def _transcript_to_cache_dict(transcript: TranscriptData) -> dict:
    """Convert TranscriptData to cacheable dict."""
    return {
        "video_id": transcript.video_id,
        "title": transcript.title,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "source": transcript.source,
        "is_generated": transcript.is_generated,
        "segments": [
            {
                "start_ms": seg.start_ms,
                "duration_ms": seg.duration_ms,
                "text": seg.text,
            }
            for seg in transcript.segments
        ],
    }


def _whisper_to_transcript_data(result: WhisperResult) -> TranscriptData:
    """Convert WhisperResult to TranscriptData for uniform handling."""
    from .parser import TimedText

    return TranscriptData(
        video_id=result.video_id,
        title="",
        language=result.language,
        language_code=result.language,
        segments=[
            TimedText(
                start_ms=int(seg.start * 1000),
                duration_ms=int((seg.end - seg.start) * 1000),
                text=seg.text,
            )
            for seg in result.segments
        ],
        source="whisper",
        is_generated=False,
    )


def _cached_to_transcript_data(cached: CachedTranscript) -> TranscriptData:
    """Reconstruct TranscriptData from cached dict."""
    from .parser import TimedText

    raw = cached.raw_data
    return TranscriptData(
        video_id=raw["video_id"],
        title=raw.get("title", ""),
        language=raw.get("language", cached.language),
        language_code=raw.get("language_code", cached.language),
        source=raw.get("source", cached.source),
        is_generated=raw.get("is_generated", False),
        segments=[
            TimedText(
                start_ms=seg["start_ms"],
                duration_ms=seg["duration_ms"],
                text=seg["text"],
            )
            for seg in raw.get("segments", [])
        ],
    )
