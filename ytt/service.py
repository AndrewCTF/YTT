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
from dataclasses import dataclass
from functools import partial

from .cache import cache, CachedTranscript
from .config import config
from .exceptions import NoTranscriptFound, YouTubeTranscriptError
from .fetcher import TranscriptData, fetch_transcript_innertube
from .formatters import format_transcript
from .http import new_session
from .parser import extract_video_id
from .rate_limiter import rate_limiter
from .summarizer import summarize_text
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


async def _fetch_captions(video_id: str, language: str) -> TranscriptData:
    """Fetch captions in a worker thread, rate-limited and session-pooled."""
    await rate_limiter.acquire()
    loop = asyncio.get_event_loop()

    def _run() -> TranscriptData:
        session = new_session()
        try:
            return fetch_transcript_innertube(video_id, language, session)
        finally:
            session.close()

    return await loop.run_in_executor(None, _run)


async def get_transcript(
    video_id_or_url: str,
    language: str = "en",
    output_format: str = "text",
    use_cache: bool = True,
    use_whisper_fallback: bool = True,
    summary_model: str | None = None,
    summary_provider: str | None = None,
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

    Returns:
        ServiceResult with formatted transcript content.

    Raises:
        YouTubeTranscriptError: If both captions and Whisper fail.
        ValueError: If the video ID/URL is invalid.
    """
    video_id = extract_video_id(video_id_or_url)
    if video_id is None:
        raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")

    # Step 1: Cache.
    if use_cache:
        cached = await cache.get(video_id, language)
        if cached:
            transcript = _cached_to_transcript_data(cached)
            return ServiceResult(
                video_id=video_id,
                title=transcript.title,
                language=transcript.language,
                source=cached.source,
                is_generated=transcript.is_generated,
                content=await _render(transcript, output_format, summary_model, summary_provider),
            )

    # Step 2: Captions first (fast, light, rate-limit resistant).
    transcript: TranscriptData | None = None
    source = "innertube"
    captions_error: Exception | None = None

    try:
        transcript = await _fetch_captions(video_id, language)
    except Exception as err:
        captions_error = err

    # Step 3: Whisper fallback only when captions are unavailable.
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
                f"Captions failed ({captions_error}); " f"Whisper fallback failed ({whisper_err})"
            )

    # Step 4: Cache.
    if use_cache:
        await cache.set(
            video_id,
            language,
            _transcript_to_cache_dict(transcript),
            source=source,
        )

    # Step 5: Format and return.
    return ServiceResult(
        video_id=video_id,
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
