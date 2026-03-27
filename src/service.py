"""High-level service orchestrator for transcript fetching."""

import asyncio
from dataclasses import dataclass

from .cache import cache, CachedTranscript
from .exceptions import WhisperError
from .fetcher import TranscriptData, fetch_transcript_innertube
from .formatters import format_transcript
from .rate_limiter import rate_limiter
from .whisper_runner import WhisperResult, fetch_transcript_whisper


@dataclass
class ServiceResult:
    """Result from the transcript service."""
    video_id: str
    title: str
    language: str
    source: str           # 'innertube' or 'whisper'
    is_generated: bool
    content: str          # Formatted transcript content


async def get_transcript(
    video_id_or_url: str,
    language: str = "en",
    output_format: str = "text",
    use_cache: bool = True,
    use_whisper_fallback: bool = True,
) -> ServiceResult:
    """Get a YouTube video transcript.

    This is the main entry point for the transcript service. It:
    1. Checks the cache first
    2. Tries Whisper (fast, accurate, no rate limits)
    3. Falls back to Innertube API on Whisper failure
    4. Formats and returns the result

    Args:
        video_id_or_url: YouTube video ID or URL.
        language: Preferred language code (e.g., 'en', 'es').
        output_format: Output format - 'text', 'json', 'srt', 'vtt'.
        use_cache: Whether to use the cache (default: True).
        use_whisper_fallback: If False, don't fall back to Innertube when Whisper fails (default: True).

    Returns:
        ServiceResult with formatted transcript content.

    Raises:
        WhisperError: If both Whisper and Innertube fail, or Whisper fails and fallback is disabled.
    """
    from .parser import extract_video_id

    # Extract video ID if URL provided
    video_id = extract_video_id(video_id_or_url)
    if video_id is None:
        raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")

    # Step 1: Check cache
    if use_cache:
        cached = await cache.get(video_id, language)
        if cached:
            # Reconstruct transcript from cached data
            if cached.source == "whisper":
                transcript = _cached_to_whisper_result(cached)
            else:
                transcript = _cached_to_transcript_data(cached)

            content = format_transcript(transcript, output_format)
            return ServiceResult(
                video_id=video_id,
                title=transcript.title if hasattr(transcript, 'title') else "",
                language=transcript.language if hasattr(transcript, 'language') else language,
                source=cached.source,
                is_generated=False,
                content=content,
            )

    # Step 2: Try Whisper first (fast, accurate, no rate limits)
    transcript = None
    used_whisper = False

    try:
        loop = asyncio.get_event_loop()
        whisper_result = await loop.run_in_executor(
            None,
            fetch_transcript_whisper,
            video_id,
        )

        # Convert Whisper result to TranscriptData format for caching
        transcript = _whisper_to_transcript_data(whisper_result)
        used_whisper = True

    except Exception as whisper_err:
        # Whisper failed, try Innertube as fallback
        if not use_whisper_fallback:
            raise WhisperError(f"Whisper failed: {whisper_err}")

        try:
            await rate_limiter.acquire()
            transcript = await loop.run_in_executor(
                None,
                fetch_transcript_innertube,
                video_id,
                language,
                None,
            )
        except Exception as innertube_err:
            raise WhisperError(
                f"Both Whisper ({whisper_err}) and Innertube ({innertube_err}) failed"
            )

    # Step 3: Cache the result
    if use_cache:
        await cache.set(
            video_id,
            language,
            _transcript_to_cache_dict(transcript),
            source="whisper" if used_whisper else "innertube",
        )

    # Step 4: Format and return
    content = format_transcript(transcript, output_format)

    return ServiceResult(
        video_id=video_id,
        title=transcript.title if hasattr(transcript, 'title') else "",
        language=transcript.language if hasattr(transcript, 'language') else language,
        source="whisper" if used_whisper else "innertube",
        is_generated=transcript.is_generated if hasattr(transcript, 'is_generated') else False,
        content=content,
    )


async def get_transcripts_batch(
    video_ids: list[str],
    language: str = "en",
    output_format: str = "text",
    max_workers: int | None = None,
) -> list[ServiceResult | Exception]:
    """Get transcripts for multiple videos concurrently.

    Args:
        video_ids: List of YouTube video IDs or URLs.
        language: Preferred language code (e.g., 'en', 'es').
        output_format: Output format - 'text', 'json', 'srt', 'vtt'.
        max_workers: Max concurrent transcriptions. Defaults to config.MAX_CONCURRENT_WORKERS.

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


def _cached_to_whisper_result(cached: CachedTranscript) -> WhisperResult:
    """Reconstruct WhisperResult from cached dict."""
    from .whisper_runner import WhisperSegment

    raw = cached.raw_data
    segments = []
    for seg in raw.get("segments", []):
        # Handle both start/end (whisper native) and start_ms/duration_ms (cached) formats
        if "start" in seg and "end" in seg:
            start = seg["start"]
            end = seg["end"]
        else:
            start = seg["start_ms"] / 1000.0
            end = start + seg["duration_ms"] / 1000.0
        segments.append(WhisperSegment(
            start=start,
            end=end,
            text=seg["text"],
        ))
    return WhisperResult(
        video_id=raw["video_id"],
        language=raw.get("language", cached.language),
        segments=segments,
        text=" ".join(seg["text"] for seg in raw.get("segments", [])),
    )