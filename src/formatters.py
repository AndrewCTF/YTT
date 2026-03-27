"""Format transcript data into various output formats."""

import json
from dataclasses import dataclass

from .fetcher import TranscriptData
from .parser import TimedText
from .whisper_runner import WhisperResult, WhisperSegment


def format_transcript_text(
    transcript: TranscriptData | WhisperResult,
    include_metadata: bool = False,
) -> str:
    """Format transcript as plain text.

    Args:
        transcript: Transcript data from Innertube or Whisper.
        include_metadata: Whether to include video title and language.

    Returns:
        Plain text transcript.
    """
    lines = []

    if include_metadata:
        if isinstance(transcript, TranscriptData):
            lines.append(f"Title: {transcript.title}")
            lines.append(f"Language: {transcript.language}")
            lines.append(f"Video ID: {transcript.video_id}")
            lines.append(f"Source: {transcript.source}")
            lines.append("")

    # Collect all text segments
    if isinstance(transcript, TranscriptData):
        segments = transcript.segments
    else:
        segments = transcript.segments

    for seg in segments:
        if isinstance(seg, TimedText):
            text = seg.text
        else:
            text = seg.text
        lines.append(text)

    return "\n".join(lines)


def format_transcript_json(
    transcript: TranscriptData | WhisperResult,
    include_words: bool = False,
) -> str:
    """Format transcript as JSON.

    Args:
        transcript: Transcript data from Innertube or Whisper.
        include_words: Whether to include word-level timestamps (Whisper only).

    Returns:
        JSON string with transcript data.
    """
    if isinstance(transcript, TranscriptData):
        data = {
            "video_id": transcript.video_id,
            "title": transcript.title,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "source": transcript.source,
            "is_generated": transcript.is_generated,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ],
        }
    else:
        data = {
            "video_id": transcript.video_id,
            "language": transcript.language,
            "source": "whisper",
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ],
        }

    return json.dumps(data, indent=2, ensure_ascii=False)


def format_transcript_srt(transcript: TranscriptData | WhisperResult) -> str:
    """Format transcript as SRT (SubRip) format.

    SRT format:
    1
    00:00:00,000 --> 00:00:04,000
    Hello world

    Args:
        transcript: Transcript data from Innertube or Whisper.

    Returns:
        SRT formatted string.
    """
    lines = []

    if isinstance(transcript, TranscriptData):
        segments = transcript.segments
    else:
        segments = transcript.segments

    for i, seg in enumerate(segments, 1):
        start = seg.start
        end = seg.end

        # Format as HH:MM:SS,mmm
        start_str = format_timestamp_srt(start)
        end_str = format_timestamp_srt(end)

        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def format_transcript_vtt(transcript: TranscriptData | WhisperResult) -> str:
    """Format transcript as WebVTT format.

    VTT format:
    WEBVTT

    00:00:00.000 --> 00:00:04.000
    Hello world

    Args:
        transcript: Transcript data from Innertube or Whisper.

    Returns:
        VTT formatted string.
    """
    lines = ["WEBVTT", ""]

    if isinstance(transcript, TranscriptData):
        segments = transcript.segments
    else:
        segments = transcript.segments

    for seg in segments:
        start = seg.start
        end = seg.end

        # Format as HH:MM:SS.mmm
        start_str = format_timestamp_vtt(start)
        end_str = format_timestamp_vtt(end)

        lines.append(f"{start_str} --> {end_str}")
        lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT (SubRip) format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string as HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for WebVTT format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string as HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_transcript(
    transcript: TranscriptData | WhisperResult,
    format: str = "text",
) -> str:
    """Format transcript in the specified format.

    Args:
        transcript: Transcript data from Innertube or Whisper.
        format: Output format - 'text', 'json', 'srt', or 'vtt'.

    Returns:
        Formatted transcript string.

    Raises:
        ValueError: If format is not supported.
    """
    format = format.lower().strip()

    if format == "text":
        return format_transcript_text(transcript)
    elif format == "json":
        return format_transcript_json(transcript)
    elif format == "srt":
        return format_transcript_srt(transcript)
    elif format == "vtt":
        return format_transcript_vtt(transcript)
    else:
        raise ValueError(f"Unsupported format: {format}. Use: text, json, srt, vtt")