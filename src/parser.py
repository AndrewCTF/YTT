"""Parse caption tracks and timed text from YouTube Innertube responses."""

import json
import re
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimedText:
    """A single timed text segment."""
    start_ms: int
    duration_ms: int
    text: str

    @property
    def start(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end(self) -> float:
        return (self.start_ms + self.duration_ms) / 1000.0

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass
class CaptionTrack:
    """A caption track with language and base URL."""
    language: str
    language_code: str
    base_url: str
    is_generated: bool = False  # auto-generated captions


def parse_player_response(response_data: dict) -> list[CaptionTrack]:
    """Extract caption tracks from the Innertube player response.

    Args:
        response_data: The parsed JSON response from the player endpoint.

    Returns:
        List of CaptionTrack objects found in the response.
    """
    tracks = []

    # Navigate to caption tracks in the response
    # Structure: response.captions.playerCaptionsRenderer.captionTracks
    captions = response_data.get("captions", {})
    renderer = captions.get("playerCaptionsRenderer", {})
    caption_tracks = renderer.get("captionTracks", [])

    for track_data in caption_tracks:
        base_url = track_data.get("baseUrl", "")
        if not base_url:
            continue

        # Extract language info
        language = track_data.get("languageName", {}).get("simpleText", "unknown")
        language_code = track_data.get("languageCode", "en")
        is_generated = track_data.get("kind", "") == "asr"

        tracks.append(CaptionTrack(
            language=language,
            language_code=language_code,
            base_url=base_url,
            is_generated=is_generated,
        ))

    return tracks


def parse_json3_caption_data(caption_data: bytes | str) -> list[TimedText]:
    """Parse JSON3 format caption data from YouTube.

    YouTube returns caption data in a custom JSON3 format like:
    {
      "events": [
        {
          "tStartMs": 0,
          "dDurationMs": 4000,
          "segs": [{ "utf8": "Hello world" }]
        }
      ]
    }

    Args:
        caption_data: Raw caption data (bytes or str).

    Returns:
        List of TimedText segments in chronological order.
    """
    if isinstance(caption_data, bytes):
        caption_data = caption_data.decode("utf-8")

    # Some JSON3 data has trailing commas which breaks stdlib json
    # Handle common patterns
    try:
        data = json.loads(caption_data)
    except json.JSONDecodeError:
        # Try to fix common JSON3 issues
        cleaned = caption_data.replace(',}', '}').replace(',]', ']')
        data = json.loads(cleaned)

    events = data.get("events", [])
    segments = []

    for event in events:
        t_start_ms = event.get("tStartMs", 0)
        d_duration_ms = event.get("dDurationMs", 0)

        # Concatenate all text segments in this event
        segs = event.get("segs", [])
        text_parts = []
        for seg in segs:
            if isinstance(seg, dict):
                text_parts.append(seg.get("utf8", ""))
            elif isinstance(seg, str):
                text_parts.append(seg)

        text = "".join(text_parts).strip()
        if text:
            segments.append(TimedText(
                start_ms=t_start_ms,
                duration_ms=d_duration_ms,
                text=text,
            ))

    return segments


def extract_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats.

    Handles:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID

    Args:
        url: A YouTube URL or video ID.

    Returns:
        The video ID if found, None otherwise.
    """
    # If it's already just a video ID (11 chars)
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url

    # Patterns for different URL formats
    patterns = [
        r'(?:v=|/v/|/embed/)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None
