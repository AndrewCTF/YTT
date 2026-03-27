"""Fetch transcripts from YouTube's Innertube API."""

import re
import time
from dataclasses import dataclass

import requests

from config import config
from .exceptions import (
    ExtractionError,
    NoTranscriptFound,
    RateLimitError,
    VideoUnavailable,
)
from .parser import (
    CaptionTrack,
    TimedText,
    extract_video_id,
    parse_json3_caption_data,
    parse_player_response,
)


@dataclass
class TranscriptData:
    """Container for transcript data from Innertube."""
    video_id: str
    title: str
    language: str
    language_code: str
    segments: list[TimedText]
    source: str = "innertube"
    is_generated: bool = False


def extract_api_key(video_page_html: str) -> str:
    """Extract the INNERTUBE_API_KEY from video page HTML.

    The API key is embedded in a JavaScript variable in the page HTML.
    Pattern: INNERTUBE_API_KEY":"{api_key}"

    Args:
        video_page_html: Raw HTML from the video page.

    Returns:
        The extracted API key.

    Raises:
        ExtractionError: If the API key cannot be found.
    """
    # Try primary pattern - find INNERTUBE_API_KEY and capture the quoted value
    pattern = r'INNERTUBE_API_KEY["\s]*[:=]["\s]*([A-Za-z0-9_-]+)'
    match = re.search(pattern, video_page_html)
    if match:
        return match.group(1)

    # Try secondary pattern
    pattern2 = r'"INNERTUBE_API_KEY"\s*,\s*\{[^}]*?"apiKey"\s*:\s*"([^"]+)"'
    match2 = re.search(pattern2, video_page_html)
    if match2:
        return match2.group(1)

    raise ExtractionError("Could not find INNERTUBE_API_KEY in video page")


def get_video_page(video_id: str, session: requests.Session | None = None) -> str:
    """Fetch the YouTube video page HTML.

    Args:
        video_id: The YouTube video ID.
        session: Optional requests session for connection pooling.

    Returns:
        Raw HTML content of the video page.

    Raises:
        VideoUnavailable: If the video does not exist.
    """
    url = f"{config.VIDEO_PAGE_URL}?v={video_id}"

    if session is None:
        session = requests.Session()

    response = session.get(url, timeout=10)

    if response.status_code == 404:
        raise VideoUnavailable(f"Video {video_id} not found")

    response.raise_for_status()
    return response.text


def fetch_player_response(
    video_id: str,
    api_key: str,
    session: requests.Session | None = None,
) -> dict:
    """Fetch the player endpoint response.

    This is an internal YouTube API that returns video metadata including
    available caption tracks.

    Args:
        video_id: The YouTube video ID.
        api_key: The Innertube API key extracted from the video page.
        session: Optional requests session.

    Returns:
        Parsed JSON response from the player endpoint.

    Raises:
        RateLimitError: If YouTube rate limits the request.
        VideoUnavailable: If the video is not available.
        ExtractionError: If the response cannot be parsed.
    """
    url = f"{config.INNERTUBE_API_URL}?key={api_key}"

    payload = {
        "context": {
            "client": {
                "clientName": config.INNERTUBE_CLIENT,
                "clientVersion": config.INNERTUBE_CLIENT_VERSION,
            }
        },
        "videoId": video_id,
    }

    if session is None:
        session = requests.Session()

    response = session.post(url, json=payload, timeout=10)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 10))
        raise RateLimitError(f"Rate limited, retry after {retry_after}s", retry_after=retry_after)

    if response.status_code == 403:
        raise VideoUnavailable(f"Video {video_id} is not available (403 Forbidden)")

    response.raise_for_status()

    try:
        return response.json()
    except ValueError as e:
        raise ExtractionError(f"Failed to parse player response: {e}")


def fetch_caption_data(
    track: CaptionTrack,
    session: requests.Session | None = None,
) -> list[TimedText]:
    """Fetch caption data from a caption track URL.

    Args:
        track: The caption track to fetch.
        session: Optional requests session.

    Returns:
        List of timed text segments.

    Raises:
        ExtractionError: If caption data cannot be fetched.
    """
    # Add fmt=json3 for easier parsing
    base_url = track.base_url
    if "&fmt=json3" not in base_url and "fmt=json3" not in base_url:
        base_url = base_url + ("&" if "?" in base_url else "?") + "fmt=json3"

    if session is None:
        session = requests.Session()

    response = session.get(base_url, timeout=15)
    response.raise_for_status()

    return parse_json3_caption_data(response.content)


def fetch_transcript_innertube(
    video_id: str,
    language: str = "en",
    session: requests.Session | None = None,
) -> TranscriptData:
    """Fetch transcript using the Innertube API.

    This is the primary fetch method. It:
    1. Gets the video page to extract the API key
    2. POSTs to the player endpoint
    3. Finds the best matching caption track
    4. Fetches the caption data

    Args:
        video_id: The YouTube video ID or URL.
        language: Preferred language code (e.g., 'en', 'es', 'ja').
        session: Optional requests session.

    Returns:
        TranscriptData with segments and metadata.

    Raises:
        VideoUnavailable: If the video doesn't exist.
        NoTranscriptFound: If no transcript is available.
        RateLimitError: If rate limited by YouTube.
    """
    # Extract video ID if full URL provided
    actual_video_id = extract_video_id(video_id)
    if actual_video_id is None:
        raise ExtractionError(f"Invalid video ID or URL: {video_id}")

    video_id = actual_video_id

    if session is None:
        session = requests.Session()

    # Step 1: Get video page and extract API key
    html = get_video_page(video_id, session)
    api_key = extract_api_key(html)

    # Step 2: Get player response with caption tracks
    player_response = fetch_player_response(video_id, api_key, session)

    # Extract title from player response
    title = "Unknown"
    video_details = player_response.get("videoDetails", {})
    if video_details:
        title = video_details.get("title", title)

    # Step 3: Find matching caption tracks
    caption_tracks = parse_player_response(player_response)

    if not caption_tracks:
        raise NoTranscriptFound(f"No caption tracks found for video {video_id}")

    # Find the best matching track
    # Priority: exact language match > generated captions > first available
    selected_track: CaptionTrack | None = None

    # First try exact language match
    for track in caption_tracks:
        if track.language_code == language:
            selected_track = track
            break

    # Fall back to language without region code
    if selected_track is None:
        lang_prefix = language.split("-")[0]
        for track in caption_tracks:
            if track.language_code.split("-")[0] == lang_prefix:
                selected_track = track
                break

    # Last resort: first available track
    if selected_track is None:
        selected_track = caption_tracks[0]

    # Step 4: Fetch caption data
    segments = fetch_caption_data(selected_track, session)

    if not segments:
        raise NoTranscriptFound(f"Empty transcript for video {video_id}")

    return TranscriptData(
        video_id=video_id,
        title=title,
        language=selected_track.language,
        language_code=selected_track.language_code,
        segments=segments,
        source="innertube",
        is_generated=selected_track.is_generated,
    )
