"""Fetch transcripts from YouTube's Innertube API (captions path).

This is the primary, low-cost path: it pulls YouTube's own caption tracks
through the Innertube ``player`` endpoint. It is fast (a few KB per video) and
LLM-friendly, and — with a hardcoded API key, realistic per-client headers and
retry/backoff in :mod:`ytt.http` — resilient to rate limiting.
"""

import json
import re
from dataclasses import dataclass

import requests

from .config import BROWSER_USER_AGENT, INNERTUBE_CLIENTS, config
from .exceptions import (
    ExtractionError,
    NoTranscriptFound,
    VideoUnavailable,
)
from .http import request
from .parser import (
    CaptionTrack,
    TimedText,
    extract_video_id,
    parse_player_response,
    parse_timedtext,
)


@dataclass
class TranscriptData:
    """Container for transcript data from Innertube captions."""

    video_id: str
    title: str
    language: str
    language_code: str
    segments: list[TimedText]
    source: str = "innertube"
    is_generated: bool = False


def fetch_player_response(
    video_id: str,
    client: dict,
    session: requests.Session | None = None,
) -> dict:
    """Fetch the Innertube ``player`` response for one client.

    Args:
        video_id: The YouTube video ID.
        client: A client entry from ``config.INNERTUBE_CLIENTS``.
        session: Optional pooled session.

    Returns:
        Parsed JSON response from the player endpoint.

    Raises:
        ExtractionError: If the response cannot be parsed.
    """
    url = config.INNERTUBE_PLAYER_URL

    payload = {
        "context": {"client": client["context"]},
        "videoId": video_id,
        # Bypass "content check" / "racy" interstitials that otherwise hide
        # caption tracks behind a confirmation gate.
        "contentCheckOk": True,
        "racyCheckOk": True,
    }

    resp = request(
        "POST",
        url,
        session=session,
        headers=client.get("headers"),
        json=payload,
        params={"key": config.INNERTUBE_API_KEY},
    )

    try:
        return resp.json()
    except ValueError as e:
        raise ExtractionError(f"Failed to parse player response: {e}")


def _playability_ok(player_response: dict) -> tuple[bool, str]:
    """Return (ok, reason). Some clients gate videos behind login/age checks."""
    status = player_response.get("playabilityStatus", {})
    state = status.get("status", "OK")
    if state in ("OK", "LIVE_STREAM_OFFLINE"):
        return True, state
    reason = status.get("reason") or status.get("errorScreen", {}).get(
        "playerErrorMessageRenderer", {}
    ).get("reason", {}).get("simpleText", state)
    return False, reason


def fetch_caption_data(
    track: CaptionTrack,
    session: requests.Session | None = None,
) -> list[TimedText]:
    """Fetch and parse caption data from a caption track URL.

    Requests json3; some signed URLs ignore the hint and return ``format=3``
    XML instead, so parsing auto-detects the format. An empty body (a common
    symptom of timedtext IP throttling) yields an empty list so callers can
    fall through to another client.
    """
    base_url = track.base_url
    if "fmt=json3" not in base_url:
        base_url = base_url + ("&" if "?" in base_url else "?") + "fmt=json3"

    resp = request("GET", base_url, session=session)
    if not resp.content or not resp.content.strip():
        return []
    return parse_timedtext(resp.content)


def _select_track(tracks: list[CaptionTrack], language: str) -> CaptionTrack:
    """Pick the best caption track for the requested language.

    Priority: exact language match > language prefix match (manual preferred
    over auto-generated) > first available.
    """
    # Exact match.
    for track in tracks:
        if track.language_code == language:
            return track

    # Prefix match, preferring manual captions over ASR.
    prefix = language.split("-")[0]
    prefix_matches = [t for t in tracks if t.language_code.split("-")[0] == prefix]
    if prefix_matches:
        prefix_matches.sort(key=lambda t: t.is_generated)  # manual (False) first
        return prefix_matches[0]

    return tracks[0]


def fetch_transcript_innertube(
    video_id: str,
    language: str = "en",
    session: requests.Session | None = None,
) -> TranscriptData:
    """Fetch a transcript from YouTube captions, trying multiple clients.

    Iterates over ``config.INNERTUBE_CLIENTS`` until one returns caption
    tracks, which sidesteps client-specific blocks (age/region/anti-bot).

    Args:
        video_id: The YouTube video ID or URL.
        language: Preferred language code (e.g. 'en', 'es', 'ja').
        session: Optional pooled session.

    Returns:
        TranscriptData with segments and metadata.

    Raises:
        VideoUnavailable: If the video doesn't exist / is private everywhere.
        NoTranscriptFound: If no caption track is available on any client.
    """
    actual_video_id = extract_video_id(video_id)
    if actual_video_id is None:
        raise ExtractionError(f"Invalid video ID or URL: {video_id}")
    video_id = actual_video_id

    title = "Unknown"
    last_reason = "no caption tracks"
    saw_unplayable = False

    for client in INNERTUBE_CLIENTS:
        try:
            player_response = fetch_player_response(video_id, client, session)
        except Exception:
            # Network error / HTTP 4xx (e.g. FAILED_PRECONDITION) for this
            # client — try the next one rather than giving up.
            continue

        ok, reason = _playability_ok(player_response)
        if not ok:
            saw_unplayable = True
            last_reason = reason
            continue

        details = player_response.get("videoDetails", {})
        if details:
            title = details.get("title", title)

        caption_tracks = parse_player_response(player_response)
        if not caption_tracks:
            last_reason = "no caption tracks for this client"
            continue

        track = _select_track(caption_tracks, language)
        try:
            segments = fetch_caption_data(track, session)
        except Exception as e:
            last_reason = f"caption fetch failed: {e}"
            continue
        if not segments:
            last_reason = "empty caption track (throttled?)"
            continue

        return TranscriptData(
            video_id=video_id,
            title=title,
            language=track.language,
            language_code=track.language_code,
            segments=segments,
            source="innertube",
            is_generated=track.is_generated,
        )

    # Last resort: scrape the watch page, whose embedded player response carries
    # consent context that sometimes succeeds where the bare API is throttled.
    watch_page_result = _fetch_via_watch_page(video_id, language, session)
    if watch_page_result is not None:
        return watch_page_result

    if saw_unplayable and title == "Unknown":
        raise VideoUnavailable(f"Video {video_id} is unavailable: {last_reason}")
    raise NoTranscriptFound(f"No captions found for {video_id} ({last_reason})")


def _fetch_via_watch_page(
    video_id: str,
    language: str,
    session: requests.Session | None = None,
) -> TranscriptData | None:
    """Fallback: extract caption tracks from the watch page HTML.

    Returns a TranscriptData on success, or None if the page yields no usable
    captions (so the caller can raise an aggregated error).
    """
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        # Consent cookie avoids the EU consent interstitial that hides the
        # player response.
        "Cookie": "CONSENT=YES+cb",
    }
    try:
        resp = request(
            "GET",
            "https://www.youtube.com/watch",
            session=session,
            headers=headers,
            params={"v": video_id, "hl": "en"},
            allow_status=(404,),
        )
    except Exception:
        return None
    if resp.status_code == 404:
        return None

    html_text = resp.text
    match = re.search(r'"captionTracks":(\[.*?\])', html_text)
    if not match:
        return None

    try:
        raw_tracks = json.loads(match.group(1))
    except ValueError:
        return None

    tracks = [
        CaptionTrack(
            language=t.get("name", {}).get("simpleText", t.get("languageCode", "")),
            language_code=t.get("languageCode", "en"),
            base_url=t.get("baseUrl", ""),
            is_generated=t.get("kind", "") == "asr",
        )
        for t in raw_tracks
        if t.get("baseUrl")
    ]
    if not tracks:
        return None

    title_match = re.search(r'"title":"([^"]+)"', html_text)
    title = title_match.group(1) if title_match else "Unknown"

    track = _select_track(tracks, language)
    try:
        segments = fetch_caption_data(track, session)
    except Exception:
        return None
    if not segments:
        return None

    return TranscriptData(
        video_id=video_id,
        title=title,
        language=track.language,
        language_code=track.language_code,
        segments=segments,
        source="innertube",
        is_generated=track.is_generated,
    )
