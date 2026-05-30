"""Parse caption tracks and timed text from YouTube Innertube responses."""

import html
import json
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET


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

    # Navigate to caption tracks in the response. The canonical key is
    # ``playerCaptionsTracklistRenderer``; older/alternate clients sometimes use
    # ``playerCaptionsRenderer``, so accept either.
    captions = response_data.get("captions", {})
    renderer = (
        captions.get("playerCaptionsTracklistRenderer")
        or captions.get("playerCaptionsRenderer")
        or {}
    )
    caption_tracks = renderer.get("captionTracks", [])

    for track_data in caption_tracks:
        base_url = track_data.get("baseUrl", "")
        if not base_url:
            continue

        # Extract language info. Some clients (e.g. ANDROID_VR) omit the display
        # name, so fall back to the language code.
        language_code = track_data.get("languageCode", "en")
        language = track_data.get("languageName", {}).get("simpleText") or language_code
        is_generated = track_data.get("kind", "") == "asr"

        tracks.append(
            CaptionTrack(
                language=language,
                language_code=language_code,
                base_url=base_url,
                is_generated=is_generated,
            )
        )

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
        cleaned = caption_data.replace(",}", "}").replace(",]", "]")
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
            segments.append(
                TimedText(
                    start_ms=t_start_ms,
                    duration_ms=d_duration_ms,
                    text=text,
                )
            )

    return segments


def parse_timedtext_xml(caption_data: bytes | str) -> list[TimedText]:
    """Parse YouTube timedtext ``format=3`` XML captions.

    Modern caption baseUrls (e.g. from the ANDROID_VR client) return XML like::

        <timedtext format="3"><body>
          <p t="1360" d="1680">[Music]</p>
          <p t="100" d="200"><s>Hello</s><s> world</s></p>
        </body></timedtext>

    where ``t``/``d`` are start/duration in ms and text may be split across
    ``<s>`` word segments. Entities and embedded newlines are normalised.
    """
    if isinstance(caption_data, bytes):
        caption_data = caption_data.decode("utf-8", "replace")

    # Defensive: YouTube timedtext never declares a DTD or custom entities.
    # Reject any that do, to avoid XXE / billion-laughs against stdlib XML.
    if "<!DOCTYPE" in caption_data or "<!ENTITY" in caption_data:
        raise ValueError("Refusing to parse caption XML containing a DTD/entity declaration")

    segments: list[TimedText] = []
    root = ET.fromstring(caption_data)
    for p in root.iter("p"):
        start_ms = int(p.get("t", 0) or 0)
        duration_ms = int(p.get("d", 0) or 0)

        parts: list[str] = []
        if p.text:
            parts.append(p.text)
        for s in p:  # <s> word-level segments
            if s.text:
                parts.append(s.text)
            if s.tail:
                parts.append(s.tail)

        text = html.unescape("".join(parts)).replace("\n", " ").strip()
        if text:
            segments.append(TimedText(start_ms=start_ms, duration_ms=duration_ms, text=text))

    return segments


def parse_timedtext(caption_data: bytes | str) -> list[TimedText]:
    """Parse caption data in either json3 or XML timedtext format.

    YouTube serves captions as json3 (``{...}``) or ``format=3`` XML
    (``<timedtext>``) depending on the client and track. This dispatches on the
    first non-whitespace byte so callers don't need to care which they got.
    """
    if isinstance(caption_data, bytes):
        text = caption_data.decode("utf-8", "replace")
    else:
        text = caption_data

    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "{":
        return parse_json3_caption_data(text)
    return parse_timedtext_xml(text)


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
    if len(url) == 11 and re.match(r"^[a-zA-Z0-9_-]+$", url):
        return url

    # Patterns for different URL formats
    patterns = [
        r"(?:v=|/v/|/embed/)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None
