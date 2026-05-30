"""Parse caption tracks and timed text from YouTube Innertube responses."""

import html
import json
import re
from dataclasses import dataclass, field
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


@dataclass
class Chapter:
    """A video chapter (start time + title), parsed from the description."""

    start_seconds: int
    title: str


@dataclass
class VideoMetadata:
    """Rich video metadata extracted from the Innertube player response.

    Captions-first and audio-free: everything here comes from the same player
    response that carries the caption tracks, so it costs no extra download.
    """

    video_id: str
    title: str = ""
    author: str = ""
    channel_id: str = ""
    length_seconds: int = 0
    view_count: int = 0
    short_description: str = ""
    keywords: list[str] = field(default_factory=list)
    is_live: bool = False
    publish_date: str = ""
    upload_date: str = ""
    category: str = ""
    thumbnail: str = ""
    chapters: list[Chapter] = field(default_factory=list)


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


def parse_translation_languages(response_data: dict) -> list[dict]:
    """Extract the on-demand translation languages offered for this video.

    YouTube can machine-translate any caption track into these languages via the
    ``&tlang=`` query param — the same capability ``yt-dlp`` exposes as
    ``--sub-langs`` translations. Returns ``[{"code", "name"}, ...]``.
    """
    captions = response_data.get("captions", {})
    renderer = (
        captions.get("playerCaptionsTracklistRenderer")
        or captions.get("playerCaptionsRenderer")
        or {}
    )
    langs = []
    for entry in renderer.get("translationLanguages", []):
        code = entry.get("languageCode")
        if not code:
            continue
        name = entry.get("languageName", {})
        label = name.get("simpleText") or "".join(r.get("text", "") for r in name.get("runs", []))
        langs.append({"code": code, "name": label or code})
    return langs


def _runs_text(node: dict) -> str:
    """Join a YouTube ``{simpleText|runs}`` text node into a plain string."""
    if not node:
        return ""
    if "simpleText" in node:
        return node["simpleText"]
    return "".join(r.get("text", "") for r in node.get("runs", []))


def _clock_to_seconds(clock: str) -> int:
    """Convert ``H:MM:SS`` / ``M:SS`` / ``SS`` to integer seconds."""
    parts = [int(p) for p in clock.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


_CHAPTER_RE = re.compile(r"(?m)^\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{2})\)?\s*[-–—)\.]?\s+(.+?)\s*$")


def parse_chapters_from_description(description: str) -> list["Chapter"]:
    """Parse ``mm:ss Title`` chapter markers from a video description.

    YouTube derives chapters from timestamp lines in the description (the first
    must be ``0:00``). This mirrors that heuristic so chapters are available on
    the captions-first path without scraping the watch page. Returns ``[]`` when
    the description has no valid chapter list.
    """
    if not description:
        return []
    found: list[Chapter] = []
    for m in _CHAPTER_RE.finditer(description):
        start = _clock_to_seconds(m.group(1))
        title = m.group(2).strip()
        if title:
            found.append(Chapter(start_seconds=start, title=title))
    # Valid chapter lists start at 0:00 and have at least three markers.
    if len(found) >= 3 and found[0].start_seconds == 0:
        # Keep only the monotonically increasing prefix (defensive against noise).
        chapters = [found[0]]
        for ch in found[1:]:
            if ch.start_seconds > chapters[-1].start_seconds:
                chapters.append(ch)
        return chapters
    return []


def parse_video_metadata(response_data: dict) -> VideoMetadata:
    """Extract rich metadata from an Innertube player response.

    Pulls from ``videoDetails`` (title, author, length, views, description,
    keywords) and ``microformat.playerMicroformatRenderer`` (publish/upload date,
    category) — and derives chapters from the description.
    """
    details = response_data.get("videoDetails", {})
    micro = response_data.get("microformat", {}).get("playerMicroformatRenderer", {})

    def _int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    description = details.get("shortDescription") or _runs_text(micro.get("description", {}))
    thumbs = details.get("thumbnail", {}).get("thumbnails", [])
    thumbnail = thumbs[-1]["url"] if thumbs else ""

    return VideoMetadata(
        video_id=details.get("videoId", ""),
        title=details.get("title", "") or _runs_text(micro.get("title", {})),
        author=details.get("author", "") or micro.get("ownerChannelName", ""),
        channel_id=details.get("channelId", "") or micro.get("externalChannelId", ""),
        length_seconds=_int(details.get("lengthSeconds") or micro.get("lengthSeconds")),
        view_count=_int(details.get("viewCount") or micro.get("viewCount")),
        short_description=description,
        keywords=details.get("keywords", []) or [],
        is_live=bool(details.get("isLiveContent", False)),
        publish_date=micro.get("publishDate", ""),
        upload_date=micro.get("uploadDate", ""),
        category=micro.get("category", ""),
        thumbnail=thumbnail,
        chapters=parse_chapters_from_description(description),
    )


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
