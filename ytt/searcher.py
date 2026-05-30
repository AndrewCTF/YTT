"""YouTube search using Innertube Search API."""

from dataclasses import dataclass
from typing import Optional

import requests

from .config import INNERTUBE_CLIENTS, config
from .exceptions import SearchError
from .http import request

# Use the WEB client context for search (its renderers parse most cleanly).
_WEB_CLIENT = next(c for c in INNERTUBE_CLIENTS if c["name"] == "WEB")


@dataclass
class VideoSearchResult:
    """A single video search result."""

    video_id: str
    title: str
    channel_name: str
    duration: str
    view_count: str


def build_search_payload(query: str) -> dict:
    """Build the Innertube search request payload."""
    return {
        "context": {"client": _WEB_CLIENT["context"]},
        "query": query,
        "params": "EgIQAQ==",  # Videos-only filter
    }


def parse_search_response(response_data: dict) -> list[VideoSearchResult]:
    """Parse Innertube search response into VideoSearchResult objects.

    The response structure is:
    {
        "contents": {
            "sectionListRenderer": {
                "contents": [{
                    "itemSectionRenderer": {
                        "contents": [
                            { compactVideoRenderer: { videoId, title, ... } },
                            ...
                        ]
                    }
                }]
            }
        }
    }
    """
    results = []

    try:
        contents = response_data.get("contents", {})
        section_list = contents.get("sectionListRenderer", {})
        section_items = section_list.get("contents", [])
    except (KeyError, TypeError):
        raise SearchError("Unexpected search response structure")

    for section in section_items:
        item_section = section.get("itemSectionRenderer", {})
        items = item_section.get("contents", [])

        for item in items:
            # Try videoRenderer first (desktop), then compactVideoRenderer (mobile)
            video_renderer = item.get("videoRenderer") or item.get("compactVideoRenderer")
            if not video_renderer:
                continue

            video_id = video_renderer.get("videoId", "")
            if not video_id:
                continue

            # Extract title (runs is a list of text runs)
            title_runs = video_renderer.get("title", {}).get("runs", [])
            title = "".join(run.get("text", "") for run in title_runs) if title_runs else "Unknown"

            # Extract channel name (shortBylineText for compact, ownerText for videoRenderer)
            channel_runs = video_renderer.get("shortBylineText", {}).get(
                "runs", []
            ) or video_renderer.get("ownerText", {}).get("runs", [])
            channel_name = (
                "".join(run.get("text", "") for run in channel_runs) if channel_runs else "Unknown"
            )

            # Extract duration
            duration = video_renderer.get("lengthText", {})
            if duration:
                duration = duration.get("simpleText", "N/A")
            else:
                duration = "N/A"

            # Extract view count (shortViewCountText for compactVideoRenderer)
            view_count = (
                video_renderer.get("shortViewCountText", {}).get("simpleText")
                or video_renderer.get("viewCountText", {}).get("simpleText")
                or "N/A"
            )

            results.append(
                VideoSearchResult(
                    video_id=video_id,
                    title=title,
                    channel_name=channel_name,
                    duration=duration,
                    view_count=view_count,
                )
            )

    return results


def search_videos_innertube(
    query: str,
    max_results: int = 5,
    session: Optional[requests.Session] = None,
) -> list[VideoSearchResult]:
    """Search YouTube using Innertube API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        session: Optional requests session for connection pooling.

    Returns:
        List of VideoSearchResult objects.

    Raises:
        SearchError: If search fails.
        RateLimitError: If rate limited by YouTube.
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty")

    response = request(
        "POST",
        config.INNERTUBE_SEARCH_URL,
        session=session,
        headers=_WEB_CLIENT.get("headers"),
        json=build_search_payload(query),
        params={"key": config.INNERTUBE_API_KEY},
    )

    try:
        data = response.json()
    except ValueError as e:
        raise SearchError(f"Failed to parse search response: {e}")

    results = parse_search_response(data)
    return results[:max_results]
