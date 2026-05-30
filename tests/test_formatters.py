"""Tests for output formatters."""

import json

from ytt.fetcher import TranscriptData
from ytt.formatters import format_transcript
from ytt.parser import TimedText


def _sample() -> TranscriptData:
    return TranscriptData(
        video_id="vid123",
        title="Sample",
        language="English",
        language_code="en",
        segments=[
            TimedText(start_ms=0, duration_ms=1000, text="hello world this"),
            TimedText(start_ms=1000, duration_ms=1000, text="world this is a test"),
        ],
        source="innertube",
        is_generated=True,
    )


def test_format_clean_dedups():
    out = format_transcript(_sample(), "clean")
    assert out == "hello world this is a test"


def test_format_text_keeps_segments():
    out = format_transcript(_sample(), "text")
    assert out.splitlines() == ["hello world this", "world this is a test"]


def test_format_json_roundtrip():
    out = format_transcript(_sample(), "json")
    data = json.loads(out)
    assert data["video_id"] == "vid123"
    assert len(data["segments"]) == 2


def test_format_srt_has_timestamps():
    out = format_transcript(_sample(), "srt")
    assert "00:00:00,000 --> 00:00:01,000" in out


def test_format_vtt_header():
    out = format_transcript(_sample(), "vtt")
    assert out.startswith("WEBVTT")


def test_format_unknown_raises():
    try:
        format_transcript(_sample(), "bogus")
    except ValueError as e:
        assert "clean" in str(e)
    else:
        raise AssertionError("expected ValueError")
