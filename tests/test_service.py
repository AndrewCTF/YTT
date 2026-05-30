"""Tests for the service orchestrator (captions-first + Whisper fallback)."""

import asyncio
from datetime import datetime, timedelta

import ytt.service as service
from ytt.cache import CachedTranscript
from ytt.exceptions import NoTranscriptFound
from ytt.fetcher import TranscriptData
from ytt.parser import TimedText
from ytt.whisper_runner import WhisperResult, WhisperSegment


def _transcript(segments_text):
    return TranscriptData(
        video_id="vid",
        title="T",
        language="English",
        language_code="en",
        segments=[
            TimedText(start_ms=i * 1000, duration_ms=1000, text=t)
            for i, t in enumerate(segments_text)
        ],
        source="innertube",
        is_generated=True,
    )


def test_captions_first_and_clean(monkeypatch):
    def fake_fetch(video_id, language, session):
        return _transcript(["hello world this", "world this is a test"])

    monkeypatch.setattr(service, "fetch_transcript_innertube", fake_fetch)

    result = asyncio.run(
        service.get_transcript("abcdefghijk", output_format="clean", use_cache=False)
    )
    assert result.source == "innertube"
    assert result.content == "hello world this is a test"


def test_whisper_fallback_when_no_captions(monkeypatch):
    def fake_fetch(video_id, language, session):
        raise NoTranscriptFound("none")

    def fake_whisper(video_id):
        return WhisperResult(
            video_id=video_id,
            language="en",
            segments=[WhisperSegment(start=0.0, end=1.0, text="spoken words")],
            text="spoken words",
        )

    monkeypatch.setattr(service, "fetch_transcript_innertube", fake_fetch)
    monkeypatch.setattr(service, "fetch_transcript_whisper", fake_whisper)

    result = asyncio.run(
        service.get_transcript("abcdefghijk", output_format="clean", use_cache=False)
    )
    assert result.source == "whisper"
    assert "spoken words" in result.content


def test_no_fallback_reraises(monkeypatch):
    def fake_fetch(video_id, language, session):
        raise NoTranscriptFound("none")

    monkeypatch.setattr(service, "fetch_transcript_innertube", fake_fetch)

    try:
        asyncio.run(
            service.get_transcript("abcdefghijk", use_cache=False, use_whisper_fallback=False)
        )
    except NoTranscriptFound:
        pass
    else:
        raise AssertionError("expected NoTranscriptFound")


def test_invalid_id_raises():
    try:
        asyncio.run(service.get_transcript("definitely not valid", use_cache=False))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_cache_dict_roundtrip():
    original = _transcript(["alpha", "beta"])
    as_dict = service._transcript_to_cache_dict(original)

    cached = CachedTranscript(
        video_id="vid",
        language="en",
        source="innertube",
        raw_data=as_dict,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(days=1),
    )
    restored = service._cached_to_transcript_data(cached)
    assert [s.text for s in restored.segments] == ["alpha", "beta"]
    assert restored.is_generated is True
