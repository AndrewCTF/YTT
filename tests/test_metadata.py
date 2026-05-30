"""Tests for metadata, chapters, translation languages, and translated captions."""

import ytt.fetcher as fetcher
from ytt.parser import (
    CaptionTrack,
    parse_chapters_from_description,
    parse_translation_languages,
    parse_video_metadata,
)


def test_parse_translation_languages():
    resp = {
        "captions": {
            "playerCaptionsTracklistRenderer": {
                "translationLanguages": [
                    {"languageCode": "es", "languageName": {"simpleText": "Spanish"}},
                    {"languageCode": "fr", "languageName": {"runs": [{"text": "French"}]}},
                    {"languageName": {"simpleText": "No code, skipped"}},
                ]
            }
        }
    }
    langs = parse_translation_languages(resp)
    assert {"code": "es", "name": "Spanish"} in langs
    assert {"code": "fr", "name": "French"} in langs
    assert len(langs) == 2


def test_parse_chapters_valid():
    desc = (
        "Intro blah blah\n"
        "0:00 Introduction\n"
        "1:30 Getting started\n"
        "12:05 Deep dive\n"
        "1:02:10 Wrap up\n"
        "Follow me on socials"
    )
    chapters = parse_chapters_from_description(desc)
    assert len(chapters) == 4
    assert chapters[0].start_seconds == 0
    assert chapters[0].title == "Introduction"
    assert chapters[2].start_seconds == 12 * 60 + 5
    assert chapters[3].start_seconds == 3600 + 2 * 60 + 10


def test_parse_chapters_requires_zero_start():
    desc = "1:00 First\n2:00 Second\n3:00 Third"
    assert parse_chapters_from_description(desc) == []


def test_parse_chapters_too_few():
    desc = "0:00 Intro\n1:00 Only two"
    assert parse_chapters_from_description(desc) == []


def test_parse_chapters_empty():
    assert parse_chapters_from_description("") == []


def test_parse_video_metadata():
    resp = {
        "videoDetails": {
            "videoId": "abc12345678",
            "title": "Async Python Deep Dive",
            "author": "Code Channel",
            "channelId": "UC123",
            "lengthSeconds": "754",
            "viewCount": "12345",
            "keywords": ["python", "async"],
            "shortDescription": "0:00 Intro\n1:00 Topic A\n2:00 Topic B",
            "isLiveContent": False,
            "thumbnail": {"thumbnails": [{"url": "small.jpg"}, {"url": "large.jpg"}]},
        },
        "microformat": {
            "playerMicroformatRenderer": {
                "publishDate": "2024-01-15",
                "uploadDate": "2024-01-14",
                "category": "Education",
            }
        },
    }
    meta = parse_video_metadata(resp)
    assert meta.video_id == "abc12345678"
    assert meta.title == "Async Python Deep Dive"
    assert meta.author == "Code Channel"
    assert meta.length_seconds == 754
    assert meta.view_count == 12345
    assert meta.keywords == ["python", "async"]
    assert meta.publish_date == "2024-01-15"
    assert meta.category == "Education"
    assert meta.thumbnail == "large.jpg"
    assert len(meta.chapters) == 3  # parsed from description


def test_fetch_caption_data_translate_adds_tlang(monkeypatch):
    captured = {}

    class FakeResp:
        content = b'{"events":[{"tStartMs":0,"dDurationMs":1000,"segs":[{"utf8":"hola"}]}]}'

    def fake_request(method, url, **kw):
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr(fetcher, "request", fake_request)
    track = CaptionTrack(
        language="English",
        language_code="en",
        base_url="http://x/api/timedtext?v=1",
        is_generated=True,
    )
    segs = fetcher.fetch_caption_data(track, translate_to="es")
    assert "tlang=es" in captured["url"]
    # Translated requests must NOT force fmt=json3 (it 429s on signed URLs).
    assert "fmt=json3" not in captured["url"]
    assert segs[0].text == "hola"
