"""Tests for URL/ID extraction and Innertube response parsing."""

import pytest

from ytt.parser import (
    extract_video_id,
    parse_json3_caption_data,
    parse_player_response,
    parse_timedtext,
    parse_timedtext_xml,
)


def test_extract_video_id_plain():
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short_url():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_embed_url():
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    assert extract_video_id("not a video") is None


def test_parse_json3_caption_data():
    raw = {
        "events": [
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]},
            {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "world"}]},
            {"tStartMs": 2000, "dDurationMs": 1000, "segs": [{"utf8": "   "}]},  # blank -> skipped
        ]
    }
    import json

    segments = parse_json3_caption_data(json.dumps(raw))
    assert [s.text for s in segments] == ["hello", "world"]
    assert segments[0].start_ms == 0
    assert segments[1].duration_ms == 1000


def test_parse_player_response_tracklist_renderer():
    response = {
        "captions": {
            "playerCaptionsTracklistRenderer": {
                "captionTracks": [
                    {
                        "baseUrl": "https://example.com/caption",
                        "languageCode": "en",
                        "languageName": {"simpleText": "English"},
                        "kind": "asr",
                    }
                ]
            }
        }
    }
    tracks = parse_player_response(response)
    assert len(tracks) == 1
    assert tracks[0].language_code == "en"
    assert tracks[0].is_generated is True


def test_parse_player_response_legacy_renderer():
    response = {
        "captions": {
            "playerCaptionsRenderer": {
                "captionTracks": [{"baseUrl": "https://example.com/c", "languageCode": "es"}]
            }
        }
    }
    tracks = parse_player_response(response)
    assert tracks[0].language_code == "es"
    assert tracks[0].is_generated is False


def test_parse_player_response_no_captions():
    assert parse_player_response({}) == []


_XML = (
    '<?xml version="1.0" encoding="utf-8" ?><timedtext format="3"><body>'
    '<p t="1360" d="1680">[Music]</p>'
    '<p t="18640" d="3240">We&#39;re no strangers\nto love</p>'
    '<p t="100" d="200"><s>Hello</s><s> world</s></p>'
    "</body></timedtext>"
)


def test_parse_timedtext_xml():
    segs = parse_timedtext_xml(_XML)
    assert [s.text for s in segs] == ["[Music]", "We're no strangers to love", "Hello world"]
    assert segs[0].start_ms == 1360
    assert segs[1].duration_ms == 3240


def test_parse_timedtext_dispatch_xml():
    segs = parse_timedtext(_XML.encode("utf-8"))
    assert segs[0].text == "[Music]"


def test_parse_timedtext_dispatch_json3():
    import json

    raw = json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 1, "segs": [{"utf8": "hi"}]}]})
    assert parse_timedtext(raw)[0].text == "hi"


def test_parse_timedtext_empty():
    assert parse_timedtext("") == []
    assert parse_timedtext(b"   ") == []


def test_parse_timedtext_xml_rejects_dtd():
    malicious = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "boom">]><timedtext></timedtext>'
    with pytest.raises(ValueError):
        parse_timedtext_xml(malicious)
