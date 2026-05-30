"""Tests for caption track selection and playability parsing."""

from ytt.fetcher import _playability_ok, _select_track
from ytt.parser import CaptionTrack


def _track(code, generated=False):
    return CaptionTrack(language=code, language_code=code, base_url="u", is_generated=generated)


def test_select_exact_match():
    tracks = [_track("es"), _track("en"), _track("fr")]
    assert _select_track(tracks, "en").language_code == "en"


def test_select_prefix_prefers_manual():
    tracks = [_track("en-US", generated=True), _track("en-GB", generated=False)]
    chosen = _select_track(tracks, "en")
    assert chosen.language_code == "en-GB"
    assert chosen.is_generated is False


def test_select_falls_back_to_first():
    tracks = [_track("de"), _track("ja")]
    assert _select_track(tracks, "en").language_code == "de"


def test_playability_ok():
    ok, reason = _playability_ok({"playabilityStatus": {"status": "OK"}})
    assert ok is True
    assert reason == "OK"


def test_playability_blocked():
    resp = {"playabilityStatus": {"status": "LOGIN_REQUIRED", "reason": "Sign in to confirm"}}
    ok, reason = _playability_ok(resp)
    assert ok is False
    assert reason == "Sign in to confirm"
