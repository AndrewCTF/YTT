"""Tests for the retry/backoff HTTP layer."""

import pytest

import ytt.http as http
from ytt.exceptions import RateLimitError


class _Resp:
    def __init__(self, status_code, headers=None, body=b"ok"):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        return self._responses.pop(0)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda *_: None)


def test_retries_then_succeeds():
    session = _FakeSession([_Resp(429, {"Retry-After": "0"}), _Resp(200)])
    resp = http.request("GET", "http://x", session=session, max_retries=3)
    assert resp.status_code == 200
    assert session.calls == 2


def test_exhausts_retries_raises_ratelimit():
    session = _FakeSession([_Resp(429), _Resp(429), _Resp(429)])
    with pytest.raises(RateLimitError):
        http.request("GET", "http://x", session=session, max_retries=2)
    assert session.calls == 3


def test_allow_status_returns_without_retry():
    session = _FakeSession([_Resp(404)])
    resp = http.request("GET", "http://x", session=session, allow_status=(404,))
    assert resp.status_code == 404
    assert session.calls == 1


def test_5xx_retried():
    session = _FakeSession([_Resp(503), _Resp(200)])
    resp = http.request("GET", "http://x", session=session, max_retries=2)
    assert resp.status_code == 200
    assert session.calls == 2


def test_sleep_for_honors_retry_after():
    assert http._sleep_for(0, retry_after=5.0) == 5.0
    # Backoff stays within configured ceiling.
    assert http._sleep_for(10, retry_after=None) <= http.config.BACKOFF_MAX
