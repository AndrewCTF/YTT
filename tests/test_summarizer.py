"""Tests for the local-LLM summarizer (no real model required)."""

import pytest
import requests

import ytt.summarizer as summarizer
from ytt.exceptions import SummarizerError


def test_strip_think():
    assert summarizer._strip_think("<think>reasoning</think>Answer") == "Answer"
    assert summarizer._strip_think("plain") == "plain"


def test_strip_think_nested():
    assert summarizer._strip_think("<think>a<think>b</think></think>Answer") == "Answer"


def test_chunk_hard_splits_oversized_word():
    big_word = "x" * 50
    chunks = summarizer._chunk("a " + big_word + " b", 20)
    # No chunk exceeds the limit even though one token is larger than it.
    assert all(len(c) <= 20 for c in chunks)
    # All original characters are preserved across chunks.
    assert "".join(chunks).replace(" ", "") == ("a" + big_word + "b")


def test_validate_url_rejects_non_http():
    with pytest.raises(SummarizerError):
        summarizer._validate_url("file:///etc/passwd")
    assert summarizer._validate_url("http://localhost:11434/") == "http://localhost:11434"


def test_chunk_short():
    assert summarizer._chunk("a b c", 100) == ["a b c"]


def test_chunk_splits_on_words():
    text = " ".join(["word"] * 100)
    chunks = summarizer._chunk(text, 40)
    assert len(chunks) > 1
    # No data lost.
    assert " ".join(chunks).split() == text.split()
    assert all(len(c) <= 40 for c in chunks[:-1])


def test_summarize_empty_raises():
    with pytest.raises(SummarizerError):
        summarizer.summarize_text("   ")


def test_summarize_single_chunk(monkeypatch):
    calls = []

    def fake_generate(prompt, model, provider, session):
        calls.append(prompt)
        return "SUMMARY"

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    out = summarizer.summarize_text("hello world", model="m", provider="ollama")
    assert out == "SUMMARY"
    assert len(calls) == 1


def test_summarize_map_reduce(monkeypatch):
    seen = {"n": 0}

    def fake_generate(prompt, model, provider, session):
        seen["n"] += 1
        return f"part{seen['n']}"

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    monkeypatch.setattr(summarizer.config, "SUMMARY_MAX_INPUT_CHARS", 20)
    long_text = " ".join(["word"] * 80)
    out = summarizer.summarize_text(long_text, model="m", provider="ollama")
    # Multiple chunk summaries + a reduce call.
    assert seen["n"] >= 3
    assert isinstance(out, str) and out


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(summarizer, "_chunk", lambda t, n: ["x"])
    with pytest.raises(SummarizerError):
        summarizer.summarize_text("x", provider="bogus")


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def post(self, *a, **k):
        return self._resp


def test_ollama_generate_ok():
    session = _FakeSession(_Resp(200, {"response": "<think>x</think>hi"}))
    assert summarizer._ollama_generate("p", "m", session) == "hi"


def test_ollama_generate_unreachable():
    class Boom:
        def post(self, *a, **k):
            raise requests.ConnectionError("refused")

    with pytest.raises(SummarizerError):
        summarizer._ollama_generate("p", "m", Boom())


def test_ollama_generate_empty_response():
    session = _FakeSession(_Resp(200, {"response": ""}))
    with pytest.raises(SummarizerError):
        summarizer._ollama_generate("p", "m", session)
