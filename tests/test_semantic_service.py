"""Tests for the semantic service layer (search-in-video, ask, index, rerank)."""

import asyncio

import ytt.search_service as search_service
import ytt.service as service
from ytt.fetcher import TranscriptData
from ytt.parser import TimedText
from ytt.searcher import VideoSearchResult


def _transcript(video_id, texts, title="T"):
    return TranscriptData(
        video_id=video_id,
        title=title,
        language="English",
        language_code="en",
        segments=[
            TimedText(start_ms=i * 4000, duration_ms=4000, text=t) for i, t in enumerate(texts)
        ],
        source="innertube",
        is_generated=True,
    )


_PY = ["python asyncio drives the event loop", "coroutines await without blocking threads"]
_BREAD = ["sourdough needs a live starter culture", "knead and proof the dough before baking"]


def _patch_transcripts(monkeypatch, mapping):
    async def fake_get(video_id_or_url, language="en", translate=None, use_cache=True, **kw):
        from ytt.parser import extract_video_id

        vid = extract_video_id(video_id_or_url) or video_id_or_url
        return mapping[vid], "innertube"

    monkeypatch.setattr(service, "get_transcript_data", fake_get)
    return fake_get


def test_search_in_video(monkeypatch):
    _patch_transcripts(monkeypatch, {"vid_python1": _transcript("vid_python1", _PY)})
    res = asyncio.run(
        service.search_in_video(
            "vid_python1", "event loop coroutines", top_k=2, embed_provider="hash"
        )
    )
    assert res.video_id == "vid_python1"
    assert res.passages
    assert (
        "event loop" in res.passages[0].text.lower() or "coroutine" in res.passages[0].text.lower()
    )
    # deep-link timestamp present
    assert res.passages[0].url().startswith("https://youtu.be/vid_python1?t=")


def test_ask_video_with_llm(monkeypatch):
    _patch_transcripts(monkeypatch, {"vid_python1": _transcript("vid_python1", _PY)})
    captured = {}

    def fake_generate(prompt, model=None, provider=None):
        captured["prompt"] = prompt
        return "Coroutines await without blocking [0:04]."

    monkeypatch.setattr(service, "_llm_generate", fake_generate)
    res = asyncio.run(
        service.ask_video("vid_python1", "how do coroutines work", embed_provider="hash")
    )
    assert res.llm_used is True
    assert res.answer and "[0:04]" in res.answer
    assert res.passages
    # The prompt is grounded: it contains timestamped excerpts.
    assert "[0:00]" in captured["prompt"] or "[0:04]" in captured["prompt"]


def test_ask_video_no_llm_returns_passages(monkeypatch):
    _patch_transcripts(monkeypatch, {"vid_python1": _transcript("vid_python1", _PY)})

    def boom(prompt, model=None, provider=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(service, "_llm_generate", boom)
    res = asyncio.run(service.ask_video("vid_python1", "coroutines", embed_provider="hash"))
    assert res.llm_used is False
    assert res.answer is None
    assert res.passages  # still get exa-style cited passages
    assert "unavailable" in res.note.lower()


def test_ask_video_answer_disabled(monkeypatch):
    _patch_transcripts(monkeypatch, {"vid_python1": _transcript("vid_python1", _PY)})
    res = asyncio.run(
        service.ask_video("vid_python1", "coroutines", answer=False, embed_provider="hash")
    )
    assert res.answer is None
    assert res.llm_used is False
    assert res.passages


def test_index_and_find(monkeypatch, tmp_path):
    db = str(tmp_path / "corpus.db")
    _patch_transcripts(
        monkeypatch,
        {
            "vid_python1": _transcript("vid_python1", _PY, title="Python"),
            "vid_bread12": _transcript("vid_bread12", _BREAD, title="Bread"),
        },
    )
    summary = asyncio.run(
        service.index_videos(["vid_python1", "vid_bread12"], embed_provider="hash", db_path=db)
    )
    assert len(summary["indexed"]) == 2
    assert not summary["failed"]

    hits = asyncio.run(
        service.find_in_corpus(
            "proof the sourdough dough", top_k=2, embed_provider="hash", db_path=db
        )
    )
    assert hits
    assert hits[0].video_id == "vid_bread12"

    stats = asyncio.run(service.corpus_stats(db_path=db))
    assert stats["videos"] == 2


def test_search_ranked(monkeypatch):
    # Two candidates; the bread video should rank first for a bread query.
    results = [
        VideoSearchResult("vid_python1", "Python", "Chan", "10:00", "1K"),
        VideoSearchResult("vid_bread12", "Bread", "Chan", "10:00", "1K"),
    ]

    async def fake_search(query, max_results=5, use_cache=True):
        return results

    async def fake_get(video_id, language="en", translate=None, use_cache=True, **kw):
        m = {
            "vid_python1": _transcript("vid_python1", _PY),
            "vid_bread12": _transcript("vid_bread12", _BREAD),
        }
        return m[video_id], "innertube"

    monkeypatch.setattr(search_service, "search", fake_search)
    monkeypatch.setattr(search_service, "get_transcript_data", fake_get)

    ranked = asyncio.run(
        search_service.search_ranked(
            "sourdough bread starter", max_results=2, embed_provider="hash"
        )
    )
    assert ranked[0][0].video_id == "vid_bread12"
    assert ranked[0][1] >= ranked[1][1]
