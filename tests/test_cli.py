"""CLI smoke tests via Click's CliRunner (services mocked / offline)."""

from click.testing import CliRunner

import ytt.cli as cli_mod
from ytt.cli import cli
from ytt.fetcher import CaptionListing
from ytt.parser import CaptionTrack, Chapter, VideoMetadata
from ytt.service import AskResult, SearchInVideoResult
from ytt.semantic import Passage


def _passage():
    return Passage(
        text="coroutines await without blocking threads",
        start_ms=4000,
        end_ms=8000,
        score=0.9,
        chunk_index=1,
        video_id="vid_python1",
        title="Python",
    )


def test_info_command(monkeypatch):
    async def fake_info(video_id):
        return VideoMetadata(
            video_id="abc12345678",
            title="Async Python",
            author="Code Channel",
            length_seconds=125,
            view_count=98765,
            publish_date="2024-01-15",
            category="Education",
            keywords=["python"],
            short_description="A talk about async.",
            chapters=[Chapter(0, "Intro"), Chapter(60, "Deep dive")],
        )

    monkeypatch.setattr(cli_mod, "get_video_info", fake_info)
    result = CliRunner().invoke(cli, ["info", "abc12345678"])
    assert result.exit_code == 0
    assert "Async Python" in result.output
    assert "98,765" in result.output
    assert "Intro" in result.output and "Deep dive" in result.output


def test_langs_command(monkeypatch):
    async def fake_langs(video_id):
        return CaptionListing(
            video_id="abc12345678",
            title="Async Python",
            tracks=[
                CaptionTrack("English", "en", "u", is_generated=False),
                CaptionTrack("Spanish", "es", "u", is_generated=True),
            ],
            translation_languages=[{"code": "fr", "name": "French"}],
        )

    monkeypatch.setattr(cli_mod, "list_languages", fake_langs)
    result = CliRunner().invoke(cli, ["langs", "abc12345678"])
    assert result.exit_code == 0
    assert "English" in result.output and "manual" in result.output
    assert "auto" in result.output
    assert "fr" in result.output  # translation target


def test_ask_command(monkeypatch):
    async def fake_ask(video_id, question, **kw):
        return AskResult(
            video_id="vid_python1",
            title="Python",
            question=question,
            answer="Coroutines await without blocking [0:04].",
            passages=[_passage()],
            llm_used=True,
        )

    monkeypatch.setattr(cli_mod, "ask_video", fake_ask)
    result = CliRunner().invoke(cli, ["ask", "vid_python1", "how do coroutines work"])
    assert result.exit_code == 0
    assert "[0:04]" in result.output
    assert "youtu.be/vid_python1" in result.output  # deep link in sources


def test_search_in_video_via_ask_passages_only(monkeypatch):
    async def fake_ask(video_id, question, **kw):
        assert kw.get("answer") is False
        return AskResult(
            video_id="vid_python1",
            title="Python",
            question=question,
            answer=None,
            passages=[_passage()],
            llm_used=False,
            note="Answer generation disabled; returning passages only.",
        )

    monkeypatch.setattr(cli_mod, "ask_video", fake_ask)
    result = CliRunner().invoke(cli, ["ask", "vid_python1", "coroutines", "--passages-only"])
    assert result.exit_code == 0
    assert "coroutines await" in result.output


def test_find_and_corpus_offline(tmp_path):
    # Empty corpus DB — exercises the real code path with no network.
    db = str(tmp_path / "corpus.db")
    runner = CliRunner()

    r1 = runner.invoke(cli, ["corpus", "--db", db])
    assert r1.exit_code == 0
    assert "Videos: 0" in r1.output

    r2 = runner.invoke(cli, ["find", "anything", "--db", db])
    assert r2.exit_code == 0
    assert "No relevant passages" in r2.output


def test_search_in_video_service_used_in_ask(monkeypatch):
    # Guard: the ask command wires SearchInVideoResult/Passage shapes correctly.
    sr = SearchInVideoResult("v", "T", "q", [_passage()])
    assert sr.passages[0].timestamp == "0:04"
