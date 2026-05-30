"""Tests for the cross-video corpus index."""

from ytt.embeddings import HashingEmbedder
from ytt.index import CorpusIndex, _pack, _unpack
from ytt.parser import TimedText


def _seg(start_ms, text):
    return TimedText(start_ms=start_ms, duration_ms=3000, text=text)


def _py_segs():
    return [
        _seg(0, "python asyncio event loop schedules coroutines"),
        _seg(3000, "await yields control back to the event loop"),
    ]


def _bread_segs():
    return [
        _seg(0, "sourdough bread needs a healthy starter culture"),
        _seg(3000, "knead the dough then proof it overnight before baking"),
    ]


def test_pack_unpack_roundtrip():
    v = [0.1, -0.5, 0.25, 1.0]
    out = _unpack(_pack(v))
    assert len(out) == 4
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, v))


def test_add_and_list(tmp_path):
    db = str(tmp_path / "idx.db")
    with CorpusIndex(db) as idx:
        n = idx.add_video("vid_python1", "Python Async", _py_segs(), embedder=HashingEmbedder())
        assert n >= 1
        assert idx.has_video("vid_python1")
        vids = idx.list_videos()
        assert len(vids) == 1
        assert vids[0].title == "Python Async"
        assert vids[0].embedder == "hash"


def test_add_replaces_existing(tmp_path):
    db = str(tmp_path / "idx.db")
    with CorpusIndex(db) as idx:
        idx.add_video("v1", "First", _py_segs(), embedder=HashingEmbedder())
        idx.add_video("v1", "Second", _bread_segs(), embedder=HashingEmbedder())
        vids = idx.list_videos()
        assert len(vids) == 1
        assert vids[0].title == "Second"


def test_cross_video_search_finds_right_video(tmp_path):
    db = str(tmp_path / "idx.db")
    emb = HashingEmbedder(dim=512)
    with CorpusIndex(db) as idx:
        idx.add_video("vid_python1", "Python Async", _py_segs(), embedder=emb)
        idx.add_video("vid_bread12", "Sourdough", _bread_segs(), embedder=emb)

        hits = idx.search("how do I proof sourdough dough", top_k=2, embedder=emb)
        assert hits
        assert hits[0].video_id == "vid_bread12"
        assert hits[0].method == "hybrid"

        hits2 = idx.search("event loop coroutines", top_k=2, embedder=emb)
        assert hits2[0].video_id == "vid_python1"


def test_search_bm25_only_without_embedder(tmp_path):
    db = str(tmp_path / "idx.db")
    with CorpusIndex(db) as idx:
        idx.add_video("vid_python1", "Python Async", _py_segs())  # no embedder
        hits = idx.search("coroutines event loop", top_k=1)
        assert hits
        assert hits[0].video_id == "vid_python1"
        assert hits[0].method == "bm25"


def test_remove_and_stats(tmp_path):
    db = str(tmp_path / "idx.db")
    with CorpusIndex(db) as idx:
        idx.add_video("v1", "A", _py_segs(), embedder=HashingEmbedder())
        idx.add_video("v2", "B", _bread_segs(), embedder=HashingEmbedder())
        s = idx.stats()
        assert s["videos"] == 2
        assert s["chunks"] >= 2
        assert idx.remove_video("v1") is True
        assert idx.stats()["videos"] == 1
        assert idx.remove_video("nope") is False


def test_search_empty_index(tmp_path):
    db = str(tmp_path / "idx.db")
    with CorpusIndex(db) as idx:
        assert idx.search("anything") == []
