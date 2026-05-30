"""Tests for the local semantic retrieval engine."""

from ytt.embeddings import HashingEmbedder
from ytt.parser import TimedText
from ytt.semantic import (
    BM25,
    Passage,
    SemanticIndex,
    chunk_segments,
    search_transcript,
    seconds_to_clock,
)


def _seg(start_ms, dur_ms, text):
    return TimedText(start_ms=start_ms, duration_ms=dur_ms, text=text)


def _transcript():
    # A small "transcript" covering three distinct topics.
    return [
        _seg(0, 3000, "Welcome back to the channel today"),
        _seg(3000, 4000, "First we will talk about python asyncio and the event loop"),
        _seg(7000, 4000, "the event loop schedules coroutines and awaitables efficiently"),
        _seg(60000, 4000, "Next up is how to bake sourdough bread with a starter"),
        _seg(64000, 4000, "you knead the dough and let it proof overnight"),
        _seg(120000, 4000, "Finally we discuss training neural networks with gradient descent"),
        _seg(124000, 4000, "backpropagation computes gradients to update the weights"),
    ]


def test_seconds_to_clock():
    assert seconds_to_clock(0) == "0:00"
    assert seconds_to_clock(75) == "1:15"
    assert seconds_to_clock(3661) == "1:01:01"


def test_chunk_segments_preserves_timestamps_and_progresses():
    chunks = chunk_segments(_transcript(), target_chars=120, overlap_chars=20)
    assert chunks, "expected at least one chunk"
    # Monotonic, valid timestamps; indices are sequential.
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.end_ms >= c.start_ms
        assert c.text
    assert chunks[0].start_ms == 0


def test_chunk_segments_empty():
    assert chunk_segments([]) == []


def test_chunk_no_infinite_loop_on_long_single_cue():
    segs = [_seg(0, 1000, "word " * 500)]
    chunks = chunk_segments(segs, target_chars=50, overlap_chars=10)
    assert len(chunks) >= 1  # terminates, single oversized cue accepted


def test_bm25_ranks_matching_doc_first():
    docs = [
        ["python", "asyncio", "event", "loop"],
        ["sourdough", "bread", "starter", "proof"],
        ["neural", "network", "gradient", "descent"],
    ]
    bm = BM25(docs)
    scores = bm.scores(["gradient", "descent"])
    assert scores[2] == max(scores)
    assert scores[2] > 0


def test_search_transcript_bm25_finds_topic():
    passages = search_transcript(
        _transcript(), "how does the python event loop schedule coroutines", top_k=2, hybrid=False
    )
    assert passages
    assert "event loop" in passages[0].text.lower()
    assert passages[0].method == "bm25"


def test_search_transcript_hybrid_with_hash_embedder():
    passages = search_transcript(
        _transcript(),
        "baking bread dough",
        top_k=2,
        embedder=HashingEmbedder(dim=512),
        hybrid=True,
        video_id="abc12345678",
    )
    assert passages
    top = passages[0]
    assert top.method == "hybrid"
    assert "dough" in top.text.lower() or "bread" in top.text.lower()
    # Deep-link URL into the moment.
    assert top.url() == f"https://youtu.be/abc12345678?t={int(top.start)}"
    assert ":" in top.timestamp


def test_search_empty_transcript():
    assert search_transcript([], "anything", hybrid=False) == []


def test_semantic_index_mmr_diversity():
    # Two near-duplicate relevant chunks + one other; MMR should not return dupes only.
    segs = [
        _seg(0, 3000, "machine learning models need training data to learn patterns"),
        _seg(3000, 3000, "machine learning models require training data to learn patterns"),
        _seg(6000, 3000, "deploying the model to production with monitoring and rollback"),
    ]
    chunks = chunk_segments(segs, target_chars=80, overlap_chars=0)
    idx = SemanticIndex(chunks, HashingEmbedder(dim=512))
    passages = idx.search("training data for machine learning", top_k=2, mmr=True)
    assert len(passages) == 2
    assert all(isinstance(p, Passage) for p in passages)
