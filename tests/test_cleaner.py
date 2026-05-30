"""Tests for the LLM-oriented transcript cleaner."""

from dataclasses import dataclass

from ytt.cleaner import (
    clean_segments,
    estimate_tokens,
    merge_overlapping,
    to_paragraphs,
)


@dataclass
class _Seg:
    text: str


def test_merge_overlapping_rolling_captions():
    # Classic auto-caption rolling overlap.
    texts = ["hello world this", "world this is a", "this is a test"]
    assert merge_overlapping(texts) == "hello world this is a test"


def test_merge_overlapping_fully_contained():
    texts = ["the quick brown fox", "quick brown fox"]
    assert merge_overlapping(texts) == "the quick brown fox"


def test_merge_overlapping_no_overlap():
    assert merge_overlapping(["alpha beta", "gamma delta"]) == "alpha beta gamma delta"


def test_normalize_decodes_entities_and_strips_tags():
    # &amp;#39; -> ' ; <c> markup removed.
    out = merge_overlapping(["it&#39;s <c>great</c>"])
    assert out == "it's great"


def test_merge_case_insensitive_overlap():
    assert merge_overlapping(["Hello World", "world again"]) == "Hello World again"


def test_clean_segments_dedups():
    segs = [_Seg("hello world this"), _Seg("world this is a"), _Seg("this is a test")]
    cleaned = clean_segments(segs, paragraphs=False)
    assert cleaned == "hello world this is a test"
    # No word should appear duplicated back-to-back.
    words = cleaned.split()
    assert "world world" not in cleaned
    assert words.count("test") == 1


def test_to_paragraphs_chunks_unpunctuated():
    text = " ".join(["word"] * 200)
    out = to_paragraphs(text, words_per_paragraph=80)
    assert "\n\n" in out
    # Reflow must not lose or add words.
    assert len(out.replace("\n\n", " ").split()) == 200


def test_to_paragraphs_sentence_aware():
    text = "First sentence. Second sentence. Third one here."
    out = to_paragraphs(text, words_per_paragraph=3)
    assert "First sentence." in out


def test_estimate_tokens():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1
