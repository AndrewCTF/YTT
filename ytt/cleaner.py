"""Clean transcripts for LLM ingestion without wasting context.

YouTube auto-captions (ASR) "roll": each cue repeats the tail of the previous
cue plus a few new words, so naively joining cues produces 2-3x duplicated
text. This module merges that overlap, decodes HTML entities, strips caption
markup, and re-flows the result into compact paragraphs — typically cutting
token count by half or more versus the raw segment dump.
"""

import html
import re

# Caption markup like <c>, </c>, <00:00:01.234> and stray formatting tags.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Decode entities, strip markup, collapse whitespace for one cue."""
    text = html.unescape(text)
    text = text.replace("\n", " ")
    text = _TAG_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def merge_overlapping(texts: list[str]) -> str:
    """Merge cue texts, removing rolling-caption word overlap.

    For each cue we find the longest suffix of the text so far that equals the
    cue's leading words (case-insensitive) and append only the remainder. This
    collapses ``"a b c" + "b c d" -> "a b c d"`` and drops fully-contained
    repeats entirely.
    """
    words: list[str] = []
    lowered: list[str] = []  # parallel lowercased view for cheap comparison

    for raw in texts:
        seg = _normalize(raw)
        if not seg:
            continue
        seg_words = seg.split()
        seg_lower = [w.lower() for w in seg_words]

        max_k = min(len(words), len(seg_words))
        overlap = 0
        for k in range(max_k, 0, -1):
            if lowered[-k:] == seg_lower[:k]:
                overlap = k
                break

        words.extend(seg_words[overlap:])
        lowered.extend(seg_lower[overlap:])

    return " ".join(words)


def to_paragraphs(text: str, words_per_paragraph: int = 80) -> str:
    """Re-flow a dense string into readable paragraphs.

    Auto-captions usually lack punctuation, so we group a fixed number of words
    per paragraph. This adds only newlines (negligible tokens) while keeping the
    text scannable. If the text already has sentence punctuation we break on
    sentence boundaries instead.
    """
    text = text.strip()
    if not text:
        return ""

    # Prefer sentence-based grouping when punctuation is present.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        paragraphs, current, count = [], [], 0
        for sentence in sentences:
            current.append(sentence)
            count += len(sentence.split())
            if count >= words_per_paragraph:
                paragraphs.append(" ".join(current))
                current, count = [], 0
        if current:
            paragraphs.append(" ".join(current))
        return "\n\n".join(paragraphs)

    # No punctuation: chunk by word count.
    words = text.split()
    paragraphs = [
        " ".join(words[i : i + words_per_paragraph])
        for i in range(0, len(words), words_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


def clean_segments(segments, paragraphs: bool = True) -> str:
    """Produce clean, deduplicated text from a list of caption segments.

    Args:
        segments: Objects exposing a ``.text`` attribute (TimedText / WhisperSegment).
        paragraphs: Re-flow into paragraphs (True) or return one dense line.

    Returns:
        Cleaned transcript text optimised for LLM ingestion.
    """
    merged = merge_overlapping([getattr(seg, "text", "") for seg in segments])
    return to_paragraphs(merged) if paragraphs else merged


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budgeting LLM context."""
    return max(1, len(text) // 4)
