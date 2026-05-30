"""Local semantic retrieval over transcripts — the exa-grade search core.

Given a transcript (timed caption segments), this turns a natural-language
query into ranked, **timestamped** passages with deep-link URLs straight to the
moment in the video. It is fully local and works with no extra dependencies:

* cues are merged (rolling-caption dedup) into overlapping, timestamp-preserving
  windows (:func:`chunk_segments`),
* a pure-Python **BM25** lexical ranker scores them,
* if an embedding backend is available (see :mod:`ytt.embeddings`), a dense
  vector ranker runs too and the two are fused with **Reciprocal Rank Fusion**
  (robust, score-scale-free),
* **MMR** re-ranking trades a little relevance for diversity so you don't get
  five near-identical passages.

The result is "search inside the video": ask a question, get the exact spots.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .cleaner import merge_overlapping
from .config import config
from .embeddings import Embedder, cosine, get_embedder, tokenize

# Small, conservative stopword list. Kept short on purpose: aggressive stopword
# removal hurts short transcript queries more than it helps.
_STOPWORDS = frozenset(
    "a an the of to in on at for and or but is are was were be been being it this that "
    "these those with as by from i you he she they we me my your our their what which who "
    "how why when where do does did so if then than too very can will just about into".split()
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def seconds_to_clock(seconds: float) -> str:
    """Format seconds as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


@dataclass
class Chunk:
    """A timestamped, deduplicated window of transcript text."""

    index: int
    text: str
    start_ms: int
    end_ms: int

    @property
    def start(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end(self) -> float:
        return self.end_ms / 1000.0


@dataclass
class Passage:
    """A retrieved chunk plus its relevance score and provenance."""

    text: str
    start_ms: int
    end_ms: int
    score: float
    chunk_index: int
    video_id: str | None = None
    title: str | None = None
    method: str = "hybrid"

    @property
    def start(self) -> float:
        return self.start_ms / 1000.0

    @property
    def timestamp(self) -> str:
        return seconds_to_clock(self.start)

    def url(self, video_id: str | None = None) -> str:
        vid = video_id or self.video_id
        return f"https://youtu.be/{vid}?t={int(self.start)}" if vid else ""


def chunk_segments(
    segments,
    target_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[Chunk]:
    """Merge caption cues into overlapping, timestamp-preserving windows.

    Each window holds up to ``target_chars`` of text; consecutive windows share
    ~``overlap_chars`` of trailing context so a passage split across a boundary
    is still findable. Rolling auto-caption duplication is removed per window via
    :func:`ytt.cleaner.merge_overlapping`, so passage text is clean to display.
    """
    target_chars = target_chars or config.CHUNK_TARGET_CHARS
    overlap_chars = overlap_chars or config.CHUNK_OVERLAP_CHARS

    segs = [
        (int(s.start_ms), int(s.end_ms), s.text.strip())
        for s in segments
        if getattr(s, "text", "").strip()
    ]
    chunks: list[Chunk] = []
    n = len(segs)
    i = 0
    idx = 0
    while i < n:
        cur: list[tuple[int, int, str]] = []
        size = 0
        j = i
        while j < n and (size == 0 or size + len(segs[j][2]) + 1 <= target_chars):
            cur.append(segs[j])
            size += len(segs[j][2]) + 1
            j += 1

        text = merge_overlapping([c[2] for c in cur])
        if text:
            chunks.append(Chunk(index=idx, text=text, start_ms=cur[0][0], end_ms=cur[-1][1]))
            idx += 1

        if j >= n:
            break

        # Step back over trailing cues to create the overlap for the next window.
        back = 0
        bsize = 0
        k = j - 1
        while k > i and bsize < overlap_chars:
            bsize += len(segs[k][2]) + 1
            k -= 1
            back += 1
        i = max(i + 1, j - back)  # always make progress

    return chunks


class BM25:
    """Okapi BM25 (BM25+ idf variant, so idf is always non-negative)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        if not self.N or self.avgdl == 0.0:
            return scores
        q = [t for t in query_tokens if t in self.idf]
        for i in range(self.N):
            dl = self.dl[i]
            tf_i = self.tf[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s = 0.0
            for term in q:
                f = tf_i.get(term, 0)
                if f:
                    s += self.idf[term] * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores


def _rank_order(scores: list[float]) -> list[int]:
    """Indices sorted by descending score (stable on ties)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _rrf_fuse(rankings: list[list[int]], n: int, k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion across several rank orderings → fused score/doc."""
    fused = [0.0] * n
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] += 1.0 / (k + rank + 1)
    return fused


class SemanticIndex:
    """In-memory hybrid index over a single document's chunks."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self._docs_tokens = [_content_tokens(c.text) for c in chunks]
        self.bm25 = BM25(self._docs_tokens)
        self.embedder = embedder
        self._vectors: list[list[float]] | None = None

    def _ensure_vectors(self) -> bool:
        if self.embedder is None or not self.chunks:
            return False
        if self._vectors is None:
            try:
                self._vectors = self.embedder.embed([c.text for c in self.chunks])
            except Exception:
                # Embedding backend unreachable (e.g. Ollama down) — degrade to
                # pure lexical BM25 rather than failing the whole query.
                self._vectors = []
                self.embedder = None
                return False
        return bool(self._vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = True,
        mmr: bool = True,
        mmr_lambda: float = 0.6,
    ) -> list[Passage]:
        if not self.chunks:
            return []
        q_tokens = _content_tokens(query)
        bm25_scores = self.bm25.scores(q_tokens)

        method = "bm25"
        rankings = [_rank_order(bm25_scores)]
        query_vec: list[float] | None = None

        if hybrid and self._ensure_vectors():
            try:
                query_vec = self.embedder.embed([query])[0]
                vec_scores = [cosine(query_vec, v) for v in self._vectors]
                rankings.append(_rank_order(vec_scores))
                method = "hybrid"
            except Exception:
                # Embedding backend failed at query time — degrade to BM25.
                query_vec = None

        if method == "hybrid":
            fused = _rrf_fuse(rankings, len(self.chunks))
        else:
            fused = bm25_scores

        order = _rank_order(fused)
        # Keep only chunks with any lexical OR semantic signal.
        order = [i for i in order if fused[i] > 0.0] or order[:top_k]

        if mmr and query_vec is not None and self._vectors is not None:
            order = self._mmr(order, query_vec, top_k, mmr_lambda)
        else:
            order = order[:top_k]

        return [
            Passage(
                text=self.chunks[i].text,
                start_ms=self.chunks[i].start_ms,
                end_ms=self.chunks[i].end_ms,
                score=round(float(fused[i]), 6),
                chunk_index=self.chunks[i].index,
                method=method,
            )
            for i in order
        ]

    def _mmr(
        self,
        candidates: list[int],
        query_vec: list[float],
        top_k: int,
        lam: float,
    ) -> list[int]:
        """Maximal Marginal Relevance: relevance minus redundancy."""
        assert self._vectors is not None
        pool = candidates[: max(top_k * 4, top_k)]
        selected: list[int] = []
        rel = {i: cosine(query_vec, self._vectors[i]) for i in pool}
        while pool and len(selected) < top_k:
            if not selected:
                best = max(pool, key=lambda i: rel[i])
            else:
                best = max(
                    pool,
                    key=lambda i: lam * rel[i]
                    - (1 - lam) * max(cosine(self._vectors[i], self._vectors[j]) for j in selected),
                )
            selected.append(best)
            pool.remove(best)
        return selected


def search_transcript(
    segments,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
    hybrid: bool = True,
    mmr: bool = True,
    video_id: str | None = None,
    title: str | None = None,
) -> list[Passage]:
    """Convenience: chunk caption segments and return ranked passages.

    By default an embedder is resolved automatically (offline-safe). Pass
    ``hybrid=False`` to force pure-BM25 lexical search.
    """
    chunks = chunk_segments(segments)
    if hybrid and embedder is None:
        try:
            embedder = get_embedder()
        except Exception:
            embedder = None
    index = SemanticIndex(chunks, embedder if hybrid else None)
    passages = index.search(query, top_k=top_k, hybrid=hybrid, mmr=mmr)
    for p in passages:
        p.video_id = video_id
        p.title = title
    return passages


def relevance_score(segments, query: str, embedder: Embedder, sample_top: int = 3) -> float:
    """A cross-document relevance score for ``query`` against a transcript.

    Mean cosine of the ``sample_top`` best-matching chunks. Because the same
    (normalised) embedder is used everywhere, these scores ARE comparable across
    videos — the basis for neural re-ranking of search results, exa-style.
    """
    chunks = chunk_segments(segments)
    if not chunks:
        return 0.0
    vecs = embedder.embed([c.text for c in chunks])
    qv = embedder.embed([query])[0]
    sims = sorted((cosine(qv, v) for v in vecs), reverse=True)
    top = sims[:sample_top] or [0.0]
    return sum(top) / len(top)


def highlight_terms(query: str) -> list[str]:
    """Content terms from the query, for caller-side highlighting of passages."""
    return list(dict.fromkeys(_content_tokens(query)))
