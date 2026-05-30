"""Local corpus index — semantic search *across many videos*.

This is the "exa for your YouTube library" piece: index a set of videos once,
then ask a natural-language question and get the most relevant timestamped
passages from *any* of them, ranked across the whole corpus. It is a single
self-contained SQLite file; embeddings are stored inline as packed float32 so
search needs no model reload when an embedding backend is configured.

Everything stays on disk and on the machine. With no embedding backend it still
works as a fast cross-video BM25 search; add one (``YTT_EMBED_PROVIDER=ollama``)
and it fuses lexical + dense ranking automatically.
"""

from __future__ import annotations

import array
import sqlite3
import time
from dataclasses import dataclass

from .config import config
from .embeddings import Embedder, cosine
from .semantic import (
    BM25,
    Passage,
    _content_tokens,
    _rank_order,
    _rrf_fuse,
    chunk_segments,
)


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


@dataclass
class IndexedVideo:
    video_id: str
    title: str
    language: str
    n_chunks: int
    embedder: str
    added_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title    TEXT,
    language TEXT,
    n_chunks INTEGER,
    embedder TEXT,
    added_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id  TEXT NOT NULL,
    idx       INTEGER,
    start_ms  INTEGER,
    end_ms    INTEGER,
    text      TEXT,
    embedder  TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
"""


class CorpusIndex:
    """A SQLite-backed, cross-video semantic index."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.INDEX_DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CorpusIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- writes ---------------------------------------------------------
    def add_video(
        self,
        video_id: str,
        title: str,
        segments,
        language: str = "en",
        embedder: Embedder | None = None,
    ) -> int:
        """Chunk, (optionally) embed, and store one video. Replaces if present."""
        chunks = chunk_segments(segments)
        embedder_name = embedder.name if embedder else ""
        vectors: list[list[float]] = []
        if embedder and chunks:
            vectors = embedder.embed([c.text for c in chunks])

        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        cur.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        cur.execute(
            "INSERT INTO videos (video_id, title, language, n_chunks, embedder, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, title, language, len(chunks), embedder_name, time.time()),
        )
        for i, c in enumerate(chunks):
            blob = _pack(vectors[i]) if vectors else None
            cur.execute(
                "INSERT INTO chunks (video_id, idx, start_ms, end_ms, text, embedder, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (video_id, c.index, c.start_ms, c.end_ms, c.text, embedder_name, blob),
            )
        self.conn.commit()
        return len(chunks)

    def remove_video(self, video_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        cur.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---- reads ----------------------------------------------------------
    def has_video(self, video_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return row is not None

    def list_videos(self) -> list[IndexedVideo]:
        rows = self.conn.execute(
            "SELECT video_id, title, language, n_chunks, embedder, added_at "
            "FROM videos ORDER BY added_at DESC"
        ).fetchall()
        return [
            IndexedVideo(
                video_id=r["video_id"],
                title=r["title"] or "",
                language=r["language"] or "",
                n_chunks=r["n_chunks"] or 0,
                embedder=r["embedder"] or "",
                added_at=r["added_at"] or 0.0,
            )
            for r in rows
        ]

    def stats(self) -> dict:
        n_videos = self.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedders = [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT embedder FROM videos WHERE embedder != ''"
            ).fetchall()
        ]
        return {"videos": n_videos, "chunks": n_chunks, "embedders": embedders}

    def search(
        self,
        query: str,
        top_k: int = 8,
        embedder: Embedder | None = None,
        hybrid: bool = True,
    ) -> list[Passage]:
        """Hybrid (BM25 + dense) search across every indexed video."""
        rows = self.conn.execute(
            "SELECT c.video_id, c.idx, c.start_ms, c.end_ms, c.text, c.embedder, c.embedding, "
            "v.title FROM chunks c LEFT JOIN videos v ON c.video_id = v.video_id"
        ).fetchall()
        if not rows:
            return []

        texts = [r["text"] for r in rows]
        bm25 = BM25([_content_tokens(t) for t in texts])
        bm25_scores = bm25.scores(_content_tokens(query))

        method = "bm25"
        rankings = [_rank_order(bm25_scores)]

        if hybrid and embedder is not None:
            # Only fuse vectors that were built with the *same* embedder.
            usable = all(r["embedder"] == embedder.name and r["embedding"] for r in rows)
            if usable:
                try:
                    qv = embedder.embed([query])[0]
                    vec_scores = [cosine(qv, _unpack(r["embedding"])) for r in rows]
                    rankings.append(_rank_order(vec_scores))
                    method = "hybrid"
                except Exception:
                    method = "bm25"

        fused = _rrf_fuse(rankings, len(rows)) if method == "hybrid" else bm25_scores
        order = _rank_order(fused)
        order = [i for i in order if fused[i] > 0.0][:top_k] or order[:top_k]

        return [
            Passage(
                text=rows[i]["text"],
                start_ms=rows[i]["start_ms"],
                end_ms=rows[i]["end_ms"],
                score=round(float(fused[i]), 6),
                chunk_index=rows[i]["idx"],
                video_id=rows[i]["video_id"],
                title=rows[i]["title"],
                method=method,
            )
            for i in order
        ]
