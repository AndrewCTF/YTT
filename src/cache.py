"""SQLite-backed transcript cache for avoiding redundant fetches."""

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from config import config


@dataclass
class CachedTranscript:
    """A cached transcript entry."""
    video_id: str
    language: str
    source: str          # 'innertube' or 'whisper'
    raw_data: dict       # JSON representation
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() >= self.expires_at


def _row_to_cached_transcript(row: tuple) -> CachedTranscript:
    """Convert a database row to a CachedTranscript."""
    video_id, language, source, raw_data, created_at_str, expires_at_str = row

    # Parse timestamps
    created_at = datetime.fromisoformat(created_at_str)
    expires_at = datetime.fromisoformat(expires_at_str)

    # Parse raw_data JSON
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)

    return CachedTranscript(
        video_id=video_id,
        language=language,
        source=source,
        raw_data=raw_data,
        created_at=created_at,
        expires_at=expires_at,
    )


class TranscriptCache:
    """SQLite-backed cache for transcript data.

    Stores fetched transcripts to avoid redundant API calls.
    Cache entries expire after CACHE_TTL_DAYS (default 7 days).
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.CACHE_DB_PATH
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS transcripts (
                        video_id    TEXT NOT NULL,
                        lang        TEXT NOT NULL,
                        source      TEXT NOT NULL,
                        raw_data    TEXT NOT NULL,
                        created_at  TEXT NOT NULL,
                        expires_at  TEXT NOT NULL,
                        PRIMARY KEY (video_id, lang)
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at
                    ON transcripts(expires_at)
                """)
                await db.commit()

            self._initialized = True

    async def get(self, video_id: str, language: str = "en") -> CachedTranscript | None:
        """Get a cached transcript if it exists and is not expired.

        Args:
            video_id: The YouTube video ID.
            language: The language code (default: 'en').

        Returns:
            CachedTranscript if found and not expired, None otherwise.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT video_id, lang, source, raw_data, created_at, expires_at
                FROM transcripts
                WHERE video_id = ? AND lang = ?
                """,
                (video_id, language),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        transcript = _row_to_cached_transcript(tuple(row))

        # Check if expired
        if transcript.is_expired():
            await self.delete(video_id, language)
            return None

        return transcript

    async def set(
        self,
        video_id: str,
        language: str,
        raw_data: dict,
        source: str,
        ttl_days: int | None = None,
    ) -> None:
        """Store a transcript in the cache.

        Args:
            video_id: The YouTube video ID.
            language: The language code.
            raw_data: The transcript data as a dict.
            source: 'innertube' or 'whisper'.
            ttl_days: Override the default TTL.
        """
        await self.initialize()

        ttl = ttl_days or config.CACHE_TTL_DAYS
        now = datetime.now()
        expires_at = now + timedelta(days=ttl)

        raw_json = json.dumps(raw_data, ensure_ascii=False)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO transcripts
                (video_id, lang, source, raw_data, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    language,
                    source,
                    raw_json,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            await db.commit()

    async def delete(self, video_id: str, language: str = "en") -> None:
        """Delete a cached transcript.

        Args:
            video_id: The YouTube video ID.
            language: The language code.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM transcripts WHERE video_id = ? AND lang = ?",
                (video_id, language),
            )
            await db.commit()

    async def cleanup_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of entries removed.
        """
        await self.initialize()

        now = datetime.now().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM transcripts WHERE expires_at < ?",
                (now,),
            )
            await db.commit()
            return cursor.rowcount

    async def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with total_entries, expired_entries, and entries_by_source.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Total entries
            cursor = await db.execute("SELECT COUNT(*) FROM transcripts")
            total = (await cursor.fetchone())[0]

            # Expired entries
            now = datetime.now().isoformat()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM transcripts WHERE expires_at < ?",
                (now,),
            )
            expired = (await cursor.fetchone())[0]

            # By source
            cursor = await db.execute(
                "SELECT source, COUNT(*) FROM transcripts GROUP BY source"
            )
            by_source = dict(await cursor.fetchall())

        return {
            "total_entries": total,
            "expired_entries": expired,
            "entries_by_source": by_source,
        }


# Global cache instance
cache = TranscriptCache()