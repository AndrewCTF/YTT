"""SQLite-backed cache for search results."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import aiosqlite

from config import config
from .searcher import VideoSearchResult


@dataclass
class CachedSearchResult:
    """A cached search result entry."""
    query: str
    video_id: str
    rank: int
    title: str
    channel_name: str
    duration: str
    view_count: str
    cached_at: datetime
    expires_at: datetime


class SearchCache:
    """SQLite cache for search results with shorter TTL than transcripts."""

    SEARCH_CACHE_TTL_HOURS = 24  # Search results go stale faster

    async def initialize(self) -> None:
        """Create search_results table if not exists."""
        async with aiosqlite.connect(config.CACHE_DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    query       TEXT NOT NULL,
                    video_id    TEXT NOT NULL,
                    rank        INTEGER NOT NULL,
                    title       TEXT NOT NULL,
                    channel     TEXT NOT NULL,
                    duration    TEXT,
                    views       TEXT,
                    cached_at   TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    PRIMARY KEY (query, video_id)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_expires
                ON search_results(expires_at)
            """)
            await db.commit()

    async def get(self, query: str) -> list[CachedSearchResult] | None:
        """Get cached search results for a query."""
        await self.initialize()

        async with aiosqlite.connect(config.CACHE_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM search_results
                   WHERE query = ? AND expires_at > ?
                   ORDER BY rank""",
                (query, datetime.now().isoformat())
            )
            rows = await cursor.fetchall()

        if not rows:
            return None

        results = []
        for row in rows:
            results.append(CachedSearchResult(
                query=row["query"],
                video_id=row["video_id"],
                rank=row["rank"],
                title=row["title"],
                channel_name=row["channel"],
                duration=row["duration"] or "N/A",
                view_count=row["views"] or "N/A",
                cached_at=datetime.fromisoformat(row["cached_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]),
            ))

        return results

    async def set(self, query: str, results: list[VideoSearchResult]) -> None:
        """Cache search results for a query."""
        await self.initialize()

        now = datetime.now()
        expires = now + timedelta(hours=self.SEARCH_CACHE_TTL_HOURS)

        async with aiosqlite.connect(config.CACHE_DB_PATH) as db:
            # Delete old entries for this query
            await db.execute("DELETE FROM search_results WHERE query = ?", (query,))

            # Insert new results
            for rank, result in enumerate(results):
                await db.execute(
                    """INSERT INTO search_results
                       (query, video_id, rank, title, channel, duration, views, cached_at, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (query, result.video_id, rank, result.title, result.channel_name,
                     result.duration, result.view_count, now.isoformat(), expires.isoformat())
                )
            await db.commit()

    async def cleanup_expired(self) -> int:
        """Remove expired search cache entries."""
        await self.initialize()

        async with aiosqlite.connect(config.CACHE_DB_PATH) as db:
            cursor = await db.execute(
                "DELETE FROM search_results WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            await db.commit()
            return cursor.rowcount


search_cache = SearchCache()
