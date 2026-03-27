# YouTube Transcript Fetcher — Full Project Plan

---

## 1. Concept & Goals

Build a **YouTube transcript fetching system from scratch** — no YouTube Data API, no official API key — using YouTube's internal Innertube API. The system should be:
- A **Python library** for programmatic use
- A **CLI tool** for one-off usage
- An **MCP server** for AI tool integration
- **Hybrid-capable** (fall back to Whisper for rate-limited or caption-less videos)

---

## 2. The Two Approaches Compared

### Approach A — Scrape Innertube API (This Project)
| Factor | Detail |
|--------|--------|
| **Cost** | Free |
| **Speed** | ~0.5–2s per video |
| **Coverage** | ~85% of videos (many have auto-generated captions) |
| **Reliability** | Rate limited, IP blocks, Innertube changes without notice |
| **Quality** | Depends on video — ASR captions can be mediocre |
| **Scale** | Hit walls fast (~5 req/10 sec per IP) |

### Approach B — Generate with Whisper (Local AI)
| Factor | Detail |
|--------|--------|
| **Cost** | Compute only (GPU recommended, CPU works) |
| **Speed** | ~1–3x video duration (e.g., 10-min video → 10–30s processing) |
| **Coverage** | 99%+ — works on ANY video with audio |
| **Reliability** | 100% — no external dependencies |
| **Quality** |state-of-the-art, often better than YouTube's auto-captions |
| **Scale** | Limited by your compute |

### Recommended Strategy: HYBRID

```
Try Innertube scrape first
    → Success + within rate limit? → Return transcript
    → Rate limited or no captions? → Fall back to Whisper
```

This gives you **speed when available** and **reliability when not**.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI / API / MCP                    │
├─────────────────────────────────────────────────────────┤
│                    transcript_service                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  cache_store  │  │  rate_limiter │  │  fallback_   │  │
│  │  (SQLite/Redis)│  │  (token bucket)│  │  whisper     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Innertube Fetcher (primary)               │
│  video_page → extract API key → player POST → baseUrl   │
├─────────────────────────────────────────────────────────┤
│              Whisper Runner (fallback)                  │
│  yt-dlp audio extract → Whisper → timestamp + text     │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Project Structure

```
yt_transcript/
├── src/
│   ├── __init__.py
│   ├── fetcher.py          # Innertube API (primary)
│   ├── whisper_runner.py   # Whisper fallback
│   ├── parser.py           # Parse captionTracks / Whisper output
│   ├── formatters.py       # text / SRT / VTT / JSON
│   ├── cache.py            # SQLite cache (avoid re-fetching)
│   ├── rate_limiter.py     # Token bucket for Innertube calls
│   ├── service.py          # High-level service (tries scrape → falls back)
│   └── exceptions.py
├── mcp_server/
│   ├── __init__.py
│   ├── server.py           # FastMCP server definition
│   ├── tools.py            # MCP tool definitions
│   └── prompts.py          # MCP prompt templates
├── cli.py                  # CLI entrypoint
├── main.py                 # Library entrypoint
├── requirements.txt
├── requirements-mcp.txt
├── Dockerfile
└── README.md
```

---

## 5. Core Implementation Details

### 5.1 Innertube Fetcher (`fetcher.py`)

**Endpoints:**
- Video page: `GET https://www.youtube.com/watch?v={videoId}`
- Player API: `POST https://www.youtube.com/youtubei/v1/player?key={apiKey}`
- Caption data: `GET https://www.youtube.com/api/timedtext?...&fmt=json3`

**Extraction flow:**
1. GET video page → extract `INNERTUBE_API_KEY` via regex
2. POST to player endpoint with Android client context
3. Extract `captionTracks[].baseUrl` from response
4. GET `baseUrl` with `&fmt=json3` → parse JSON3 events

**Client context (Android works best):**
```python
payload = {
    "context": {
        "client": {
            "clientName": "ANDROID",
            "clientVersion": "20.10.38",
        }
    },
    "videoId": video_id,
}
```

**Return format (JSON3 events):**
```json
{
  "events": [
    {
      "tStartMs": 0,
      "dDurationMs": 4000,
      "segs": [{ "utf8": "Hello world" }]
    }
  ]
}
```

### 5.2 Whisper Runner (`whisper_runner.py`)

**Dependency:** `yt-dlp` (for audio extraction) + `openai-whisper` (or `faster-whisper`)

**Flow:**
```bash
yt-dlp -x --audio-format mp3 --audio-quality 0 -o audio.mp4 https://youtube.com/watch?v={id}
whisper audio.mp4 --model base --output_format json > transcript.json
```

**Output from Whisper includes:**
- Word-level timestamps
- Segment-level timestamps and text
- Language detection

### 5.3 Cache Layer (`cache.py`)

SQLite database to store fetched transcripts:

```sql
CREATE TABLE transcripts (
    video_id    TEXT PRIMARY KEY,
    lang        TEXT,
    source      TEXT,        -- 'innertube' or 'whisper'
    raw_data    TEXT,        -- JSON
    created_at  TIMESTAMP,
    expires_at  TIMESTAMP   -- cache TTL (7 days)
);
```

**Why cache?** Avoids re-hitting YouTube for the same video. MCP servers handle many concurrent requests — cache is critical.

### 5.4 Rate Limiter (`rate_limiter.py`)

Token bucket algorithm:
- Bucket size: 5 tokens
- Refill rate: 1 token per 2 seconds
- When bucket empty: wait `retry_after` from 429 response OR wait for refill

```python
class RateLimiter:
    def __init__(self, rate: float = 0.5, burst: int = 5):
        self.rate = rate        # tokens per second
        self.burst = burst      # max tokens
        self.tokens = burst
        self.last_refill = time.time()

    async def acquire(self):
        while self.tokens < 1:
            self._refill()
            if self.tokens < 1:
                await asyncio.sleep(0.1)
        self.tokens -= 1
```

### 5.5 Service Layer (`service.py`)

The smart router:

```python
async def get_transcript(video_id: str, lang: str = "en") -> Transcript:
    # 1. Check cache
    cached = await cache.get(video_id, lang)
    if cached:
        return cached

    # 2. Try Innertube (with rate limiter)
    try:
        result = await innertube_fetch(video_id, lang)
        await cache.set(video_id, lang, result, source="innertube")
        return result
    except RateLimitedError:
        pass  # fall through to Whisper

    # 3. Fall back to Whisper
    result = await whisper_run(video_id, lang)
    await cache.set(video_id, lang, result, source="whisper")
    return result
```

---

## 6. Handling Rate Limits (The Key Question)

### Problem
- YouTube Innertube: ~5 requests per 10 seconds per IP
- Cloud provider IPs (AWS, GCP, Azure) are often blocked
- YouTube returns HTTP 429 when rate limited

### Solutions

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Whisper fallback** | On 429 → auto-fallback to Whisper | 100% reliability, no dependency | Slower, requires compute |
| **Cache everything** | SQLite → never re-fetch same video | Eliminates redundant calls | Uses disk space |
| **Distributed IP pool** | Multiple outbound IPs (proxy rotation) | Higher throughput at scale | Complexity, cost |
| **Request coalescing** | Multiple requests for same video → single fetch | Reduces duplicate work | Code complexity |
| **Exponential backoff** | On 429: wait 2^n seconds up to max | Respects YouTube, self-healing | Slow under heavy load |
| **Async queuing** | MCP requests queued with rate-limited executor | MCP clients don't block | Needs queue infrastructure |

### For MCP: Rate Limits Are NOT a Problem Because...

1. **Cache is the primary defense** — MCP calls for the same video return instantly from SQLite
2. **Whisper fallback** — when Innertube hits a wall, Whisper picks up seamlessly
3. **Request deduplication** — if 10 MCP clients ask for the same video simultaneously, only ONE actual fetch happens (others wait/return cached)
4. **Async batching** — MCP tool can accept a list of video IDs and process them with built-in delays
5. **Per-MCP-client rate limiting** — the service itself enforces limits; individual MCP clients are isolated

```
MCP Client A ──┐
MCP Client B ──┼──→ [Service Queue] → Rate Limiter → Innertube ─→ Cache
MCP Client C ──┘              │                          │
                              └── Whisper (fallback) ────┘
```

---

## 7. MCP Server Design

### Tools

#### `get_transcript`
```python
@tool
async def get_transcript(
    video_id: str,
    lang: str = "en",
    format: str = "text"  # text | srt | vtt | json
) -> str:
    """Get transcript for a YouTube video. Falls back to Whisper if needed."""
```

#### `get_transcripts_batch`
```python
@tool
async def get_transcripts_batch(
    video_ids: list[str],
    lang: str = "en",
    format: str = "text"
) -> list[dict]:  # [{video_id, transcript, source, success}]
    """Get transcripts for multiple videos. Handles rate limits internally."""
```

#### `transcript_search`
```python
@tool
async def transcript_search(
    query: str,
    channel_id: str | None = None,
    max_results: int = 5
) -> list[dict]:  # [{video_id, title, transcript_snippet, timestamp}]
    """Search within transcripts (requires local index or additional search layer)."""
```

### MCP Server Setup

```python
# mcp_server/server.py
from fastmcp import FastMCP

mcp = FastMCP("yt-transcript")

@mcp.tool()
async def get_transcript(video_id: str, lang: str = "en", format: str = "text"):
    return await transcript_service.get_transcript(video_id, lang, format)

@mcp.tool()
async def get_transcripts_batch(video_ids: list[str], lang: str = "en", format: str = "text"):
    results = []
    for vid in video_ids:
        results.append(await transcript_service.get_transcript(vid, lang, format))
    return results
```

---

## 8. AI-Ready Features

### Why This Is Great for AI

1. **Structured output** — JSON with word-level timestamps, ideal for RAG systems
2. **Ground truth** — Use transcripts to fine-tune or evaluate other AI models
3. **Multi-language** — Whisper supports 100+ languages natively
4. **Training data** — SRT/VTT files are standard formats for training data pipelines
5. **No API cost** — At scale, YouTube Data API costs money; Innertube + Whisper is free
6. **Fresh data** — Real-time fetching vs cached API responses
7. **Audio + Transcript** — Whisper fallback gives you both audio embeddings + transcript

### Output Formats for AI

```python
# JSON (best for RAG / structured data)
{
  "video_id": "...",
  "title": "...",
  "language": "en",
  "segments": [
    {"start": 0.0, "end": 4.5, "text": "Hello world"},
    {"start": 4.5, "end": 9.2, "text": "Welcome to this video"}
  ],
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.4},
    ...
  ]
}
```

```python
# SRT (best for subtitle training, ASR training)
# VTT (WebVTT for web/video players)
# Plain text (LLM context windows)
```

### MCP Integration for AI

- **Claude/AI agents** can call `get_transcript` directly as an MCP tool
- **Context window efficient** — transcript is pre-processed, not raw video
- **Timestamped** — AI can reference exact moments in video
- **Batch mode** — AI can process entire playlists/channels at once

---

## 9. Implementation Roadmap

### Phase 1 — Core Library (Day 1)
- [ ] `exceptions.py`
- [ ] `fetcher.py` — Innertube fetch + parse
- [ ] `formatters.py` — text/SRT/VTT/JSON output
- [ ] `main.py` — library interface
- [ ] Test with known video IDs

### Phase 2 — Whisper Fallback (Day 1–2)
- [ ] `whisper_runner.py` — yt-dlp audio + Whisper
- [ ] `service.py` — smart routing (scrape → fallback)
- [ ] Verify Whisper output matches Innertube format

### Phase 3 — Reliability (Day 2–3)
- [ ] `rate_limiter.py` — token bucket
- [ ] `cache.py` — SQLite persistence
- [ ] Retry logic with exponential backoff
- [ ] Handle 429 / IP blocks gracefully

### Phase 4 — MCP Server (Day 3–4)
- [ ] FastMCP server setup
- [ ] `get_transcript` tool
- [ ] `get_transcripts_batch` tool
- [ ] Docker containerization

### Phase 5 — Polish (Day 4–5)
- [ ] CLI improvements (progress bars, verbose mode)
- [ ] Playlist support (`--playlist` flag)
- [ ] Video URL parsing (handle `youtu.be/ID` and full URLs)
- [ ] Language auto-detection
- [ ] Unit tests

---

## 10. Dependencies

```
# Core
requests
httpx
beautifulsoup4

# Whisper fallback
yt-dlp
openai-whisper   # OR faster-whisper (faster, CPU-friendly)

# MCP
fastmCP

# Cache
aiosqlite  # async SQLite

# CLI
click
rich      # pretty output

# Utils
tiktoken   # token counting for AI context
```

---

## 11. Configuration

```python
# config.py
class Config:
    INNERTUBE_CLIENT = "ANDROID"
    INNERTUBE_CLIENT_VERSION = "20.10.38"
    RATE_LIMIT_RATE = 0.5      # tokens per second
    RATE_LIMIT_BURST = 5       # max concurrent
    CACHE_TTL_DAYS = 7
    WHISPER_MODEL = "base"      # tiny/base/small/medium/large
    WHISPER_FALLBACK_ENABLED = True
    MAX_BATCH_SIZE = 50         # max videos per batch request
```

---

## 12. Key Insights Summary

| Question | Answer |
|---|---|
| **Is generating own captions more efficient?** | For reliability and quality: YES. For speed on cache hits: NO. Use hybrid. |
| **How to overcome rate limits?** | Cache everything, Whisper fallback, request coalescing, async queuing |
| **How to be more efficient?** | SQLite cache, Whisper fallback avoids retries, batch processing, connection pooling |
| **How is this good for AI?** | Timestamped JSON output, RAG-ready format, free vs API costs, Whisper gives word-level timestamps |
| **How does MCP solve rate limits?** | MCP layer has cache + queue; individual clients don't hit YouTube directly; Whisper fallback on 429 |
| **Won't multiple MCP requests still rate limit?** | Only if they're distinct uncached videos. Cache + queue + coalescing = rate limits never reach the user |

---

## 13. Why This Beats Using YouTube Data API

| YouTube Data API v3 | This Project (Innertube + Whisper) |
|---|---|
| Requires API key | No key needed |
| Quota costs money at scale | Free |
| Caption track endpoint requires OAuth | No auth needed |
| Official, stable (but rate limited) | Undocumented, can break |
| ~10,000 units/day free | Unlimited (compute-bound) |
| No Whisper fallback | Whisper fallback = 99%+ reliability |
| Single endpoint | Handles ANY video with audio |

