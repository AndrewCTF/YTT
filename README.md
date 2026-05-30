# yttranscript-mcp

Fetch, clean, and search YouTube transcripts — **captions-first** (fast, light, rate‑limit resistant) with an optional local **Whisper** fallback, an **MCP server**, a CLI, and an async Python library. No YouTube Data API key required.

Built for feeding transcripts to LLMs: the `clean` format strips rolling auto‑caption duplication, HTML entities, markup, and timestamps so you spend the fewest tokens possible.

## Why it avoids rate limits

- **Captions first, audio last.** It pulls YouTube's own caption tracks (a few KB) instead of downloading audio. Whisper only runs when no captions exist.
- **No watch‑page scrape on the hot path.** A hardcoded public Innertube key skips the ~1 MB HTML fetch that triggers bot detection.
- **Realistic headers.** Every request sends a browser/app `User-Agent` and language headers (the default `python-requests` UA is the #1 cause of 429s).
- **Multi‑client fallback.** Tries `ANDROID_VR → WEB → MWEB → watch‑page` until one returns captions, sidestepping client‑specific blocks.
- **Retry/backoff.** Transient 429/5xx are retried with exponential backoff + jitter, honoring `Retry-After`.
- **Escape hatches.** `YTT_PROXY` and `YTT_COOKIES_FILE` route around IP‑level blocks and age/region gates.
- **Caching.** SQLite cache avoids re‑fetching.

## Install

```bash
pip install yttranscript-mcp            # core: captions + CLI + library
pip install "yttranscript-mcp[mcp]"     # + MCP server
pip install "yttranscript-mcp[whisper]" # + local Whisper fallback (needs ffmpeg)
pip install "yttranscript-mcp[whisper,torch]"  # + GPU
```

The Whisper fallback needs `ffmpeg`: `winget install ffmpeg` (Windows) · `brew install ffmpeg` (macOS) · `sudo apt install ffmpeg` (Linux).

## CLI

```bash
ytt transcript VIDEO_ID                     # clean, LLM-ready text (default)
ytt transcript "https://youtu.be/VIDEO_ID"  # URLs work too
ytt transcript VIDEO_ID -f json             # or text | srt | vtt
ytt transcript VIDEO_ID -o out.txt          # save to file
ytt transcript VIDEO_ID --no-whisper        # captions only, never download audio

ytt batch VID1 VID2 VID3                     # many videos, concurrent
ytt search "python tutorial" -n 5            # search
ytt search "python tutorial" --with-transcripts
ytt cache-stats [--clean]
```

## Python library

```python
import asyncio
from ytt import get_transcript, search, search_and_get_transcripts

async def main():
    # clean = deduplicated, no timestamps — best for LLMs
    r = await get_transcript("dQw4w9WgXcQ", output_format="clean")
    print(r.source, r.language, r.content[:200])

    for v in await search("python tutorial", max_results=5):
        print(v.video_id, v.title)

    for video, transcript in await search_and_get_transcripts("python", max_results=3):
        if transcript:
            print(video.title, "→", transcript.content[:80])

asyncio.run(main())
```

`output_format`: `clean` (default for the CLI/MCP), `text`, `json`, `srt`, `vtt`.

## MCP server

```bash
yttranscript-mcp          # or: python -m ytt.mcp.server
```

Claude Desktop / Cursor config:

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "yttranscript-mcp"
    }
  }
}
```

Tools: `get_transcript`, `get_transcripts_batch`, `search_videos` (all default to the `clean` format), plus `setup_gpu` / `download_cuda`.

## Configuration (environment variables)

Everything is tunable without code via `YTT_*` env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `YTT_PROXY` | – | `http://user:pass@host:port` — route every request through a proxy |
| `YTT_COOKIES_FILE` | – | Netscape `cookies.txt` for age/region‑restricted videos |
| `YTT_MAX_RETRIES` | `4` | Retry attempts on 429/5xx |
| `YTT_RATE` / `YTT_BURST` | `1.0` / `5` | Client‑side token‑bucket rate limit |
| `YTT_TIMEOUT` | `15` | Per‑request timeout (s) |
| `YTT_CACHE_DB` | `.transcript_cache.db` | Cache path |
| `YTT_CACHE_TTL_DAYS` | `7` | Cache TTL |
| `YTT_WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large` |
| `YTT_WHISPER_GPU` | `true` | Use GPU for Whisper if available |

## How it works

```
video id / url → cache → captions (ANDROID_VR → WEB → MWEB → watch page)
                                  ↓ none?
                          Whisper fallback (download audio, transcribe locally)
                                  ↓
                          cache → format (clean / text / json / srt / vtt)
```

## Development

```bash
uv sync --extra dev --extra mcp
uv run pytest
uv run ruff check ytt/ && uv run black --check ytt/
```

## License

MIT
