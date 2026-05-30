# yttranscript-mcp

Fetch, clean, **semantically search**, and ask questions about YouTube transcripts — **captions-first** (fast, light, rate‑limit resistant) with an optional local **Whisper** fallback, an **MCP server**, a CLI, and an async Python library. No YouTube Data API key required. **Everything runs locally.**

Built for feeding transcripts to LLMs: the `clean` format strips rolling auto‑caption duplication, HTML entities, markup, and timestamps so you spend the fewest tokens possible.

### What makes it special

- **Search *inside* videos.** Ask a natural-language question; get the exact timestamped moments with deep-link URLs (`https://youtu.be/ID?t=123`). Hybrid **BM25 + local embeddings**, fused with Reciprocal Rank Fusion and diversified with MMR — exa-quality retrieval, **fully local, zero new dependencies** (it works with no model at all and gets better when you add one).
- **Ask questions (local RAG).** `ytt ask ID "question"` retrieves the relevant passages and, if a local LLM is running, writes a grounded answer that cites timestamps. No LLM? You still get the cited passages.
- **Cross-video corpus search.** Index a library of videos once (`ytt index …`), then `ytt find "query"` searches across all of them — exa for your own YouTube collection, in a single SQLite file.
- **Beats `yt-dlp` for transcripts, audio-free.** List every caption language (`ytt langs`), **machine-translate captions into any language** (`--translate`), and dump rich metadata + chapters (`ytt info`) — all from the lightweight captions path, no video download.

### vs. the tools you already use

| | **yttranscript-mcp** | `yt-dlp` | exa |
|---|---|---|---|
| Captions / subtitles | ✅ captions-first, multi-client anti-throttle | ✅ (downloads via page scrape) | ❌ |
| List caption languages | ✅ `ytt langs` | ✅ `--list-subs` | ❌ |
| Translate captions | ✅ `--translate es` | ✅ | ❌ |
| Metadata + chapters (no download) | ✅ `ytt info` | ⚠️ `--dump-json` (heavier) | ❌ |
| Semantic search inside content | ✅ timestamped passages | ❌ | ✅ (web, cloud) |
| Neural/RAG question answering | ✅ local | ❌ | ✅ (cloud) |
| Cross-document semantic search | ✅ local corpus index | ❌ | ✅ (cloud) |
| Runs fully local / private | ✅ | ✅ | ❌ |

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
ytt transcript VIDEO_ID --translate es      # machine-translate captions to Spanish

ytt batch VID1 VID2 VID3                     # many videos, concurrent
ytt search "python tutorial" -n 5            # YouTube keyword search
ytt search "python tutorial" --rank          # NEURAL re-rank by transcript relevance
ytt search "python tutorial" --with-transcripts
ytt cache-stats [--clean]
```

### Semantic search & Q&A (fully local)

```bash
# Metadata + chapters, no audio download (beats yt-dlp --dump-json for the essentials)
ytt info VIDEO_ID

# Every available caption language + translation targets (like yt-dlp --list-subs)
ytt langs VIDEO_ID

# Search INSIDE a video — ask a question, get the exact timestamped moments
ytt ask VIDEO_ID "how does the event loop schedule coroutines?"
ytt ask VIDEO_ID "what's the main argument?" --passages-only   # skip the LLM answer

# Build a local corpus index, then search across ALL indexed videos
ytt index VID1 VID2 VID3
ytt find "retrieval augmented generation tradeoffs"
ytt corpus                                   # index stats
```

`ask` retrieves the most relevant passages and (if a local LLM is reachable) writes a grounded answer citing timestamps; otherwise it just returns the cited passages. `find` searches your whole indexed library. Both print deep-link URLs straight to the moment in each video.

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

Semantic search, RAG, and the cross-video corpus index are all public too:

```python
import asyncio
from ytt import (
    search_in_video, ask_video, index_videos, find_in_corpus, get_video_info, search_ranked,
)

async def main():
    # Search inside one video → timestamped passages with deep-link URLs.
    res = await search_in_video("VIDEO_ID", "the key tradeoffs")
    for p in res.passages:
        print(p.timestamp, p.score, p.url(), p.text[:80])

    # Local RAG: grounded answer + cited passages (falls back to passages if no LLM).
    a = await ask_video("VIDEO_ID", "what is the main conclusion?")
    print(a.answer or a.note)

    # Cross-video semantic search over a local corpus index.
    await index_videos(["VID1", "VID2", "VID3"])
    for p in await find_in_corpus("retrieval augmented generation"):
        print(p.title, p.timestamp, p.url())

    # Neural re-ranking of YouTube results by transcript relevance.
    for video, score in await search_ranked("python asyncio", max_results=5):
        print(f"{score:.3f}", video.title)

    meta = await get_video_info("VIDEO_ID")
    print(meta.title, meta.view_count, "views;", len(meta.chapters), "chapters")

asyncio.run(main())
```

### Embedding backends (how "fully local" stays local)

Semantic search resolves an embedder in this order, and **never needs the network by default**:

| `YTT_EMBED_PROVIDER` | Backend | Notes |
|---|---|---|
| `auto` (default) | `sentence-transformers` if installed, else `hash` | Offline-safe, zero config |
| `hash` | Dependency-free hashing embedder | Instant, deterministic, no install |
| `sentence-transformers` | Local neural model (CPU/GPU) | `pip install sentence-transformers` |
| `ollama` | Local Ollama embedding model | Best quality; `ollama pull nomic-embed-text` |
| `openai` | Any OpenAI-compatible `/v1/embeddings` | Point at localhost (llama.cpp, LM Studio, vLLM) |

With no embedding model at all, search still works as fast cross-video **BM25**. Add a real embedder and it automatically upgrades to **hybrid lexical + dense** ranking. Nothing leaves the machine unless you configure a remote endpoint.

`output_format`: `clean` (default for the CLI/MCP), `text`, `json`, `srt`, `vtt`, `summary`.

## Local summarization (save tokens)

Compress a transcript to a short summary with a **local** LLM, so the agent that
consumes it ingests a summary instead of the whole transcript. Nothing leaves
the machine. Opt-in; needs a local model (default [Ollama](https://ollama.com)).

```bash
ollama serve
ollama pull qwen3.6:27b      # default — smaller: qwen3:8b, qwen3:4b, qwen3.5:2b
```

```bash
ytt transcript VIDEO_ID --summarize
ytt transcript VIDEO_ID --summarize --summary-model qwen3:8b
```

```python
from ytt import get_transcript
r = await get_transcript("VIDEO_ID", output_format="summary")
print(r.content)   # bullet-point summary
```

- Long transcripts are summarized map-reduce (chunk → reduce) to fit context.
- `YTT_SUMMARY_KEEP_ALIVE` keeps the model hot between calls (`-1` = forever).
- `YTT_SUMMARY_AUTO_PULL=1` pulls a missing model on demand.
- `YTT_SUMMARY_PROVIDER=openai` targets any OpenAI-compatible server
  (llama.cpp, LM Studio, vLLM) via `YTT_SUMMARY_OPENAI_BASE`.

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

Tools:
- **Transcripts:** `get_transcript`, `get_transcripts_batch` (default to the `clean` format), `summarize_video` (local-LLM summary)
- **Metadata:** `get_video_info` (title/channel/views/chapters, no download), `list_caption_languages` (tracks + translation targets)
- **Semantic (fully local):** `search_transcript` (timestamped passages inside a video), `ask_video` (local RAG with cited timestamps), `index_videos` + `find_in_corpus` (cross-video search)
- **Search:** `search_videos` (add `rank=True` for neural transcript-relevance re-ranking)
- **GPU:** `setup_gpu`, `download_cuda`

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
| `YTT_SUMMARY_PROVIDER` | `ollama` | `ollama` or `openai` (OpenAI-compatible) |
| `YTT_SUMMARY_MODEL` | `qwen3.6:27b` | Local summary model |
| `YTT_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `YTT_SUMMARY_KEEP_ALIVE` | `5m` | Keep model hot (`-1` = forever) |
| `YTT_SUMMARY_AUTO_PULL` | `false` | Pull missing model on demand |
| `YTT_EMBED_PROVIDER` | `auto` | `auto`/`hash`/`sentence-transformers`/`ollama`/`openai` |
| `YTT_EMBED_MODEL` | – | Embedding model (e.g. `nomic-embed-text`) |
| `YTT_CHUNK_CHARS` / `YTT_CHUNK_OVERLAP` | `480` / `80` | Retrieval window size + overlap |
| `YTT_INDEX_DB` | `.ytt_index.db` | Corpus index path |

## How it works

```
video id / url → cache → captions (ANDROID_VR → WEB → MWEB → watch page)
                                  ↓ none?
                          Whisper fallback (download audio, transcribe locally)
                                  ↓
                          cache → format (clean / text / json / srt / vtt)
```

Semantic search (`ask` / `find` / `search --rank`) runs on top, fully local:

```
transcript → timestamped chunks (rolling-caption dedup, overlap)
                  ↓
         BM25 (always) + dense embeddings (if a local backend is configured)
                  ↓
         Reciprocal Rank Fusion → MMR diversity → ranked passages w/ deep links
                  ↓ (ask only)
         local LLM writes a grounded answer citing the timestamps
```

## Development

```bash
uv sync --extra dev --extra mcp
uv run pytest
uv run ruff check ytt/ && uv run black --check ytt/
```

## License

MIT
