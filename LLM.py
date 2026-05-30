# Adding YouTube Transcript Tools to an LLM Agent (MCP)

This guide helps LLM agents (OpenCLAW, Claude Code, Cursor, etc.) add YouTube
transcript + search capabilities via the Model Context Protocol (MCP).

**PyPI:** `yttranscript-mcp` · **Repo:** https://github.com/AndrewCTF/YTT

## Install

```bash
pip install "yttranscript-mcp[mcp]"            # captions path (no audio/ML deps)
pip install "yttranscript-mcp[mcp,whisper]"    # + local Whisper fallback (needs ffmpeg)
```

This installs the `yttranscript-mcp` console command that launches the server.

## Tools exposed

1. `get_transcript` — transcript for one video
2. `get_transcripts_batch` — transcripts for many videos, concurrently
3. `search_videos` — search YouTube (optionally with transcripts)
4. `setup_gpu` / `download_cuda` — optional CUDA setup for Whisper

All transcript tools default to the **`clean`** format: deduplicated,
timestamp-free text optimized for LLM ingestion (fewest tokens).

## Configuration

### Claude Code / Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "yttranscript-mcp"
    }
  }
}
```

Or run from a source checkout without installing:

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "python",
      "args": ["-m", "ytt.mcp.server"],
      "cwd": "/absolute/path/to/YTT"
    }
  }
}
```

### Beating rate limits (optional env)

The captions path is already rate-limit resistant (browser headers, multi-client
fallback, retry/backoff). From a throttled cloud IP, add:

```json
"env": {
  "YTT_PROXY": "http://user:pass@host:port",
  "YTT_COOKIES_FILE": "/path/to/cookies.txt",
  "YTT_MAX_RETRIES": "6"
}
```

`YTT_AUTO_DOWNLOAD_CUDA=1` enables on-demand download of NVIDIA CUDA libraries
for GPU Whisper (avoids bundling NVIDIA binaries).

## Tool reference

### get_transcript
```
get_transcript(video_id: str, language: str = "en", format: str = "clean") -> str
```
`format`: `clean` (default), `text`, `json`, `srt`, `vtt`. Accepts a video ID or URL.

### get_transcripts_batch
```
get_transcripts_batch(video_ids: list[str], language="en", format="clean", max_workers=4) -> list[dict]
```
Each result: `video_id`, `success`, and either `transcript`/`source`/`language` or `error`.

### search_videos
```
search_videos(query: str, max_results=5, language="en", with_transcripts=False, format="clean") -> list[dict]
```
Returns `video_id`, `title`, `channel_name`, `duration`, `view_count`, and
(when `with_transcripts=True`) `transcript`.

## How it works

- **Captions first** via the Innertube API (ANDROID_VR → WEB → MWEB → watch-page
  fallback). Fast, a few KB per video, no API key, no audio download.
- **Whisper fallback** only when a video has no captions (downloads audio,
  transcribes locally; requires the `whisper` extra + `ffmpeg`).
- **SQLite cache** avoids redundant fetches.

## Troubleshooting

- **`yttranscript-mcp: command not found`** — ensure the venv with the package is
  active, or use the `python -m ytt.mcp.server` form with `cwd` set.
- **`ffmpeg not found`** — only needed for the Whisper fallback; install ffmpeg.
- **Persistent throttling** — set `YTT_PROXY` and/or `YTT_COOKIES_FILE`.

## Example prompts

- "Get the transcript for https://www.youtube.com/watch?v=VIDEO_ID"
- "Search YouTube for 'machine learning tutorial' and summarize the top 3."
- "Compare the transcripts of VIDEO_ID_1 and VIDEO_ID_2."
