# yttranscript-mcp

> Fetch, clean, search, and (locally) summarize YouTube transcripts. Captions-first and rate-limit resistant, with an optional local-LLM summarizer that compresses transcripts to save tokens. Exposes a Model Context Protocol (MCP) server. No YouTube API key required.

This file follows the [llms.txt](https://llmstxt.org) convention so an AI agent can install and wire up the tool autonomously.

## Install (pick one)

```bash
pip install "yttranscript-mcp[mcp]"           # MCP server + captions (recommended for agents)
pip install "yttranscript-mcp[mcp,whisper]"   # + local Whisper fallback (needs ffmpeg)
pip install yttranscript-mcp                   # library + CLI only
uvx --from "yttranscript-mcp[mcp]" yttranscript-mcp   # run without installing
```

Python >= 3.10. PyPI: https://pypi.org/project/yttranscript-mcp/ · Repo: https://github.com/AndrewCTF/YTT

## Register the MCP server

After install, the `yttranscript-mcp` command launches the server over stdio.

Claude Code:
```bash
claude mcp add yt-transcript -- yttranscript-mcp
```

Claude Desktop / Cursor (`claude_desktop_config.json` or `~/.cursor/mcp.json`):
```json
{ "mcpServers": { "yt-transcript": { "command": "yttranscript-mcp" } } }
```

From a source checkout instead of an install:
```json
{ "mcpServers": { "yt-transcript": {
  "command": "python", "args": ["-m", "ytt.mcp.server"], "cwd": "/abs/path/to/YTT"
} } }
```

## MCP tools

- `get_transcript(video_id, language="en", format="clean")` -> str
  - `format`: `clean` (default, deduplicated, no timestamps — best for LLMs), `text`, `json`, `srt`, `vtt`, `summary` (local-LLM summary).
- `get_transcripts_batch(video_ids, language="en", format="clean", max_workers=4)` -> list
- `search_videos(query, max_results=5, language="en", with_transcripts=False, format="clean")` -> list
- `summarize_video(video_id, language="en", model=None)` -> {video_id, source, language, summary}
  - Summarizes with a LOCAL model (Ollama by default) so you ingest a short summary instead of the full transcript. Nothing leaves the machine.
- `setup_gpu()`, `download_cuda()` -> CUDA setup for the Whisper fallback (opt-in).

`video_id` accepts a raw ID or any YouTube URL.

## Local summarization (token-saving)

Requires a local LLM. With [Ollama](https://ollama.com):
```bash
ollama serve
ollama pull qwen3.6:27b            # default; smaller: qwen3:8b, qwen3:4b, qwen3.5:2b
```
Then call `summarize_video(...)` or `get_transcript(..., format="summary")`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `YTT_PROXY` | – | `http://user:pass@host:port` — beat IP-level rate limits |
| `YTT_COOKIES_FILE` | – | Netscape cookies.txt for age/region-gated videos |
| `YTT_SUMMARY_PROVIDER` | `ollama` | `ollama` or `openai` (OpenAI-compatible endpoint) |
| `YTT_SUMMARY_MODEL` | `qwen3.6:27b` | Local summary model |
| `YTT_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `YTT_SUMMARY_OPENAI_BASE` | `http://localhost:8080/v1` | OpenAI-compatible base URL |
| `YTT_SUMMARY_KEEP_ALIVE` | `5m` | Keep model hot in memory (`-1` = forever) |
| `YTT_SUMMARY_AUTO_PULL` | `false` | Pull the model on demand if missing |

## CLI (for humans)

```bash
ytt transcript VIDEO_ID                 # clean text
ytt transcript VIDEO_ID --summarize     # local-LLM summary
ytt search "query" -n 5 --with-transcripts
ytt batch VID1 VID2 VID3
```

## Notes

- Captions path needs no API key and no audio download; Whisper is a fallback only for caption-less videos.
- All summarization is local and opt-in; no transcript content is sent to any cloud service.
