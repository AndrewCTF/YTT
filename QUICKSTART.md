# Quick Start

## Install

```bash
pip install yttranscript-mcp            # captions + CLI + library
pip install "yttranscript-mcp[mcp]"     # + MCP server
pip install "yttranscript-mcp[whisper]" # + local Whisper fallback (needs ffmpeg)
```

The captions path needs **no** extra system packages. Only the Whisper fallback
needs `ffmpeg`:

- **Windows:** `winget install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## CLI

```bash
ytt transcript VIDEO_ID                 # clean, LLM-ready text (default)
ytt transcript "https://youtu.be/ID"    # URLs work
ytt transcript VIDEO_ID -f json|text|srt|vtt
ytt transcript VIDEO_ID -o out.txt
ytt transcript VIDEO_ID --no-whisper    # captions only

ytt batch VID1 VID2 VID3
ytt search "python tutorial" -n 5 [--with-transcripts]
ytt cache-stats [--clean]
```

## Library

```python
import asyncio
from ytt import get_transcript, search

async def main():
    r = await get_transcript("dQw4w9WgXcQ", output_format="clean")
    print(r.source, r.language)
    print(r.content)

asyncio.run(main())
```

`output_format`: `clean` (default), `text`, `json`, `srt`, `vtt`, `summary`.

## Summarize locally (save tokens)

Needs a local LLM (default [Ollama](https://ollama.com)):

```bash
ollama serve && ollama pull qwen3.6:27b   # smaller: qwen3:8b, qwen3:4b, qwen3.5:2b
ytt transcript VIDEO_ID --summarize
```

`YTT_SUMMARY_MODEL` overrides the model; `YTT_SUMMARY_KEEP_ALIVE` keeps it hot;
`YTT_SUMMARY_AUTO_PULL=1` pulls on demand. Provider `openai` targets any
OpenAI-compatible server via `YTT_SUMMARY_OPENAI_BASE`.

## MCP server

```bash
yttranscript-mcp        # or: python -m ytt.mcp.server
```

Claude Desktop (`claude_desktop_config.json`) / Cursor (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "yt-transcript": { "command": "yttranscript-mcp" }
  }
}
```

Tools: `get_transcript`, `get_transcripts_batch`, `search_videos`
(all default to the `clean` format), plus `setup_gpu` / `download_cuda`.

## Beating rate limits

If you hit throttling from a datacenter/cloud IP, set:

```bash
export YTT_PROXY="http://user:pass@host:port"     # route around IP blocks
export YTT_COOKIES_FILE="/path/to/cookies.txt"     # age/region-gated videos
export YTT_MAX_RETRIES=6                            # more retries on 429/5xx
```

The cache (`.transcript_cache.db`, 7-day TTL) also prevents repeat requests.
See the README for the full `YTT_*` variable list.

## GPU (Whisper fallback only)

```bash
pip install "yttranscript-mcp[whisper,torch]"
export YTT_WHISPER_GPU=1
# or let it fetch CUDA libs on demand:
export YTT_AUTO_DOWNLOAD_CUDA=1
```

## How it works

```
video id/url → cache → captions (ANDROID_VR → WEB → MWEB → watch page)
                              ↓ none?
                      Whisper fallback (download audio, transcribe locally)
                              ↓
                      cache → format (clean / text / json / srt / vtt)
```

Captions are fast (a few KB) and rate-limit resistant; Whisper is the fallback
for videos that have no captions at all.
