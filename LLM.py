# Instructions for LLM Agents: Adding YouTube Transcript Tools to Yourself

This document provides guidance for LLM agents (such as OpenCLAW, Claude Code, or other AI assistants) to add YouTube transcript and search capabilities to their toolset via the MCP (Model Context Protocol) protocol.

**Repository:** https://github.com/andrewctf/ytt

## Overview

The `ytt` (YouTube Transcripts) project exposes the following tools via MCP:

1. `get_transcript` - Fetch transcript for a single YouTube video
2. `get_transcripts_batch` - Fetch transcripts for multiple videos concurrently
3. `search_videos` - Search YouTube for videos matching a query, optionally fetching transcripts

## For OpenCLAW Agents

OpenCLAW supports MCP tools. To add this tool:

### Method 1: Via Configuration File

Add to your OpenCLAW configuration:

```json
{
  "mcpServers": {
    "ytt": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/ytt"
    }
  }
}
```

### Method 2: Via openclaw tool command

If OpenCLAW provides a tool-add command:

```
/tool-add ytt python -m mcp_server.server --cwd /path/to/ytt
```

## For Claude Code Agents

Claude Code supports MCP servers. Add to your settings:

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/ytt"
    }
  }
}
```

Or use the `/mcp` command to add servers dynamically.

## For Cursor IDE

Add to `%APPDATA%\Cursor\mcp.json` (Windows) or `~/.cursor/mcp.json` (Mac/Linux):

```json
{
  "mcpServers": {
    "ytt": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "env": {
        "PYTHONPATH": "/path/to/ytt"
      }
    }
  }
}
```

## For VS Code (Continue Extension)

Add to your `~/.continue/config.py`:

```python
from continue.sdk.core.main import ContinueSDK

# In your config:
{
    "mcp_servers": {
        "ytt": {
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": "/path/to/ytt"
        }
    }
}
```

## For Generic MCP-Compatible Agents

### Step 1: Clone the Repository

```bash
git clone https://github.com/andrewctf/ytt.git
cd ytt
```

### Step 2: Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-mcp.txt
```

### Step 3: Add MCP Server Configuration

Add to your agent's MCP configuration:

```json
{
  "mcpServers": {
    "ytt": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/ytt",
      "env": {
        "YTT_AUTO_DOWNLOAD_CUDA": "1"
      }
    }
  }
}
```

Setting `YTT_AUTO_DOWNLOAD_CUDA=1` enables automatic download of NVIDIA CUDA
libraries when GPU acceleration is requested. This avoids bundling potentially
copyrighted NVIDIA binaries directly in the repository.

### Step 4: Verify

Test by asking your agent:

> "Search YouTube for 'Python tutorial' and get transcripts for the top 3 results"

## Available Tools

### get_transcript

```
get_transcript(video_id: str, language: str = "en", format: str = "text") -> str
```

**Parameters:**
- `video_id` - YouTube video ID or full URL
- `language` - Language code (e.g., "en", "es", "fr")
- `format` - Output format: "text", "json", "srt", or "vtt"

**Returns:** Formatted transcript string

### get_transcripts_batch

```
get_transcripts_batch(video_ids: list[str], language: str = "en", format: str = "text", max_workers: int = 4) -> list[dict]
```

**Parameters:**
- `video_ids` - List of YouTube video IDs or URLs
- `language` - Language code
- `format` - Output format
- `max_workers` - Max concurrent transcription tasks

**Returns:** List of results with `video_id`, `success`, `transcript`, `source`, `error`

### search_videos

```
search_videos(query: str, max_results: int = 5, language: str = "en", with_transcripts: bool = False) -> list[dict]
```

**Parameters:**
- `query` - Search query string
- `max_results` - Max number of results (default 5, max 20)
- `language` - Language for transcripts
- `with_transcripts` - If True, fetch transcripts for each result

**Returns:** List of video results with `video_id`, `title`, `channel_name`, `duration`, `view_count`, and optionally `transcript`

### setup_gpu

```
setup_gpu() -> dict
```

Downloads and sets up NVIDIA CUDA libraries for GPU acceleration.
Only downloads if `YTT_AUTO_DOWNLOAD_CUDA=1` is set in environment.

**Returns:** Dict with `success` and `message` keys

### download_cuda

```
download_cuda() -> dict
```

Manually triggers download of NVIDIA CUDA runtime packages.
Downloads nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12.

**Returns:** Dict with `success`, `message`, and `packages` keys

## Implementation Notes

- **No API keys required** - Uses YouTube's internal Innertube API and Whisper AI
- **Rate limiting** - Innertube API calls are rate-limited; use caching to avoid repeated calls
- **Whisper fallback** - If Innertube captions unavailable, automatically falls back to Whisper transcription
- **Local processing** - Whisper runs locally (CPU or GPU), no external API calls for transcription

## Troubleshooting

### "Module not found" errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install -r requirements-mcp.txt
```

### "ffmpeg not found"
Install ffmpeg:
- Windows: `winget install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### "cublas not found" or GPU errors
CUDA libraries are not bundled to avoid copyright issues. Set `YTT_AUTO_DOWNLOAD_CUDA=1`
to enable automatic download, or call the `download_cuda()` tool:

```python
# In your agent, call:
download_cuda()  # Downloads nvidia-cublas-cu12, etc.
```

Or set environment variable before starting the MCP server:
```bash
YTT_AUTO_DOWNLOAD_CUDA=1 python -m mcp_server.server
```

### Connection issues
The MCP server must be running. Start it with:
```bash
python -m mcp_server.server
```

## Example Usage in Conversations

When this tool is properly configured, you can ask:

- "Get the transcript for this video: https://www.youtube.com/watch?v=VIDEO_ID"
- "Search for 'machine learning tutorial' and get the top 5 results with transcripts"
- "Find YouTube videos about Python programming and summarize what each video is about"
- "Compare the transcripts of these two videos: VIDEO_ID_1 and VIDEO_ID_2"
