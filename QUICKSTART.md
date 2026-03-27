# Quick Start

## Prerequisites

- Python 3.10+
- ffmpeg (for Whisper audio extraction)
- (Optional) CUDA-capable GPU for faster transcription

### Install ffmpeg

**Windows (with winget):**
```powershell
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/andrewctf/ytt.git
cd ytt
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

### 3. Install base dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU Acceleration (optional)

For GPU acceleration with faster transcription speeds:

**Option A: Auto-download CUDA (recommended)**
Set environment variable or config to auto-download CUDA libraries:
```bash
YTT_AUTO_DOWNLOAD_CUDA=1 python -m mcp_server.server
```

Or in `config.py`:
```python
WHISPER_USE_GPU = True
AUTO_DOWNLOAD_CUDA = True  # Automatically download CUDA DLLs when needed
```

**Option B: Manual CUDA installation**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** NVIDIA CUDA/cuBLAS libraries are not bundled in the repository to avoid copyright issues. When `AUTO_DOWNLOAD_CUDA=True` or `YTT_AUTO_DOWNLOAD_CUDA=1` is set, the MCP server will automatically download the official NVIDIA packages (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12) via pip.

### 5. Install MCP server dependencies (optional)

```bash
pip install -r requirements-mcp.txt
```

---

## Configuration

Edit `config.py` to customize behavior:

```python
class Config:
    # Whisper settings
    WHISPER_MODEL = "base"        # tiny/base/small/medium/large
    WHISPER_USE_GPU = True         # Enable GPU acceleration
    WHISPER_BATCH_SIZE = 16       # Batch size for GPU inference
    WHISPER_FALLBACK_ENABLED = True

    # Cache settings
    CACHE_TTL_DAYS = 7
    CACHE_DB_PATH = ".transcript_cache.db"

    # Rate limiting (for Innertube fallback)
    RATE_LIMIT_RATE = 0.5         # tokens per second
    RATE_LIMIT_BURST = 5         # max bucket size

    # Batch processing
    MAX_CONCURRENT_WORKERS = 4    # Max parallel transcription tasks

    # Output defaults
    DEFAULT_LANGUAGE = "en"
    DEFAULT_FORMAT = "text"       # text/json/srt/vtt

    # CUDA settings
    AUTO_DOWNLOAD_CUDA = False    # Auto-download CUDA libraries when GPU needed
```

### Whisper Model Comparison

| Model  | Speed | Accuracy | Memory | Best For                    |
|--------|-------|----------|--------|-----------------------------|
| tiny   | 10x   | ~75%     | ~1 GB  | Fast testing, low resource |
| base   | 7x    | ~85%     | ~1 GB  | **Recommended** (default)   |
| small  | 4x    | ~90%     | ~2 GB  | Higher accuracy             |
| medium | 2x    | ~95%     | ~5 GB  | Best accuracy              |
| large  | 1x    | ~97%     | ~6 GB  | Maximum accuracy           |

---

## Usage

### CLI

```bash
# Get transcript (Whisper is used by default)
python cli.py transcript VIDEO_ID

# Or with a full YouTube URL
python cli.py transcript "https://www.youtube.com/watch?v=a1JTPFfshI0"

# Different output formats
python cli.py transcript VIDEO_ID --format json
python cli.py transcript VIDEO_ID --format srt
python cli.py transcript VIDEO_ID --format vtt

# Save to file
python cli.py transcript VIDEO_ID --output transcript.txt

# Batch processing
python cli.py batch VIDEO_ID1 VIDEO_ID2 VIDEO_ID3

# Search YouTube for videos
python cli.py search "Python tutorial" --limit 5

# Search with transcripts
python cli.py search "Python tutorial" --with-transcripts --limit 3

# JSON output
python cli.py search "Python tutorial" --format json

# Cache management
python cli.py cache-stats
python cli.py cache-stats --clean  # Remove expired entries
```

### Python Library

```python
from src.service import get_transcript
from src.search_service import search, search_and_get_transcripts

# Basic usage
result = await get_transcript("VIDEO_ID")
print(result.content)

# With options
result = await get_transcript(
    "VIDEO_ID",
    language="en",
    output_format="json",
    use_cache=True,
)

# Access metadata
print(f"Source: {result.source}")      # 'whisper' or 'innertube'
print(f"Language: {result.language}")   # Detected language
print(f"Video ID: {result.video_id}")

# Search YouTube for videos
results = await search("Python tutorial", max_results=5)
for video in results:
    print(f"{video.title} ({video.video_id}) - {video.channel_name}")

# Search and get transcripts for results
results = await search_and_get_transcripts("Python tutorial", max_results=3)
for video, transcript in results:
    if transcript:
        print(f"{video.title}: {transcript.content[:100]}...")
```

For synchronous usage:

```python
import asyncio
from src.service import get_transcript
from src.search_service import search

def fetch_transcript(video_id):
    return asyncio.run(get_transcript(video_id))

def search_videos(query, max_results=5):
    return asyncio.run(search(query, max_results=max_results))

result = fetch_transcript("VIDEO_ID")
print(result.content)

videos = search_videos("Python tutorial")
```

---

## MCP Server Setup

The MCP server lets you integrate with AI tools like Claude Desktop, Cursor, and other MCP-compatible editors.

### Option 1: Run Manually

```bash
python -m mcp_server.server
```

The server exposes three tools:
- `get_transcript` - Get transcript for a single video
- `get_transcripts_batch` - Get transcripts for multiple videos concurrently
- `search_videos` - Search YouTube for videos matching a query

### Option 2: Claude Desktop Integration

#### Step 1: Install Claude Desktop

Download from https://claude.ai/desktop

#### Step 2: Add MCP Server

Open Claude Desktop settings and add a new MCP server:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/ytt"
    }
  }
}
```

#### Step 3: Restart Claude Desktop

After editing the config, restart Claude Desktop to load the new MCP server.

#### Step 4: Verify Installation

In Claude Desktop, you should see the yt-transcript server connected. You can now use:

```
Get the transcript from this YouTube video: https://www.youtube.com/watch?v=VIDEO_ID
```

### Option 3: Cursor IDE Integration

Add to Cursor settings (`~/.cursor/mcp.json` on macOS, `%APPDATA%\Cursor\mcp.json` on Windows):

```json
{
  "mcpServers": {
    "yt-transcript": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/ytt"
      }
    }
  }
}
```

### Option 4: VS Code with Continue Extension

Add to your `~/.continue/config.py`:

```python
{
    "mcp_servers": {
        "yt-transcript": {
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": "/absolute/path/to/ytt"
        }
    }
}
```

### Troubleshooting MCP Connection

1. **Server not starting**: Verify Python path is correct in config
2. **Module not found**: Run `pip install -r requirements-mcp.txt`
3. **Timeout errors**: Reduce `MAX_CONCURRENT_WORKERS` in config.py
4. **Restart Claude Desktop** after config changes

---

## How It Works

```
Video ID → Cache Check
              ↓ found?
         Return Cached
              ↓ not found
         Whisper (primary)
         - Download audio via yt-dlp
         - Transcribe with faster-whisper
         - Returns word-level timestamps
              ↓ fails?
         Innertube API (fallback)
         - Extract API key from video page
         - Fetch caption tracks
         - Parse JSON3 timed text
              ↓
         Cache Result
              ↓
         Format & Return
```

### Whisper (Primary)
- Downloads audio using `yt-dlp`
- Transcribes using `faster-whisper` (CPU or GPU)
- Returns word-level timestamps and segment text
- Works on **any video with audio**
- ~1-3x real-time processing speed on CPU

### Innertube API (Fallback)
- Scrapes YouTube's internal API
- No API key required
- Fast (~0.5-2s per video)
- ~85% coverage (some videos lack captions)
- Rate limited (~5 req/10s per IP)

---

## Output Formats

### Text (default)
```
Good morning, here we are, a live suturing course like nobody else has ever
done and what are we covering, we're covering every suturing technique...
```

### JSON
```json
{
  "video_id": "a1JTPFfshI0",
  "language": "en",
  "source": "whisper",
  "segments": [
    {"start": 0.0, "end": 4.5, "text": "Good morning, here we are..."},
    {"start": 4.5, "end": 9.2, "text": "a live suturing course..."}
  ]
}
```

### SRT (SubRip)
```
1
00:00:00,000 --> 00:00:04,500
Good morning, here we are, a live suturing course...

2
00:00:04,500 --> 00:00:09,200
a live suturing course like nobody else...
```

### VTT (WebVTT)
```
WEBVTT

00:00:00.000 --> 00:00:04.500
Good morning, here we are, a live suturing course...

00:00:04.500 --> 00:00:09.200
a live suturing course like nobody else...
```

---

## Troubleshooting

### "No module named 'rich'" or import errors
```bash
pip install -r requirements.txt
```

### Whisper fails with "ffmpeg not found"
Install ffmpeg (see Prerequisites section above).

### Slow transcription speed
- Use a smaller Whisper model (`base` instead of `large`)
- Enable GPU acceleration in `config.py`: `WHISPER_USE_GPU = True`
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Enable cache to avoid re-transcribing

### Rate limiting from Innertube
The Innertube fallback is rate-limited by YouTube (~5 req/10s). Use Whisper as primary (default) to avoid this. The cache also prevents redundant requests.

### Cache not working
```bash
python cli.py cache-stats
python cli.py cache-stats --clean
```

### GPU/CUDA errors
If you see cublas or CUDA errors:
1. Set `WHISPER_USE_GPU = False` in config.py to use CPU
2. Or enable auto-download: Set `AUTO_DOWNLOAD_CUDA = True` in config.py (the MCP server will automatically download CUDA libraries)
3. Or manually install CUDA packages:
   ```bash
   pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
   ```

---

## File Structure

```
ytt/
├── src/
│   ├── __init__.py
│   ├── fetcher.py          # Innertube API client
│   ├── whisper_runner.py    # Whisper transcription
│   ├── parser.py            # Caption parsing utilities
│   ├── formatters.py        # Output formatters
│   ├── cache.py             # SQLite cache
│   ├── rate_limiter.py      # Token bucket
│   ├── service.py           # Orchestrator
│   ├── searcher.py          # YouTube search
│   ├── search_cache.py      # Search result cache
│   ├── search_service.py    # Search orchestrator
│   ├── cuda_dll_manager.py  # Auto-download CUDA libraries
│   └── exceptions.py         # Custom exceptions
├── mcp_server/
│   ├── __init__.py
│   └── server.py            # FastMCP server
├── cli.py                    # CLI entrypoint
├── main.py                   # Library entrypoint
├── config.py                 # Configuration
├── requirements.txt          # Core dependencies
├── requirements-mcp.txt      # MCP dependencies
└── QUICKSTART.md            # This file
```
