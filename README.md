# YouTube Transcript Fetcher

Fetch transcripts from any YouTube video using Whisper AI transcription. Search YouTube and get transcripts for the top results. No YouTube API key required.

## Features

- **Whisper-powered** — State-of-the-art AI transcription, 99%+ accuracy
- **YouTube Search** — Search YouTube and get transcripts for top results
- **No API key needed** — Works without YouTube Data API credentials
- **Multiple formats** — Text, JSON, SRT, VTT output
- **Caching** — SQLite-backed cache avoids re-transcribing
- **Rate-limit free** — Whisper runs locally, no external API limits
- **CLI & library** — Use as a command-line tool or Python module
- **MCP server** — Integrate with AI tools via Model Context Protocol

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewctf/ytt.git
cd ytt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** For detailed GPU/CUDA setup, see [QUICKSTART.md](QUICKSTART.md#4-install-pytorch-with-cuda-support-optional).

### Additional Setup for Whisper

Whisper requires `ffmpeg` for audio extraction:

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

## Quick Start

For detailed installation and setup instructions, see [QUICKSTART.md](QUICKSTART.md).

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
python cli.py transcript VIDEO_ID1 VIDEO_ID2 VIDEO_ID3

# Search YouTube for videos and get transcripts
python cli.py search "Python tutorial" --limit 5 --with-transcripts

# Search only (no transcripts)
python cli.py search "Python tutorial" --limit 10

# JSON output for search
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
results = await search_and_get_transcripts("Python tutorial", max_results=3, language="en")
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

### MCP Server

> **Note:** See [QUICKSTART.md](QUICKSTART.md#mcp-server-setup) for detailed configuration with Claude Desktop, Cursor, and VS Code.

Start the MCP server:

```bash
python -m mcp_server.server
```

The server exposes three tools:
- `get_transcript` - Get transcript for a single video
- `get_transcripts_batch` - Get transcripts for multiple videos concurrently
- `search_videos` - Search YouTube for videos matching a query

Or integrate with Claude Desktop by adding to your MCP settings:

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
- Transcribes using `faster-whisper` (CPU-optimized)
- Returns word-level timestamps and segment text
- Works on any video with audio
- ~1-3x real-time processing speed

### Innertube API (Fallback)

- Scrapes YouTube's internal API
- No API key required
- Fast (~0.5-2s per video)
- ~85% coverage (some videos lack captions)
- Rate limited (~5 req/10s per IP)

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

## Configuration

Edit `config.py` to customize behavior:

```python
class Config:
    # Whisper settings
    WHISPER_MODEL = "base"      # tiny/base/small/medium/large
    WHISPER_FALLBACK_ENABLED = True

    # Cache settings
    CACHE_TTL_DAYS = 7
    CACHE_DB_PATH = ".transcript_cache.db"

    # Rate limiting (for Innertube fallback)
    RATE_LIMIT_RATE = 0.5       # tokens per second
    RATE_LIMIT_BURST = 5       # max bucket size

    # Batch processing
    MAX_BATCH_SIZE = 50
```

### Whisper Models

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| tiny | 10x | ~75% | ~1GB |
| base | 7x | ~85% | ~1GB |
| small | 4x | ~90% | ~2GB |
| medium | 2x | ~95% | ~5GB |
| large | 1x | ~97% | ~6GB |

The `base` model is recommended for most use cases — fast and accurate enough.

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
│   └── exceptions.py        # Custom exceptions
├── mcp_server/
│   ├── __init__.py
│   └── server.py           # FastMCP server
├── cli.py                   # CLI entrypoint
├── main.py                  # Library entrypoint
├── config.py                # Configuration
├── requirements.txt         # Core dependencies
├── requirements-mcp.txt     # MCP dependencies
├── README.md
└── QUICKSTART.md
```

## Troubleshooting

### "No module named 'rich'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Whisper fails with "ffmpeg not found"

Install ffmpeg (see Installation section above).

### Slow transcription speed

- Use a smaller Whisper model (`base` instead of `large`)
- Use GPU acceleration by changing `device="cpu"` to `device="cuda"` in `whisper_runner.py`
- Enable cache to avoid re-transcribing

### Rate limiting from Innertube

The Innertube fallback is rate-limited by YouTube (~5 req/10s). Use Whisper as primary (default) to avoid this. The cache also prevents redundant requests.

### Cache not working

Check cache stats:
```bash
python cli.py cache-stats
```

Clean expired entries:
```bash
python cli.py cache-stats --clean
```

## Development

### Run tests
```bash
pytest
```

### Format code
```bash
black src/
ruff check src/
```

## License

MIT License
