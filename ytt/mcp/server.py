"""FastMCP server exposing transcript tools.

Run with ``yttranscript-mcp`` (console script) or ``python -m ytt.mcp.server``.
"""

import os

from fastmcp import FastMCP

# Import the library functions under private aliases. The MCP tool functions
# below intentionally reuse these public names, so without the alias the tool
# body would call itself and recurse infinitely.
from ..config import config
from ..cuda_dll_manager import download_cuda_dlls, setup_gpu_if_needed
from ..search_service import search as _search
from ..search_service import search_and_get_transcripts as _search_and_get_transcripts
from ..service import get_transcript as _get_transcript
from ..service import get_transcripts_batch as _get_transcripts_batch

mcp = FastMCP("yt-transcript")


def _is_auto_download_enabled() -> bool:
    """Check if CUDA auto-download is enabled via env var or config."""
    if os.environ.get("YTT_AUTO_DOWNLOAD_CUDA", "").lower() in ("1", "true", "yes"):
        return True
    return config.AUTO_DOWNLOAD_CUDA


_gpu_setup_done = False


def _init_gpu() -> None:
    """Initialize GPU environment on server start if configured."""
    global _gpu_setup_done
    if not _gpu_setup_done:
        if _is_auto_download_enabled():
            _, msg = setup_gpu_if_needed(verbose=True)
            print(f"GPU setup: {msg}")
        _gpu_setup_done = True


@mcp.tool()
async def setup_gpu() -> dict:
    """Set up GPU by downloading CUDA/cuBLAS libraries (opt-in).

    Only acts if YTT_AUTO_DOWNLOAD_CUDA=1 or AUTO_DOWNLOAD_CUDA=True.

    Returns:
        Dict with 'success' and 'message' keys.
    """
    if not _is_auto_download_enabled():
        return {
            "success": False,
            "message": "Auto-download disabled. Set YTT_AUTO_DOWNLOAD_CUDA=1 to enable.",
            "action": "none",
        }
    success, message = setup_gpu_if_needed(verbose=True)
    return {
        "success": success,
        "message": message,
        "action": "download_cuda" if success else "cpu_mode",
    }


@mcp.tool()
async def download_cuda() -> dict:
    """Manually download and install CUDA runtime libraries via pip.

    Returns:
        Dict with 'success', 'message', and 'packages' keys.
    """
    success = download_cuda_dlls(verbose=True)
    return {
        "success": success,
        "message": "CUDA libraries installed successfully" if success else "Installation failed",
        "packages": ["nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12"],
    }


@mcp.tool()
async def get_transcript(
    video_id: str,
    language: str = "en",
    format: str = "clean",
) -> str:
    """Get the transcript for a YouTube video.

    Args:
        video_id: YouTube video ID or URL.
        language: Language code (e.g., 'en', 'es').
        format: 'clean' (default, deduplicated and LLM-friendly), 'text',
            'json', 'srt', or 'vtt'.

    Returns:
        Formatted transcript string.
    """
    result = await _get_transcript(video_id, language=language, output_format=format)
    return result.content


@mcp.tool()
async def get_transcripts_batch(
    video_ids: list[str],
    language: str = "en",
    format: str = "clean",
    max_workers: int = 4,
) -> list[dict]:
    """Get transcripts for multiple YouTube videos concurrently.

    Args:
        video_ids: List of YouTube video IDs or URLs.
        language: Language code (e.g., 'en', 'es').
        format: 'clean' (default), 'text', 'json', 'srt', or 'vtt'.
        max_workers: Max concurrent transcriptions.

    Returns:
        List of results, each with video_id, success, transcript, source, error.
    """
    results = await _get_transcripts_batch(
        video_ids, language=language, output_format=format, max_workers=max_workers
    )

    output = []
    for vid, result in zip(video_ids, results):
        if isinstance(result, Exception):
            output.append({"video_id": vid, "success": False, "error": str(result)})
        else:
            output.append(
                {
                    "video_id": vid,
                    "success": True,
                    "transcript": result.content,
                    "source": result.source,
                    "language": result.language,
                }
            )
    return output


@mcp.tool()
async def search_videos(
    query: str,
    max_results: int = 5,
    language: str = "en",
    with_transcripts: bool = False,
    format: str = "clean",
) -> list[dict]:
    """Search YouTube for videos matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5, max 20).
        language: Language code for transcripts (e.g., 'en', 'es').
        with_transcripts: If True, also fetch transcripts for each video.
        format: Transcript format when with_transcripts is True ('clean' default).

    Returns:
        List of video results with video_id, title, channel_name, duration,
        view_count, and (optionally) transcript content.
    """
    if with_transcripts:
        results = await _search_and_get_transcripts(
            query, max_results=max_results, language=language, output_format=format
        )
        output = []
        for video, transcript in results:
            item = {
                "video_id": video.video_id,
                "title": video.title,
                "channel_name": video.channel_name,
                "duration": video.duration,
                "view_count": video.view_count,
            }
            if transcript:
                item["transcript"] = transcript.content
                item["transcript_source"] = transcript.source
            output.append(item)
        return output

    results = await _search(query, max_results=max_results)
    return [
        {
            "video_id": r.video_id,
            "title": r.title,
            "channel_name": r.channel_name,
            "duration": r.duration,
            "view_count": r.view_count,
        }
        for r in results
    ]


def main() -> None:
    """Console-script entry point."""
    _init_gpu()
    mcp.run()


if __name__ == "__main__":
    main()
