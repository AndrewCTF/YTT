"""FastMCP server exposing transcript tools."""

import os
from fastmcp import FastMCP

from src.service import get_transcript, get_transcripts_batch
from src.search_service import search, search_and_get_transcripts
from src.cuda_dll_manager import setup_gpu_if_needed, download_cuda_dlls
from config import config


# Create MCP server instance
mcp = FastMCP("yt-transcript")


def _is_auto_download_enabled() -> bool:
    """Check if auto-download is enabled via env var or config."""
    if os.environ.get("YTT_AUTO_DOWNLOAD_CUDA", "").lower() in ("1", "true", "yes"):
        return True
    return config.AUTO_DOWNLOAD_CUDA


def _ensure_gpu_ready(verbose: bool = True) -> dict:
    """Ensure GPU is ready, downloading CUDA if needed.

    Returns:
        Dict with 'success' and 'message' keys.
    """
    if not _is_auto_download_enabled():
        return {
            "success": True,
            "message": "Auto-download disabled. Set YTT_AUTO_DOWNLOAD_CUDA=1 or AUTO_DOWNLOAD_CUDA=True to enable."
        }

    return {"success": True, "message": "GPU ready"}


# Initialize GPU setup on server start if configured
_gpu_setup_done = False
def _init_gpu():
    """Initialize GPU environment on server start."""
    global _gpu_setup_done
    if not _gpu_setup_done:
        if _is_auto_download_enabled():
            success, msg = setup_gpu_if_needed(verbose=True)
            print(f"GPU setup: {msg}")
        _gpu_setup_done = True


_init_gpu()


@mcp.tool()
async def setup_gpu() -> dict:
    """Set up GPU environment by downloading CUDA/cuBLAS libraries.

    This tool downloads NVIDIA CUDA runtime libraries if they're not already
    installed. Use this before running transcription tasks that need GPU.

    The download only happens if:
    - YTT_AUTO_DOWNLOAD_CUDA environment variable is set to "1", "true", or "yes", OR
    - AUTO_DOWNLOAD_CUDA is set to True in config.py
    - No existing CUDA installation is found

    Returns:
        Dict with 'success' and 'message' keys.
    """
    if not _is_auto_download_enabled():
        return {
            "success": False,
            "message": "Auto-download disabled. Set YTT_AUTO_DOWNLOAD_CUDA=1 or AUTO_DOWNLOAD_CUDA=True in config.",
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
    """Manually download and install CUDA runtime libraries.

    Downloads NVIDIA CUDA/cuBLAS/cuDNN packages via pip.
    These packages are officially distributed by NVIDIA.

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
    format: str = "text",
) -> str:
    """Get transcript for a YouTube video.

    Args:
        video_id: YouTube video ID or URL.
        language: Language code (e.g., 'en', 'es').
        format: Output format - 'text', 'json', 'srt', 'vtt'.

    Returns:
        Formatted transcript string.
    """
    result = await get_transcript(
        video_id,
        language=language,
        output_format=format,
    )
    return result.content


@mcp.tool()
async def get_transcripts_batch(
    video_ids: list[str],
    language: str = "en",
    format: str = "text",
    max_workers: int = 4,
) -> list[dict]:
    """Get transcripts for multiple YouTube videos concurrently.

    Args:
        video_ids: List of YouTube video IDs or URLs.
        language: Language code (e.g., 'en', 'es').
        format: Output format - 'text', 'json', 'srt', 'vtt'.
        max_workers: Max concurrent transcriptions.

    Returns:
        List of results, each with video_id, success, transcript, source, error.
    """
    results = await get_transcripts_batch(
        video_ids,
        language=language,
        output_format=format,
        max_workers=max_workers,
    )

    output = []
    for vid, result in zip(video_ids, results):
        if isinstance(result, Exception):
            output.append({
                "video_id": vid,
                "success": False,
                "error": str(result),
            })
        else:
            output.append({
                "video_id": vid,
                "success": True,
                "transcript": result.content,
                "source": result.source,
                "language": result.language,
            })

    return output


@mcp.tool()
async def search_videos(
    query: str,
    max_results: int = 5,
    language: str = "en",
    with_transcripts: bool = False,
) -> list[dict]:
    """Search YouTube for videos matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5, max 20).
        language: Language code for transcripts (e.g., 'en', 'es').
        with_transcripts: If True, also fetch transcripts for each video.

    Returns:
        List of video results with video_id, title, channel_name, duration, view_count.
        If with_transcripts is True, each result also includes transcript content.
    """
    if with_transcripts:
        results = await search_and_get_transcripts(
            query, max_results=max_results, language=language
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
    else:
        results = await search(query, max_results=max_results)
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


if __name__ == "__main__":
    mcp.run()