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
from ..search_service import search_ranked as _search_ranked
from ..service import ask_video as _ask_video
from ..service import corpus_stats as _corpus_stats
from ..service import find_in_corpus as _find_in_corpus
from ..service import get_transcript as _get_transcript
from ..service import get_transcripts_batch as _get_transcripts_batch
from ..service import get_video_info as _get_video_info
from ..service import index_videos as _index_videos
from ..service import list_languages as _list_languages
from ..service import search_in_video as _search_in_video

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
            'json', 'srt', 'vtt', or 'summary' (local-LLM summary).

    Returns:
        Formatted transcript string.
    """
    result = await _get_transcript(video_id, language=language, output_format=format)
    return result.content


def _passage_dict(p) -> dict:
    """Serialise a retrieved passage with its timestamp and deep link."""
    return {
        "timestamp": p.timestamp,
        "start_seconds": int(p.start),
        "score": p.score,
        "text": p.text,
        "video_id": p.video_id,
        "title": p.title,
        "url": p.url(),
    }


@mcp.tool()
async def get_video_info(video_id: str) -> dict:
    """Get rich video metadata (title, channel, views, description, chapters).

    Captions-first and audio-free — equivalent to ``yt-dlp --dump-json`` for the
    essentials, but without downloading the video.

    Args:
        video_id: YouTube video ID or URL.

    Returns:
        Dict with title, author, view_count, length_seconds, publish/upload date,
        category, keywords, description, and a list of chapters.
    """
    try:
        m = await _get_video_info(video_id)
        return {
            "success": True,
            "video_id": m.video_id,
            "title": m.title,
            "author": m.author,
            "channel_id": m.channel_id,
            "length_seconds": m.length_seconds,
            "view_count": m.view_count,
            "publish_date": m.publish_date,
            "upload_date": m.upload_date,
            "category": m.category,
            "keywords": m.keywords,
            "description": m.short_description,
            "chapters": [{"start_seconds": c.start_seconds, "title": c.title} for c in m.chapters],
        }
    except Exception as e:
        return {"success": False, "video_id": video_id, "error": str(e)}


@mcp.tool()
async def list_caption_languages(video_id: str) -> dict:
    """List all caption tracks and translation targets (like ``yt-dlp --list-subs``).

    Args:
        video_id: YouTube video ID or URL.

    Returns:
        Dict with the available tracks (code, language, auto/manual) and the
        languages YouTube can machine-translate into via ``translate``.
    """
    try:
        listing = await _list_languages(video_id)
        return {
            "success": True,
            "video_id": listing.video_id,
            "title": listing.title,
            "tracks": [
                {
                    "code": t.language_code,
                    "language": t.language,
                    "auto_generated": t.is_generated,
                }
                for t in listing.tracks
            ],
            "translation_languages": listing.translation_languages,
        }
    except Exception as e:
        return {"success": False, "video_id": video_id, "error": str(e)}


@mcp.tool()
async def search_transcript(
    video_id: str,
    query: str,
    top_k: int = 5,
    language: str = "en",
) -> dict:
    """Semantic search INSIDE a video — find the exact timestamped moments.

    Fully local (BM25 + local embeddings if configured). Returns ranked passages
    with timestamps and deep-link URLs straight to the moment in the video.

    Args:
        video_id: YouTube video ID or URL.
        query: Natural-language query.
        top_k: Number of passages to return.
        language: Caption language.

    Returns:
        Dict with video_id, title, and a list of passages (timestamp, text, url).
    """
    try:
        res = await _search_in_video(video_id, query, top_k=top_k, language=language)
        return {
            "success": True,
            "video_id": res.video_id,
            "title": res.title,
            "query": query,
            "passages": [_passage_dict(p) for p in res.passages],
        }
    except Exception as e:
        return {"success": False, "video_id": video_id, "error": str(e)}


@mcp.tool()
async def ask_video(
    video_id: str,
    question: str,
    top_k: int = 6,
    language: str = "en",
    answer: bool = True,
) -> dict:
    """Answer a question about a video using ONLY its transcript (local RAG).

    Retrieves the most relevant timestamped passages, then (if a local LLM is
    reachable) generates a grounded answer citing timestamps. If no LLM is
    available, the cited passages are still returned. Nothing leaves the machine.

    Args:
        video_id: YouTube video ID or URL.
        question: The question to answer.
        top_k: Number of passages to retrieve as context.
        language: Caption language.
        answer: If False, skip the LLM and return passages only.

    Returns:
        Dict with answer (or null), the cited passages, and llm_used.
    """
    try:
        res = await _ask_video(video_id, question, top_k=top_k, language=language, answer=answer)
        return {
            "success": True,
            "video_id": res.video_id,
            "title": res.title,
            "question": question,
            "answer": res.answer,
            "llm_used": res.llm_used,
            "note": res.note,
            "passages": [_passage_dict(p) for p in res.passages],
        }
    except Exception as e:
        return {"success": False, "video_id": video_id, "error": str(e)}


@mcp.tool()
async def index_videos(video_ids: list[str], language: str = "en") -> dict:
    """Add videos to the local semantic corpus index for cross-video search.

    Args:
        video_ids: YouTube video IDs or URLs to index.
        language: Caption language.

    Returns:
        Dict with the indexed videos (and chunk counts) and any failures.
    """
    try:
        summary = await _index_videos(video_ids, language=language)
        summary["success"] = True
        return summary
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def find_in_corpus(query: str, top_k: int = 8) -> dict:
    """Semantic search across ALL videos in the local corpus index (exa-style).

    Args:
        query: Natural-language query.
        top_k: Number of passages to return.

    Returns:
        Dict with ranked passages spanning the indexed videos (each with
        video_id, title, timestamp, and deep-link URL).
    """
    try:
        hits = await _find_in_corpus(query, top_k=top_k)
        stats = await _corpus_stats()
        return {
            "success": True,
            "query": query,
            "corpus": {"videos": stats["videos"], "chunks": stats["chunks"]},
            "passages": [_passage_dict(p) for p in hits],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def summarize_video(
    video_id: str,
    language: str = "en",
    model: str | None = None,
) -> dict:
    """Summarize a YouTube video with a LOCAL LLM to save tokens.

    Fetches captions, cleans them, and runs a local model (Ollama by default,
    e.g. qwen3.6:27b) so you ingest a short summary instead of the full
    transcript. Requires a running local LLM; nothing is sent to the cloud.

    Args:
        video_id: YouTube video ID or URL.
        language: Language code (e.g., 'en', 'es').
        model: Optional override of the local model (default: qwen3.6:27b).

    Returns:
        Dict with video_id, source, language, and summary (or error).
    """
    try:
        result = await _get_transcript(
            video_id, language=language, output_format="summary", summary_model=model
        )
        return {
            "video_id": result.video_id,
            "success": True,
            "source": result.source,
            "language": result.language,
            "summary": result.content,
        }
    except Exception as e:
        return {"video_id": video_id, "success": False, "error": str(e)}


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
        format: 'clean' (default), 'text', 'json', 'srt', 'vtt', or 'summary'.
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
    rank: bool = False,
) -> list[dict]:
    """Search YouTube for videos matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 5, max 20).
        language: Language code for transcripts (e.g., 'en', 'es').
        with_transcripts: If True, also fetch transcripts for each video.
        format: Transcript format when with_transcripts is True ('clean' default).
        rank: If True, neural re-rank results by how well each video's TRANSCRIPT
            (not just its title) answers the query — fully local. Adds a
            ``relevance`` score to each result.

    Returns:
        List of video results with video_id, title, channel_name, duration,
        view_count, and (optionally) transcript content / relevance score.
    """
    if rank:
        ranked = await _search_ranked(query, max_results=max_results, language=language)
        return [
            {
                "video_id": v.video_id,
                "title": v.title,
                "channel_name": v.channel_name,
                "duration": v.duration,
                "view_count": v.view_count,
                "relevance": round(float(score), 4),
            }
            for v, score in ranked
        ]

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
