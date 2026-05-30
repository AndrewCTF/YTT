#!/usr/bin/env python3
"""CLI for YouTube transcript fetching."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .service import (
    ask_video,
    corpus_stats,
    find_in_corpus,
    get_transcript,
    get_transcripts_batch,
    get_video_info,
    index_videos,
    list_languages,
)
from .cache import cache
from .search_service import search, search_and_get_transcripts, search_ranked
from .semantic import seconds_to_clock

console = Console(legacy_windows=False, force_terminal=True)


def validate_video_id(ctx, param, value):
    """Validate video ID or URL."""
    if value is None:
        return None

    # Basic validation - could be video ID or URL
    if "/" in value or "?" in value:
        # Looks like a URL - let the service handle it
        return value

    # 11-char video ID
    if len(value) == 11:
        return value

    raise click.BadParameter("Must be a valid YouTube video ID or URL")


@click.group()
def cli():
    """YouTube Transcript Fetcher - Get transcripts from any YouTube video."""
    pass


@cli.command()
@click.argument("video_id", callback=validate_video_id)
@click.option(
    "--language",
    "-l",
    default="en",
    help="Language code (e.g., en, es, fr, de)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["clean", "text", "json", "srt", "vtt", "summary"]),
    default="clean",
    help="Output format (clean = deduplicated, LLM-friendly; summary = local-LLM summary)",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Summarize with a local LLM to save tokens (same as --format summary)",
)
@click.option(
    "--summary-model",
    default=None,
    help="Override the local summary model (default: qwen3.6:27b)",
)
@click.option(
    "--translate",
    "-t",
    default=None,
    help="Machine-translate captions into this language code (e.g. es, fr, ja)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: stdout)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip cache lookup",
)
@click.option(
    "--no-whisper",
    is_flag=True,
    help="Disable Whisper fallback",
)
def transcript(
    video_id, language, format, summarize, summary_model, translate, output, no_cache, no_whisper
):
    """Get transcript for a YouTube video."""
    if summarize:
        format = "summary"

    async def fetch():
        return await get_transcript(
            video_id,
            language=language,
            output_format=format,
            use_cache=not no_cache,
            use_whisper_fallback=not no_whisper,
            summary_model=summary_model,
            translate=translate,
        )

    console.print("Fetching transcript...")
    try:
        result = asyncio.run(fetch())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Display result
    if format == "json":
        syntax = Syntax(result.content, "json", theme="monokai", line_numbers=False)
        panel = Panel(syntax, title=f"[bold]{result.video_id}[/bold]")
    else:
        panel = Panel(
            result.content[:2000] + ("..." if len(result.content) > 2000 else ""),
            title=f"[bold]{result.video_id}[/bold]",
            subtitle=f"Source: {result.source} | Language: {result.language}",
        )

    console.print(panel)

    if output:
        Path(output).write_text(result.content, encoding="utf-8")
        console.print(f"[green]Saved to {output}[/green]")


@cli.command()
@click.argument("video_ids", nargs=-1, callback=lambda ctx, param, values: values)
@click.option(
    "--language",
    "-l",
    default="en",
    help="Language code",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["clean", "text", "json", "srt", "vtt", "summary"]),
    default="clean",
    help="Output format (clean = deduplicated; summary = local-LLM summary)",
)
@click.option(
    "--summary-model",
    default=None,
    help="Override the local summary model (default: qwen3.6:27b)",
)
@click.option(
    "--workers",
    "-w",
    default=4,
    help="Max concurrent workers",
)
def batch(video_ids, language, format, summary_model, workers):
    """Get transcripts for multiple videos."""

    if not video_ids:
        console.print("[yellow]No video IDs provided[/yellow]")
        return

    async def fetch_all():
        return await get_transcripts_batch(
            video_ids,
            language=language,
            output_format=format,
            max_workers=workers,
            summary_model=summary_model,
        )

    console.print(f"Fetching transcripts for {len(video_ids)} videos (workers={workers})...")

    results = asyncio.run(fetch_all())

    success = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - success

    console.print(f"\n[green]Success: {success}[/green] | [red]Failed: {failed}[/red]")

    for vid, r in zip(video_ids, results):
        if isinstance(r, Exception):
            console.print(f"  [red][FAIL][/red] {vid}: {r}")
        else:
            console.print(f"  [green][OK][/green] {vid} ({r.source})")


@cli.command()
@click.option("--clean", is_flag=True, help="Remove expired entries")
def cache_stats(clean):
    """Show cache statistics."""

    async def do_cache():
        if clean:
            deleted = await cache.cleanup_expired()
            console.print(f"[yellow]Removed {deleted} expired entries[/yellow]")

        stats = await cache.get_stats()

        console.print(
            Panel(
                f"""[bold]Cache Statistics[/bold]

Total entries: {stats['total_entries']}
Expired: {stats['expired_entries']}
By source: {stats['entries_by_source']}
""",
                title="Cache",
            )
        )

    asyncio.run(do_cache())


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    "-n",
    default=5,
    type=click.IntRange(1, 20),
    help="Number of results to return",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "table"]),
    default="table",
    help="Output format",
)
@click.option(
    "--with-transcripts",
    is_flag=True,
    help="Also fetch transcripts for each result",
)
@click.option(
    "--rank",
    is_flag=True,
    help="Neural re-rank results by transcript relevance to the query (fully local)",
)
@click.option(
    "--language",
    "-l",
    default="en",
    help="Language for transcripts",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip cache lookup",
)
def search_cmd(query, limit, format, with_transcripts, rank, language, no_cache):
    """Search YouTube for videos."""
    from rich.table import Table

    if rank:
        _run_ranked_search(query, limit, language, not no_cache)
        return

    async def do_search():
        if with_transcripts:
            return await search_and_get_transcripts(
                query, max_results=limit, language=language, use_cache=not no_cache
            )
        return await search(query, max_results=limit, use_cache=not no_cache)

    console.print(f"Searching YouTube for: '{query}'...")

    try:
        results = asyncio.run(do_search())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    if format == "json":
        import json

        output = []
        for r in results:
            if with_transcripts:
                video, transcript = r
                item = {
                    "video_id": video.video_id,
                    "title": video.title,
                    "channel": video.channel_name,
                    "duration": video.duration,
                    "views": video.view_count,
                }
                if transcript:
                    item["transcript"] = transcript.content
                output.append(item)
            else:
                output.append(
                    {
                        "video_id": r.video_id,
                        "title": r.title,
                        "channel": r.channel_name,
                        "duration": r.duration,
                        "views": r.view_count,
                    }
                )
        console.print(json.dumps(output, indent=2))
    else:
        table = Table(title=f"Search Results: '{query}'")
        table.add_column("Video ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Channel", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Views", justify="right")

        for r in results:
            video = r if hasattr(r, "video_id") else r[0]
            title = video.title[:50] + "..." if len(video.title) > 50 else video.title
            table.add_row(
                video.video_id,
                title,
                video.channel_name,
                video.duration,
                video.view_count,
            )

        console.print(table)

    if with_transcripts:
        success_count = (
            sum(1 for r in results if r[1] is not None)
            if results and not hasattr(results[0], "video_id")
            else 0
        )
        console.print(f"\n[dim]Transcripts fetched for {success_count}/{len(results)} videos[/dim]")


def _print_passages(passages, header: str | None = None) -> None:
    """Render retrieved passages as timestamped, deep-linkable snippets."""
    from rich.table import Table

    if header:
        console.print(header)
    if not passages:
        console.print("[yellow]No relevant passages found.[/yellow]")
        return
    table = Table(show_lines=True)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Passage", style="white")
    table.add_column("Link", style="blue")
    for p in passages:
        vid_title = f"\n[dim]{p.title}[/dim]" if getattr(p, "title", None) else ""
        table.add_row(p.timestamp, f"{p.score:.3f}", p.text + vid_title, p.url())
    console.print(table)


def _run_ranked_search(query, limit, language, use_cache) -> None:
    from rich.table import Table

    console.print(f"Neural re-ranking YouTube results for: '{query}'...")
    try:
        ranked = asyncio.run(
            search_ranked(query, max_results=limit, language=language, use_cache=use_cache)
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    if not ranked:
        console.print("[yellow]No results found[/yellow]")
        return
    table = Table(title=f"Neural-ranked results: '{query}'")
    table.add_column("Rank", justify="right")
    table.add_column("Relevance", justify="right", style="magenta")
    table.add_column("Video ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Channel", style="green")
    for i, (video, score) in enumerate(ranked, 1):
        title = video.title[:50] + "..." if len(video.title) > 50 else video.title
        table.add_row(str(i), f"{score:.3f}", video.video_id, title, video.channel_name)
    console.print(table)


@cli.command()
@click.argument("video_id", callback=validate_video_id)
def info(video_id):
    """Show rich video metadata and chapters (no audio download)."""
    try:
        meta = asyncio.run(get_video_info(video_id))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    mins, secs = divmod(meta.length_seconds, 60)
    body = (
        f"[bold]{meta.title}[/bold]\n"
        f"[green]{meta.author}[/green]  ·  {meta.view_count:,} views  ·  {mins}m{secs:02d}s\n"
        f"Published: {meta.publish_date or 'N/A'}   Category: {meta.category or 'N/A'}\n"
    )
    if meta.keywords:
        body += f"[dim]Keywords:[/dim] {', '.join(meta.keywords[:12])}\n"
    if meta.short_description:
        desc = meta.short_description[:500]
        body += f"\n{desc}{'...' if len(meta.short_description) > 500 else ''}"
    console.print(Panel(body, title=f"[bold]{meta.video_id}[/bold]"))

    if meta.chapters:
        from rich.table import Table

        table = Table(title="Chapters")
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        for ch in meta.chapters:
            table.add_row(seconds_to_clock(ch.start_seconds), ch.title)
        console.print(table)


@cli.command()
@click.argument("video_id", callback=validate_video_id)
def langs(video_id):
    """List available caption languages and translation targets (like yt-dlp --list-subs)."""
    from rich.table import Table

    try:
        listing = asyncio.run(list_languages(video_id))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    table = Table(title=f"Caption tracks: {listing.title}")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    table.add_column("Type", style="green")
    for t in listing.tracks:
        table.add_row(t.language_code, t.language, "auto" if t.is_generated else "manual")
    console.print(table)

    if listing.translation_languages:
        codes = ", ".join(sorted(t["code"] for t in listing.translation_languages))
        console.print(
            f"\n[dim]Translatable into {len(listing.translation_languages)} languages "
            f"(use --translate CODE):[/dim]\n{codes}"
        )


@cli.command()
@click.argument("video_id", callback=validate_video_id)
@click.argument("question")
@click.option("--top-k", "-k", default=6, type=click.IntRange(1, 20), help="Passages to retrieve")
@click.option("--language", "-l", default="en", help="Caption language")
@click.option("--passages-only", is_flag=True, help="Skip the LLM answer; just show passages")
@click.option("--model", default=None, help="Local LLM model for the answer")
@click.option(
    "--embed-provider", default=None, help="auto|hash|sentence-transformers|ollama|openai"
)
@click.option("--no-cache", is_flag=True, help="Skip cache lookup")
def ask(video_id, question, top_k, language, passages_only, model, embed_provider, no_cache):
    """Ask a question about a video — local semantic retrieval + grounded answer."""
    console.print(f"Searching transcript for: '{question}'...")
    try:
        res = asyncio.run(
            ask_video(
                video_id,
                question,
                top_k=top_k,
                language=language,
                use_cache=not no_cache,
                answer=not passages_only,
                model=model,
                embed_provider=embed_provider,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if res.answer:
        console.print(Panel(res.answer, title=f"[bold]Answer · {res.title}[/bold]"))
    elif res.note:
        console.print(f"[yellow]{res.note}[/yellow]")
    _print_passages(res.passages, header="\n[bold]Sources[/bold]")


@cli.command(name="index")
@click.argument("video_ids", nargs=-1, callback=lambda ctx, param, values: values)
@click.option("--language", "-l", default="en", help="Caption language")
@click.option(
    "--embed-provider", default=None, help="auto|hash|sentence-transformers|ollama|openai"
)
@click.option("--embed-model", default=None, help="Embedding model name")
@click.option("--db", default=None, help="Corpus index DB path (default: .ytt_index.db)")
@click.option("--no-cache", is_flag=True, help="Skip cache lookup")
def index_cmd(video_ids, language, embed_provider, embed_model, db, no_cache):
    """Add videos to the local semantic corpus index for cross-video search."""
    if not video_ids:
        console.print("[yellow]No video IDs provided[/yellow]")
        return
    console.print(f"Indexing {len(video_ids)} video(s)...")
    try:
        summary = asyncio.run(
            index_videos(
                list(video_ids),
                language=language,
                use_cache=not no_cache,
                embed_provider=embed_provider,
                embed_model=embed_model,
                db_path=db,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    for item in summary["indexed"]:
        console.print(
            f"  [green][OK][/green] {item['video_id']} — {item['chunks']} chunks "
            f"([dim]{item['embedder']}[/dim]) {item['title']}"
        )
    for vid, err in summary["failed"].items():
        console.print(f"  [red][FAIL][/red] {vid}: {err}")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=8, type=click.IntRange(1, 50), help="Passages to return")
@click.option(
    "--embed-provider", default=None, help="auto|hash|sentence-transformers|ollama|openai"
)
@click.option("--embed-model", default=None, help="Embedding model name")
@click.option("--db", default=None, help="Corpus index DB path (default: .ytt_index.db)")
def find(query, top_k, embed_provider, embed_model, db):
    """Semantic search across every video in the local corpus index."""
    try:
        hits = asyncio.run(
            find_in_corpus(
                query,
                top_k=top_k,
                embed_provider=embed_provider,
                embed_model=embed_model,
                db_path=db,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    _print_passages(hits, header=f"[bold]Corpus search:[/bold] '{query}'")


@cli.command()
@click.option("--db", default=None, help="Corpus index DB path (default: .ytt_index.db)")
def corpus(db):
    """Show local corpus index statistics."""
    try:
        stats = asyncio.run(corpus_stats(db_path=db))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    from rich.table import Table

    console.print(
        Panel(
            f"Videos: {stats['videos']}\nChunks: {stats['chunks']}\n"
            f"Embedders: {', '.join(stats['embedders']) or 'none (BM25 only)'}",
            title="Corpus Index",
        )
    )
    if stats.get("video_list"):
        table = Table()
        table.add_column("Video ID", style="cyan")
        table.add_column("Chunks", justify="right")
        table.add_column("Title", style="white")
        for v in stats["video_list"]:
            table.add_row(v["video_id"], str(v["chunks"]), v["title"])
        console.print(table)


if __name__ == "__main__":
    cli()
