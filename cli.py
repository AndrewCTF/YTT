#!/usr/bin/env python3
"""CLI for YouTube transcript fetching."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from src.service import get_transcript, get_transcripts_batch
from src.cache import cache
from src.search_service import search, search_and_get_transcripts


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
    "--language", "-l",
    default="en",
    help="Language code (e.g., en, es, fr, de)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "srt", "vtt"]),
    default="text",
    help="Output format",
)
@click.option(
    "--output", "-o",
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
def transcript(video_id, language, format, output, no_cache, no_whisper):
    """Get transcript for a YouTube video."""

    async def fetch():
        return await get_transcript(
            video_id,
            language=language,
            output_format=format,
            use_cache=not no_cache,
            use_whisper_fallback=not no_whisper,
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
    "--language", "-l",
    default="en",
    help="Language code",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "srt", "vtt"]),
    default="text",
    help="Output format",
)
@click.option(
    "--workers", "-w",
    default=4,
    help="Max concurrent workers",
)
def batch(video_ids, language, format, workers):
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

        console.print(Panel(
            f"""[bold]Cache Statistics[/bold]

Total entries: {stats['total_entries']}
Expired: {stats['expired_entries']}
By source: {stats['entries_by_source']}
""",
            title="Cache",
        ))

    asyncio.run(do_cache())


@cli.command()
@click.argument("query")
@click.option(
    "--limit", "-n",
    default=5,
    type=click.IntRange(1, 20),
    help="Number of results to return",
)
@click.option(
    "--format", "-f",
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
    "--language", "-l",
    default="en",
    help="Language for transcripts",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip cache lookup",
)
def search_cmd(query, limit, format, with_transcripts, language, no_cache):
    """Search YouTube for videos."""
    from rich.table import Table

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
                output.append({
                    "video_id": r.video_id,
                    "title": r.title,
                    "channel": r.channel_name,
                    "duration": r.duration,
                    "views": r.view_count,
                })
        console.print(json.dumps(output, indent=2))
    else:
        table = Table(title=f"Search Results: '{query}'")
        table.add_column("Video ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Channel", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Views", justify="right")

        for r in results:
            video = r if hasattr(r, 'video_id') else r[0]
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
        success_count = sum(1 for r in results if r[1] is not None) if results and not hasattr(results[0], 'video_id') else 0
        console.print(f"\n[dim]Transcripts fetched for {success_count}/{len(results)} videos[/dim]")


if __name__ == "__main__":
    cli()