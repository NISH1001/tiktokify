"""CLI interface for tiktokify."""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.command()
@click.option(
    "--base-url",
    "-u",
    required=True,
    help="Base URL of the Jekyll blog (e.g., https://nish1001.github.io)",
)
@click.option(
    "--output-html",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for generated HTML file",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model for enrichment (e.g., gpt-4o-mini, claude-3-haiku-20240307). Skip if not provided.",
)
@click.option(
    "--n-key-points",
    type=int,
    default=5,
    help="Number of key points to generate per post",
)
@click.option(
    "--n-wiki",
    type=int,
    default=3,
    help="Number of Wikipedia articles to suggest per post",
)
@click.option(
    "--sources",
    type=str,
    default="",
    help="Comma-separated external sources to fetch. Available: hackernews (hn), hn-frontpage (frontpage), links (linked)",
)
@click.option(
    "--n-external",
    type=int,
    default=3,
    help="Number of items to fetch per external source",
)
@click.option(
    "--content-weight",
    type=float,
    default=0.6,
    help="Weight for content-based similarity (0-1)",
)
@click.option(
    "--metadata-weight",
    type=float,
    default=0.4,
    help="Weight for tag/category similarity (0-1)",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model for semantic similarity (e.g., sentence-transformers/all-MiniLM-L6-v2 or text-embedding-3-small)",
)
@click.option(
    "--embedding-weight",
    type=float,
    default=0.3,
    help="Weight for semantic similarity (0-1, only used if --embedding-model is set)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of recommendations per post",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=5,
    help="Maximum concurrent requests",
)
@click.option(
    "--max-depth",
    type=int,
    default=1,
    help="Spider crawl depth (0=seed only, 1=seed+linked pages, 2=two levels, etc.)",
)
@click.option(
    "--filter-meta-pages/--no-filter-meta-pages",
    default=True,
    help="Filter out meta pages (tags, categories, about, etc.) from results",
)
@click.option(
    "--min-word-count",
    type=int,
    default=100,
    help="Minimum word count for content filter (default: 100)",
)
@click.option(
    "--min-similarity",
    type=float,
    default=0.0,
    help="Minimum similarity score for recommendations (0.0-1.0, default: 0.0)",
)
@click.option(
    "--use-llm-filter/--no-llm-filter",
    default=False,
    help="Use LLM for content quality assessment (requires --model)",
)
@click.option(
    "--skip-url-filter",
    is_flag=True,
    help="Skip URL pattern filtering",
)
@click.option(
    "--skip-content-filter",
    is_flag=True,
    help="Skip content quality filtering",
)
@click.option(
    "--stealth/--no-stealth",
    default=True,
    help="Enable stealth mode for anti-detection (default: enabled)",
)
@click.option(
    "--headless/--no-headless",
    default=True,
    help="Run browser in headless mode (default: enabled)",
)
@click.option(
    "--follow-external/--no-follow-external",
    default=False,
    help="Follow external links during crawling (default: internal only)",
)
@click.option(
    "--external-depth",
    type=int,
    default=1,
    help="Max depth for external links (default: 1, only directly linked external pages)",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.5,
    help="LLM temperature (0.0-1.0). Default: 0.5",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug mode: limit to 5 posts, skip external sources, enable verbose",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of posts to process (for testing)",
)
@click.option(
    "--save-cache",
    type=click.Path(),
    default=None,
    help="Save intermediate data to JSON file (for reuse)",
)
@click.option(
    "--load-cache",
    type=click.Path(exists=True),
    default=None,
    help="Load from cached JSON file (skip crawling/enrichment)",
)
def main(
    base_url: str,
    output_html: str,
    model: str | None,
    n_key_points: int,
    n_wiki: int,
    sources: str,
    n_external: int,
    content_weight: float,
    metadata_weight: float,
    embedding_model: str | None,
    embedding_weight: float,
    top_k: int,
    max_concurrent: int,
    max_depth: int,
    filter_meta_pages: bool,
    min_word_count: int,
    min_similarity: float,
    use_llm_filter: bool,
    skip_url_filter: bool,
    skip_content_filter: bool,
    stealth: bool,
    headless: bool,
    follow_external: bool,
    external_depth: int,
    temperature: float | None,
    verbose: bool,
    debug: bool,
    limit: int | None,
    save_cache: str | None,
    load_cache: str | None,
) -> None:
    """
    TikTokify - Generate a TikTok-style swipe interface for your Jekyll blog.

    Example:

        uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html

    With LLM enrichment (key points + Wikipedia):

        uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html -m gpt-4o-mini

    With deeper spider crawling:

        uv run tiktokify -u https://example.com -o output.html --max-depth 2

    Debug mode (fast testing with 5 posts):

        uv run tiktokify -u https://example.com -o output.html --debug
    """
    # Debug mode overrides
    if debug:
        verbose = True
        limit = limit or 5
        sources = ""  # Skip external sources in debug mode

    asyncio.run(
        _main_async(
            base_url=base_url,
            output_html=Path(output_html),
            model=model,
            n_key_points=n_key_points,
            n_wiki=n_wiki,
            sources=[s.strip() for s in sources.split(",") if s.strip()],
            n_external=n_external,
            content_weight=content_weight,
            metadata_weight=metadata_weight,
            embedding_model=embedding_model,
            embedding_weight=embedding_weight,
            top_k=top_k,
            max_concurrent=max_concurrent,
            max_depth=max_depth,
            filter_meta_pages=filter_meta_pages,
            min_word_count=min_word_count,
            min_similarity=min_similarity,
            use_llm_filter=use_llm_filter,
            skip_url_filter=skip_url_filter,
            skip_content_filter=skip_content_filter,
            stealth=stealth,
            headless=headless,
            follow_external=follow_external,
            external_depth=external_depth,
            temperature=temperature,
            verbose=verbose,
            limit=limit,
            save_cache=Path(save_cache) if save_cache else None,
            load_cache=Path(load_cache) if load_cache else None,
        )
    )


async def _main_async(
    base_url: str,
    output_html: Path,
    model: str | None,
    n_key_points: int,
    n_wiki: int,
    sources: list[str],
    n_external: int,
    content_weight: float,
    metadata_weight: float,
    embedding_model: str | None,
    embedding_weight: float,
    top_k: int,
    max_concurrent: int,
    max_depth: int,
    filter_meta_pages: bool,
    min_word_count: int,
    min_similarity: float,
    use_llm_filter: bool,
    skip_url_filter: bool,
    skip_content_filter: bool,
    stealth: bool,
    headless: bool,
    follow_external: bool,
    external_depth: int,
    temperature: float | None,
    verbose: bool,
    limit: int | None = None,
    save_cache: Path | None = None,
    load_cache: Path | None = None,
    api_key: str | None = None,
) -> None:
    """Async main function."""
    from tiktokify.crawler import SpiderCrawler
    from tiktokify.enrichment import (
        HackerNewsProvider,
        HNFrontPageProvider,
        LinkedContentProvider,
        PostEnricher,
    )
    from tiktokify.filters import ContentFilter, ContentFilterConfig, URLFilter
    from tiktokify.generator import HTMLGenerator
    from tiktokify.models import ExternalContentItem
    from tiktokify.recommender import RecommendationEngine

    # Map source names to provider classes
    PROVIDERS = {
        "hackernews": HackerNewsProvider,
        "hn": HackerNewsProvider,  # alias
        "hn-frontpage": HNFrontPageProvider,
        "frontpage": HNFrontPageProvider,  # alias
        "links": LinkedContentProvider,
        "linked": LinkedContentProvider,  # alias
    }

    console.print(f"\n[bold blue]TikTokify[/bold blue] - Generating swipe UI for {base_url}\n")

    # Load from cache if provided (skip all crawling/enrichment)
    if load_cache:
        from tiktokify.models import RecommendationGraph

        console.print(f"  [cyan]📦[/cyan] Loading from cache: {load_cache}")
        with open(load_cache) as f:
            cache_data = json.load(f)
        graph = RecommendationGraph.model_validate(cache_data)
        console.print(f"  [green]✓[/green] Loaded {len(graph.posts)} posts from cache")

        # Generate HTML directly
        generator = HTMLGenerator()
        generator.generate(graph, base_url, output_html)
        console.print(f"  [green]✓[/green] Generated {output_html}")
        console.print(f"\n[bold green]Done![/bold green] Open {output_html} in a browser to view.\n")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Spider crawl with URL filter
        url_filter = None if skip_url_filter else URLFilter()
        depth_info = f" (depth={max_depth})" if max_depth > 1 else ""
        task = progress.add_task(f"Spider crawling{depth_info}...", total=None)
        crawler = SpiderCrawler(
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_depth=max_depth,
            verbose=verbose,
            url_filter=url_filter,
            stealth=stealth,
            headless=headless,
            follow_external=follow_external,
            external_depth=external_depth,
        )
        posts = await crawler.crawl()
        progress.remove_task(task)

        if not posts:
            console.print("[red]Error: No posts found![/red]")
            return

        filter_info = "" if skip_url_filter else " (URL filtered)"
        console.print(f"  [green]✓[/green] Found {len(posts)} posts{filter_info}")

        # Apply limit if specified (for debugging/testing)
        if limit and len(posts) > limit:
            posts = posts[:limit]
            console.print(f"  [yellow]⚡[/yellow] Limited to {limit} posts (--debug/--limit)")

        # Step 1.5: Content quality filtering (optional)
        if not skip_content_filter:
            original_count = len(posts)
            task = progress.add_task("Filtering by content quality...", total=None)
            content_filter = ContentFilter(
                config=ContentFilterConfig(min_word_count=min_word_count),
                model=model if use_llm_filter else None,
                max_concurrent=max_concurrent,
                verbose=verbose,
            )
            posts, rejected = await content_filter.filter(posts)
            progress.remove_task(task)
            if rejected:
                console.print(
                    f"  [green]✓[/green] Content filter: kept {len(posts)}, "
                    f"removed {len(rejected)} low-quality pages"
                )
            else:
                console.print(f"  [green]✓[/green] Content filter: all {len(posts)} pages passed")

        if not posts:
            console.print("[red]Error: No posts after content filtering![/red]")
            return

        # Step 1.6: Relevancy filtering (optional)
        if filter_meta_pages:
            from tiktokify.relevancy import RelevancyClassifier

            original_count = len(posts)
            task = progress.add_task("Filtering meta pages...", total=None)
            classifier = RelevancyClassifier(
                model=model,
                max_concurrent=max_concurrent,
                temperature=temperature,
                verbose=verbose,
            )
            posts = await classifier.classify_posts(posts)
            progress.remove_task(task)
            filtered_count = original_count - len(posts)
            console.print(
                f"  [green]✓[/green] Filtered to {len(posts)} content pages "
                f"(removed {filtered_count} meta pages)"
            )

        if not posts:
            console.print("[red]Error: No content posts after filtering![/red]")
            return

        # Step 2: Build recommendations
        embed_info = f" + {embedding_model}" if embedding_model else ""
        task = progress.add_task(f"Building recommendation graph{embed_info}...", total=None)
        engine = RecommendationEngine(
            content_weight=content_weight,
            metadata_weight=metadata_weight,
            embedding_weight=embedding_weight,
            embedding_model=embedding_model,
            top_k=top_k,
            min_similarity=min_similarity,
            max_concurrent=max_concurrent,
            verbose=verbose,
        )
        graph = await engine.build_graph(posts)
        progress.remove_task(task)
        threshold_info = f" (min_similarity={min_similarity})" if min_similarity > 0 else ""
        embed_suffix = f" with semantic embeddings" if embedding_model else ""
        console.print(f"  [green]✓[/green] Built recommendation graph{threshold_info}{embed_suffix}")

        # Step 3: LLM enrichment (optional)
        if model:
            task = progress.add_task(f"Enriching posts with LLM ({model})...", total=None)
            enricher = PostEnricher(
                model=model,
                max_key_points=n_key_points,
                max_wikipedia=n_wiki,
                max_concurrent=max_concurrent,
                temperature=temperature,
                verbose=verbose,
                api_key=api_key,
            )
            await enricher.enrich_posts(list(graph.posts.values()))
            progress.remove_task(task)

            enriched_count = sum(
                1 for p in graph.posts.values() if p.key_points
            )
            console.print(f"  [green]✓[/green] Enriched {enriched_count} posts with key points + Wikipedia")
        else:
            console.print("  [dim]⊘ Skipping LLM enrichment (no --model specified)[/dim]")

        # Step 4: External sources (optional)
        if sources:
            valid_sources = [s for s in sources if s in PROVIDERS]
            if valid_sources:
                task = progress.add_task(f"Fetching from {', '.join(valid_sources)}...", total=None)

                # Build list of (provider, post) pairs for parallel fetching
                fetch_tasks = []
                task_info = []  # Track (source_name, post) for each task

                for source_name in valid_sources:
                    provider_class = PROVIDERS[source_name]
                    provider = provider_class(max_items=n_external, verbose=verbose)

                    for post in graph.posts.values():
                        fetch_tasks.append(provider.fetch_for_post(post))
                        task_info.append((source_name, post))

                # Fetch all in parallel with concurrency limit
                semaphore = asyncio.Semaphore(max_concurrent)

                async def fetch_with_limit(coro, info):
                    async with semaphore:
                        try:
                            return await coro, info, None
                        except Exception as e:
                            return [], info, e

                results = await asyncio.gather(
                    *[fetch_with_limit(t, info) for t, info in zip(fetch_tasks, task_info)]
                )

                # Process results
                for external_items, (source_name, post), error in results:
                    if error:
                        if verbose:
                            console.print(f"[yellow]Warning: {source_name} failed for {post.slug}: {error}[/yellow]")
                        continue

                    for item in external_items:
                        post.external_content.append(
                            ExternalContentItem(
                                source=item.source,
                                title=item.title,
                                url=item.url,
                                description=item.description,
                                relevance=item.relevance,
                                metadata=item.metadata,
                            )
                        )

                progress.remove_task(task)
                console.print(f"  [green]✓[/green] Fetched external content from {', '.join(valid_sources)}")
            else:
                console.print(f"  [yellow]⚠ Unknown sources: {sources}. Available: {list(PROVIDERS.keys())}[/yellow]")

        # Save cache if requested (before HTML generation)
        if save_cache:
            task = progress.add_task("Saving cache...", total=None)
            with open(save_cache, "w") as f:
                json.dump(graph.model_dump(mode="json"), f, indent=2)
            progress.remove_task(task)
            console.print(f"  [green]✓[/green] Saved cache to {save_cache}")

        # Step 5: Generate HTML
        task = progress.add_task("Generating HTML...", total=None)
        generator = HTMLGenerator()
        generator.generate(graph, base_url, output_html)
        progress.remove_task(task)
        console.print(f"  [green]✓[/green] Generated {output_html}")

    console.print(f"\n[bold green]Done![/bold green] Open {output_html} in a browser to view.\n")


if __name__ == "__main__":
    main()
