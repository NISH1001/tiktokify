"""LLM-based post enrichment using litellm.

This module uses LLM to:
1. Generate key points/takeaways for each post
2. Suggest relevant Wikipedia articles

The actual Wikipedia extract fetching is done by providers/wikipedia.py
"""

import asyncio
import json
import re

import litellm
from pydantic import ValidationError
from rich.console import Console

from tiktokify.models import Post, WikipediaSuggestion

console = Console()

# Disable litellm's verbose logging
litellm.suppress_debug_info = True
# Drop unsupported params (e.g., temperature for GPT-5 models)
litellm.drop_params = True


class PostEnricher:
    """Enrich posts with key points and Wikipedia suggestions using LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_key_points: int = 5,
        max_wikipedia: int = 3,
        max_concurrent: int = 3,
        temperature: float | None = None,
        verbose: bool = False,
    ):
        self.model = model
        self.max_key_points = max_key_points
        self.max_wikipedia = max_wikipedia
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.verbose = verbose
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def enrich_post(self, post: Post) -> None:
        """Enrich a single post with key points and Wikipedia suggestions."""
        prompt = self._build_prompt(post)

        try:
            # Start with base tokens, backoff with 2x multiplier on empty response
            base_tokens = (self.max_key_points * 50) + (self.max_wikipedia * 100) + 200
            max_tokens = max(2000, base_tokens * 2)
            max_retries = 3
            content = None

            for attempt in range(max_retries):
                completion_kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                if self.temperature is not None:
                    completion_kwargs["temperature"] = self.temperature

                response = await litellm.acompletion(**completion_kwargs)
                content = response.choices[0].message.content

                if content:
                    break  # Got a response, exit retry loop

                # Empty response - backoff with 2x tokens
                if self.verbose:
                    console.print(
                        f"[yellow]Empty response for {post.slug}, "
                        f"retrying with {max_tokens * 2} tokens...[/yellow]"
                    )
                max_tokens = min(max_tokens * 2, 16000)  # Cap at 16k

            if not content and self.verbose:
                console.print(f"[yellow]No response after {max_retries} retries for {post.slug}[/yellow]")
            key_points, wikipedia = self._parse_response(content)

            # Fetch Wikipedia extracts for each suggestion
            wikipedia_with_extracts = await self._fetch_wiki_extracts(wikipedia)

            post.key_points = key_points
            post.wikipedia_suggestions = wikipedia_with_extracts

        except Exception as e:
            if self.verbose:
                console.print(
                    f"[yellow]Warning: LLM call failed for {post.slug}: {e}[/yellow]"
                )

    async def _fetch_wiki_extracts(
        self, suggestions: list[WikipediaSuggestion]
    ) -> list[WikipediaSuggestion]:
        """Fetch Wikipedia extracts for all suggestions concurrently."""
        from tiktokify.enrichment.providers.wikipedia import WikipediaProvider

        provider = WikipediaProvider(max_items=len(suggestions), verbose=self.verbose)

        async def fetch_one(suggestion: WikipediaSuggestion) -> WikipediaSuggestion:
            extract = await provider._fetch_extract(
                provider._extract_title_from_url(str(suggestion.url)) or suggestion.title
            )
            return WikipediaSuggestion(
                title=suggestion.title,
                url=suggestion.url,
                relevance=suggestion.relevance,
                extract=extract,
            )

        tasks = [fetch_one(s) for s in suggestions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, WikipediaSuggestion)]

    def _build_prompt(self, post: Post) -> str:
        """Build LLM prompt for key points and Wikipedia suggestions."""
        content_excerpt = post.content_text[:2000] if post.content_text else ""

        return f"""Analyze this blog post and provide:
1. {self.max_key_points} key points/takeaways (concise bullet points)
2. {self.max_wikipedia} relevant Wikipedia articles for further reading

Title: {post.metadata.title}
Subtitle: {post.metadata.subtitle or "N/A"}
Categories: {', '.join(post.metadata.categories)}
Tags: {', '.join(post.metadata.tags)}

Content:
{content_excerpt}

Return ONLY valid JSON with this exact structure:
{{
  "keyPoints": ["point 1", "point 2", ...],
  "wikipedia": [
    {{"title": "Article Title", "url": "https://en.wikipedia.org/wiki/...", "relevance": "Why it's relevant"}}
  ]
}}

Guidelines:
- Key points should be insightful takeaways, not just summaries
- Each key point should be 1-2 sentences max
- Wikipedia URLs must be valid (use underscores for spaces)
- Return ONLY the JSON, no markdown formatting"""

    def _parse_response(self, content: str | None) -> tuple[list[str], list[WikipediaSuggestion]]:
        """Parse LLM response into key points and Wikipedia suggestions."""
        key_points: list[str] = []
        wikipedia: list[WikipediaSuggestion] = []

        # Handle empty or None content
        if not content:
            if self.verbose:
                console.print("[yellow]LLM returned empty response[/yellow]")
            return key_points, wikipedia

        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines if not line.startswith("```")
            )

        # Fix unescaped backslashes from LaTeX (e.g., \overline -> \\overline)
        # JSON requires \\ for literal backslash, but LLMs often return raw LaTeX
        content = re.sub(r'\\([^"\\nrtbfu/])', r'\\\\\1', content)

        try:
            data = json.loads(content)

            # Parse key points
            if "keyPoints" in data and isinstance(data["keyPoints"], list):
                key_points = [str(p) for p in data["keyPoints"] if p]

            # Parse Wikipedia suggestions
            if "wikipedia" in data and isinstance(data["wikipedia"], list):
                for item in data["wikipedia"]:
                    try:
                        suggestion = WikipediaSuggestion(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            relevance=item.get("relevance", ""),
                            extract="",  # Will be filled later
                        )
                        wikipedia.append(suggestion)
                    except ValidationError:
                        continue

        except json.JSONDecodeError as e:
            # Only show in verbose mode - empty responses are handled gracefully
            if self.verbose:
                console.print(f"[yellow]JSON parse error: {e}[/yellow]")
                if content:
                    console.print(f"[dim]Raw content: {content[:200]}...[/dim]")

        return key_points, wikipedia

    async def enrich_posts(self, posts: list[Post]) -> None:
        """Enrich all posts concurrently."""

        async def enrich_one(post: Post) -> None:
            async with self.semaphore:
                await self.enrich_post(post)

        tasks = [enrich_one(post) for post in posts]
        await asyncio.gather(*tasks, return_exceptions=True)


# Backwards compatibility alias
WikipediaSuggester = PostEnricher
