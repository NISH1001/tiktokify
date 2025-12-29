"""Relevancy classifier for filtering meta pages from content pages."""

import asyncio
import json

import litellm
from rich.console import Console

from tiktokify.models import Post
from tiktokify.relevancy.patterns import RelevancyResult, classify_by_url

console = Console()

# Disable litellm's verbose logging
litellm.suppress_debug_info = True


class RelevancyClassifier:
    """Hybrid relevancy classifier using URL patterns + optional LLM."""

    def __init__(
        self,
        model: str | None = None,
        max_concurrent: int = 5,
        verbose: bool = False,
    ):
        """
        Initialize classifier.

        Args:
            model: LLM model for edge cases (None = patterns only)
            max_concurrent: Max concurrent LLM requests
            verbose: Enable verbose logging
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def classify_posts(self, posts: list[Post]) -> list[Post]:
        """
        Classify all posts and filter to relevant ones.

        Returns:
            List of posts classified as relevant
        """
        # Phase 1: URL pattern classification (sync, fast)
        relevant: list[Post] = []
        maybe: list[Post] = []
        filtered_count = 0

        for post in posts:
            result, reason = classify_by_url(post.url)

            if result == RelevancyResult.RELEVANT:
                post.is_relevant = True
                post.relevancy_reason = reason
                relevant.append(post)
                if self.verbose:
                    console.print(f"[green]✓ Kept:[/green] {post.url}")
                    console.print(f"  [dim]→ {reason}[/dim]")
            elif result == RelevancyResult.NOT_RELEVANT:
                post.is_relevant = False
                post.relevancy_reason = reason
                filtered_count += 1
                if self.verbose:
                    console.print(f"[red]✗ Filtered:[/red] {post.url}")
                    console.print(f"  [dim]→ {reason}[/dim]")
            else:  # MAYBE
                maybe.append(post)
                if self.verbose:
                    console.print(f"[yellow]? Uncertain:[/yellow] {post.url}")
                    console.print(f"  [dim]→ {reason} (will use LLM if available)[/dim]")

        if self.verbose:
            console.print(
                f"[dim]Pattern classification: {len(relevant)} relevant, "
                f"{filtered_count} filtered, {len(maybe)} uncertain[/dim]"
            )

        # Phase 2: LLM classification for uncertain posts
        if maybe and self.model:
            llm_relevant = await self._classify_with_llm(maybe)
            relevant.extend(llm_relevant)
        elif maybe:
            # No LLM available - be permissive, include uncertain posts
            for post in maybe:
                post.is_relevant = True
                post.relevancy_reason = "Included by default (no LLM for verification)"
            relevant.extend(maybe)

        return relevant

    async def _classify_with_llm(self, posts: list[Post]) -> list[Post]:
        """Use LLM to classify uncertain posts."""

        async def classify_one(post: Post) -> Post:
            async with self.semaphore:
                is_relevant, reason = await self._llm_classify_single(post)
                post.is_relevant = is_relevant
                post.relevancy_reason = f"LLM: {reason}"
                return post

        tasks = [classify_one(post) for post in posts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        relevant = []
        for i, result in enumerate(results):
            if isinstance(result, Post):
                if result.is_relevant:
                    relevant.append(result)
                elif self.verbose:
                    console.print(
                        f"[dim]LLM filtered: {posts[i].url} ({result.relevancy_reason})[/dim]"
                    )
            elif isinstance(result, Exception):
                # On error, be permissive
                posts[i].is_relevant = True
                posts[i].relevancy_reason = "Included by default (LLM error)"
                relevant.append(posts[i])
                if self.verbose:
                    console.print(
                        f"[yellow]LLM classification failed for {posts[i].url}: {result}[/yellow]"
                    )

        return relevant

    async def _llm_classify_single(self, post: Post) -> tuple[bool, str]:
        """Classify a single post using LLM."""
        prompt = self._build_classification_prompt(post)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temp for consistent classification
                max_tokens=200,
            )

            content = response.choices[0].message.content
            return self._parse_llm_response(content)

        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]LLM error for {post.slug}: {e}[/yellow]")
            return True, "classification failed"

    def _build_classification_prompt(self, post: Post) -> str:
        """Build LLM prompt for content classification."""
        # Use title, URL, and content excerpt for classification
        content_excerpt = post.content_text[:500] if post.content_text else ""

        return f"""Classify whether this webpage is a CONTENT page (blog post, article, essay) or a META page (tag listing, category page, about page, archive, etc.).

URL: {post.url}
Title: {post.metadata.title}
Content excerpt: {content_excerpt}

Return ONLY valid JSON:
{{"is_content": true, "reason": "brief explanation"}}
or
{{"is_content": false, "reason": "brief explanation"}}

Guidelines:
- Blog posts, articles, tutorials, essays = CONTENT (is_content: true)
- Tag listings, category pages, archives, about pages, contact pages = META (is_content: false)
- If the title contains "Posts tagged with" or "Category:" it's META
- If content has actual paragraphs of text discussing a topic, it's CONTENT"""

    def _parse_llm_response(self, content: str) -> tuple[bool, str]:
        """Parse LLM classification response."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(line for line in lines if not line.startswith("```"))

        try:
            data = json.loads(content)
            is_content = data.get("is_content", True)
            reason = data.get("reason", "no reason provided")
            return is_content, reason
        except json.JSONDecodeError:
            # Fallback parsing
            if "false" in content.lower():
                return False, "classified as meta page"
            return True, "classified as content"
