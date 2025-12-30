"""Content filter for post-crawl validation of page content."""

import asyncio
import json
import re

import litellm
from pydantic import BaseModel
from rich.console import Console

from tiktokify.filters.base import BaseFilter
from tiktokify.models import Post

console = Console()

# Disable litellm's verbose logging
litellm.suppress_debug_info = True
litellm.drop_params = True


class ContentFilterConfig(BaseModel):
    """Configuration for content filtering thresholds."""

    min_word_count: int = 100
    min_paragraphs: int = 1  # Lowered from 3 - paragraph detection is unreliable
    min_text_density: float = 0.3  # Not used (modern HTML is bloated)
    max_link_ratio: float = 0.5


class ContentFilter(BaseFilter[Post]):
    """Post-crawl content validation filter.

    Uses LLM (when available) or heuristics to determine if a page
    contains actual content worth including in recommendations.
    """

    def __init__(
        self,
        config: ContentFilterConfig | None = None,
        model: str | None = None,
        max_concurrent: int = 5,
        verbose: bool = False,
    ):
        """Initialize content filter.

        Args:
            config: Content filtering thresholds
            model: LLM model for assessment (None = heuristics only)
            max_concurrent: Max concurrent LLM requests
            verbose: Enable verbose logging
        """
        self.config = config or ContentFilterConfig()
        self.model = model
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def name(self) -> str:
        return "ContentFilter"

    async def filter(
        self, posts: list[Post]
    ) -> tuple[list[Post], list[tuple[Post, str]]]:
        """Filter posts by content quality.

        Args:
            posts: List of posts to filter

        Returns:
            Tuple of (passed_posts, rejected_posts_with_reasons)
        """
        passed: list[Post] = []
        rejected: list[tuple[Post, str]] = []

        # Process with concurrency control
        async def assess_one(post: Post) -> tuple[Post, bool, str]:
            async with self.semaphore:
                is_valid, reason = await self._assess_post(post)
                return post, is_valid, reason

        tasks = [assess_one(post) for post in posts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # On error, be permissive - keep the post
                if self.verbose:
                    console.print(f"[yellow]Assessment error: {result}[/yellow]")
                # Add the original post to passed (permissive on error)
                passed.append(posts[i])
                continue

            post, is_valid, reason = result
            if is_valid:
                passed.append(post)
            else:
                rejected.append((post, reason))
                if self.verbose:
                    console.print(
                        f"[dim]  ✗ {post.url}: {reason}[/dim]"
                    )

        return passed, rejected

    async def _assess_post(self, post: Post) -> tuple[bool, str]:
        """Assess a single post for content quality.

        Args:
            post: Post to assess

        Returns:
            Tuple of (is_valid, reason)
        """
        # Quick heuristic pre-filter (always run)
        if reason := self._quick_reject(post):
            return False, reason

        # LLM assessment if available (primary)
        if self.model:
            return await self._llm_assess(post)

        # Heuristics fallback
        return self._heuristic_assess(post)

    def _quick_reject(self, post: Post) -> str | None:
        """Fast rejection for obvious junk.

        Args:
            post: Post to check

        Returns:
            Rejection reason if obvious junk, None otherwise
        """
        word_count = len(post.content_text.split())

        # Very short content (< 50 words is almost certainly not an article)
        if word_count < 50:
            return f"Too short ({word_count} words)"

        # Note: We don't try to detect error pages by content patterns because
        # it's too error-prone. The crawler already filters failed requests
        # via result.success check. Real error pages typically have very
        # short content anyway and get caught by the word count check.

        return None

    async def _llm_assess(self, post: Post) -> tuple[bool, str]:
        """LLM-based content assessment.

        Args:
            post: Post to assess

        Returns:
            Tuple of (is_valid, reason)
        """
        prompt = self._build_assessment_prompt(post)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            content = response.choices[0].message.content

            if not content:
                # Empty response - fall back to heuristics
                return self._heuristic_assess(post)

            return self._parse_llm_response(content)

        except Exception as e:
            if self.verbose:
                console.print(
                    f"[yellow]LLM assessment failed for {post.slug}: {e}[/yellow]"
                )
            # Be permissive on LLM error - pass the post through
            # (it already passed quick_reject, so it's likely valid content)
            return True, f"LLM error (permissive pass): {type(e).__name__}"

    def _build_assessment_prompt(self, post: Post) -> str:
        """Build LLM prompt for content assessment.

        Args:
            post: Post to assess

        Returns:
            Prompt string
        """
        excerpt = post.content_text[:500] if post.content_text else ""

        return f"""Assess if this webpage contains actual readable content (article, blog post, tutorial, essay) or is junk (error page, login wall, link list, stub, popup, placeholder).

URL: {post.url}
Title: {post.metadata.title}
Content excerpt: {excerpt}

Return ONLY valid JSON:
{{"is_content": true, "reason": "brief explanation"}}
or
{{"is_content": false, "reason": "brief explanation"}}

Guidelines:
- Articles, blog posts, tutorials, essays with paragraphs = CONTENT (is_content: true)
- Error pages, login walls, paywalls, link lists, stubs, empty pages = JUNK (is_content: false)
- Pages that are mostly navigation or ads = JUNK"""

    def _parse_llm_response(self, content: str) -> tuple[bool, str]:
        """Parse LLM assessment response.

        Args:
            content: LLM response content

        Returns:
            Tuple of (is_valid, reason)
        """
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines if not line.startswith("```")
            )

        try:
            data = json.loads(content)
            is_content = data.get("is_content", True)
            reason = data.get("reason", "LLM assessment")
            return is_content, f"LLM: {reason}"

        except json.JSONDecodeError:
            # Fallback parsing
            if "false" in content.lower():
                return False, "LLM: classified as junk"
            return True, "LLM: classified as content"

    def _heuristic_assess(self, post: Post) -> tuple[bool, str]:
        """Heuristic-based content assessment (fallback when no LLM).

        Args:
            post: Post to assess

        Returns:
            Tuple of (is_valid, reason)
        """
        config = self.config
        content = post.content_text

        # Word count check
        word_count = len(content.split())
        if word_count < config.min_word_count:
            return (
                False,
                f"Too short ({word_count} < {config.min_word_count} words)",
            )

        # Paragraph structure check
        # Count sentences that start new paragraphs (period followed by capital letter)
        paragraphs = len(re.findall(r"\.\s+[A-Z]", content))
        if paragraphs < config.min_paragraphs:
            return (
                False,
                f"Too few paragraphs ({paragraphs} < {config.min_paragraphs})",
            )

        # Note: We skip text density check because modern HTML (React, Medium, etc.)
        # has very bloated markup that makes text/HTML ratio unreliable.
        # Medium pages have ~15% density which is perfectly valid content.

        # Link density check
        # Count markdown links [text](url) and raw URLs
        link_count = len(re.findall(r"\[.*?\]\(.*?\)", content))
        link_count += len(re.findall(r"https?://\S+", content))
        words = word_count
        if words > 0:
            link_ratio = link_count / words
            if link_ratio > config.max_link_ratio:
                return (
                    False,
                    f"Too many links ({link_ratio:.2f} > {config.max_link_ratio})",
                )

        return True, "Passed heuristic checks"
