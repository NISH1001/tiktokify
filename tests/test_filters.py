"""Tests for pipeline filters."""

import pytest

from tiktokify.filters import ContentFilter, ContentFilterConfig, URLFilter
from tiktokify.models import Post, PostMetadata


class TestURLFilter:
    """Tests for URLFilter pre-crawl filtering."""

    @pytest.fixture
    def url_filter(self) -> URLFilter:
        return URLFilter()

    @pytest.mark.asyncio
    async def test_filters_auth_urls(self, url_filter: URLFilter):
        """Should filter out authentication URLs."""
        urls = [
            "https://example.com/article",
            "https://example.com/signup",
            "https://example.com/login",
            "https://example.com/auth/callback",
            "https://example.com/register",
        ]
        passed, rejected = await url_filter.filter(urls)

        assert len(passed) == 1
        assert passed[0] == "https://example.com/article"
        assert len(rejected) == 4

    @pytest.mark.asyncio
    async def test_filters_social_urls(self, url_filter: URLFilter):
        """Should filter out social action URLs."""
        urls = [
            "https://example.com/post/123",
            "https://example.com/share",
            "https://example.com/like",
            "https://example.com/follow",
        ]
        passed, rejected = await url_filter.filter(urls)

        assert len(passed) == 1
        assert passed[0] == "https://example.com/post/123"

    @pytest.mark.asyncio
    async def test_filters_feed_urls(self, url_filter: URLFilter):
        """Should filter out feed/utility URLs."""
        urls = [
            "https://example.com/blog/post",
            "https://example.com/feed",
            "https://example.com/rss",
            "https://example.com/api/data",
        ]
        passed, rejected = await url_filter.filter(urls)

        assert len(passed) == 1
        assert passed[0] == "https://example.com/blog/post"

    @pytest.mark.asyncio
    async def test_strips_tracking_params(self, url_filter: URLFilter):
        """Should strip tracking parameters instead of rejecting URLs."""
        urls = [
            "https://example.com/article",
            "https://example.com/page?utm_source=twitter",
            "https://example.com/other?ref=homepage",
            "https://example.com/post?source=user_profile&id=123",
        ]
        passed, rejected = await url_filter.filter(urls)

        # All should pass, but with tracking params stripped
        assert len(passed) == 4
        assert "https://example.com/article" in passed
        assert "https://example.com/page" in passed  # utm_source stripped
        assert "https://example.com/other" in passed  # ref stripped
        assert "https://example.com/post?id=123" in passed  # source stripped, id kept
        assert len(rejected) == 0

    @pytest.mark.asyncio
    async def test_deduplicates_after_stripping(self, url_filter: URLFilter):
        """Should deduplicate URLs that become identical after param stripping."""
        urls = [
            "https://example.com/page?utm_source=twitter",
            "https://example.com/page?utm_source=facebook",
            "https://example.com/page?ref=homepage",
        ]
        passed, rejected = await url_filter.filter(urls)

        # All become https://example.com/page - should only keep one
        assert len(passed) == 1
        assert passed[0] == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_passes_content_urls(self, url_filter: URLFilter):
        """Should pass through legitimate content URLs."""
        urls = [
            "https://example.com/blog/my-article",
            "https://example.com/posts/2024/01/hello",
            "https://example.com/tutorials/python",
            "https://medium.com/@user/my-story-abc123",
        ]
        passed, rejected = await url_filter.filter(urls)

        assert len(passed) == 4
        assert len(rejected) == 0

    @pytest.mark.asyncio
    async def test_extra_patterns(self):
        """Should support custom patterns."""
        url_filter = URLFilter(extra_patterns=[r"/custom-junk/"])
        urls = [
            "https://example.com/article",
            "https://example.com/custom-junk/page",
        ]
        passed, rejected = await url_filter.filter(urls)

        assert len(passed) == 1
        assert passed[0] == "https://example.com/article"


class TestContentFilter:
    """Tests for ContentFilter post-crawl filtering."""

    def _make_post(
        self,
        content: str,
        title: str = "Test Post",
        html: str = "",
    ) -> Post:
        """Helper to create test posts."""
        from datetime import datetime

        return Post(
            url="https://example.com/test",
            slug="test",
            metadata=PostMetadata(title=title, date=datetime.now()),
            content_text=content,
            content_html=html or f"<html><body>{content}</body></html>",
        )

    @pytest.fixture
    def content_filter(self) -> ContentFilter:
        return ContentFilter(
            config=ContentFilterConfig(min_word_count=100, min_paragraphs=3)
        )

    @pytest.mark.asyncio
    async def test_rejects_short_content(self, content_filter: ContentFilter):
        """Should reject posts with too few words."""
        short_post = self._make_post("Just a few words here.")
        passed, rejected = await content_filter.filter([short_post])

        assert len(passed) == 0
        assert len(rejected) == 1
        assert "Too short" in rejected[0][1]

    @pytest.mark.asyncio
    async def test_rejects_very_short_error_pages(self, content_filter: ContentFilter):
        """Should reject very short pages (like error pages) via word count."""
        # Error pages typically have very few words
        error_post = self._make_post(
            "404 - Page Not Found. The page you're looking for doesn't exist."
        )
        passed, rejected = await content_filter.filter([error_post])

        assert len(passed) == 0
        assert len(rejected) == 1
        assert "Too short" in rejected[0][1]

    @pytest.mark.asyncio
    async def test_rejects_short_paywall_pages(self, content_filter: ContentFilter):
        """Should reject short paywall pages via word count."""
        # Paywall stub pages are typically very short
        paywall_post = self._make_post(
            "Sign in to read this article. Subscribe now."
        )
        passed, rejected = await content_filter.filter([paywall_post])

        assert len(passed) == 0
        assert len(rejected) == 1
        assert "Too short" in rejected[0][1]

    @pytest.mark.asyncio
    async def test_passes_quality_content(self, content_filter: ContentFilter):
        """Should pass posts with quality content."""
        # Generate content with 100+ words and paragraph breaks (. followed by capital)
        # Each paragraph needs to end with period and next starts with capital
        content = """This is the first paragraph of the article about technology and programming. It contains several sentences about an interesting topic that we want to explore in great detail throughout this comprehensive piece.

The second paragraph continues the discussion with more depth and analysis. Here we explore more details and provide concrete examples of the concepts that were introduced earlier in the article.

In the third paragraph, we conclude our main thoughts and summarize key findings. This wraps up the key points and provides actionable takeaways for readers who want to learn more about this subject.

Additional content follows here with more supporting words and explanations. We need to ensure there are enough words in total to pass the minimum threshold for quality content filtering that is set to one hundred words."""

        good_post = self._make_post(content)
        passed, rejected = await content_filter.filter([good_post])

        # If it failed, let's see why
        if rejected:
            print(f"Rejected: {rejected[0][1]}")
            print(f"Word count: {len(content.split())}")

        assert len(passed) == 1
        assert len(rejected) == 0

    @pytest.mark.asyncio
    async def test_heuristic_paragraph_check(self):
        """Should check for paragraph structure when no LLM."""
        content_filter = ContentFilter(
            config=ContentFilterConfig(min_word_count=50, min_paragraphs=3)
        )

        # Content with enough words but no paragraph structure
        no_paragraphs = self._make_post("word " * 100)
        passed, rejected = await content_filter.filter([no_paragraphs])

        assert len(passed) == 0
        assert "paragraphs" in rejected[0][1].lower()

    @pytest.mark.asyncio
    async def test_configurable_thresholds(self):
        """Should respect custom thresholds."""
        strict_filter = ContentFilter(
            config=ContentFilterConfig(min_word_count=500, min_paragraphs=1)
        )
        lenient_filter = ContentFilter(
            config=ContentFilterConfig(min_word_count=10, min_paragraphs=1)
        )

        # Content with 50+ words (passes quick_reject) but <500 words (fails strict)
        # Also has paragraph structure (period + space + capital)
        content = """This is a medium-length article that contains enough words to pass the quick rejection filter. It has multiple sentences with proper structure.

The content continues here with additional paragraphs and more words. We need to ensure there are enough words to pass the fifty word minimum threshold that is hardcoded in the quick reject function.

This third paragraph adds even more content to make sure we have sufficient word count for testing purposes."""
        post = self._make_post(content)

        strict_passed, strict_rejected = await strict_filter.filter([post])
        lenient_passed, lenient_rejected = await lenient_filter.filter([post])

        # Debug output
        if lenient_rejected:
            print(f"Lenient rejected: {lenient_rejected[0][1]}")

        assert len(strict_passed) == 0  # Too short for strict (needs 500)
        assert len(lenient_passed) == 1  # Passes lenient (needs only 10)

    @pytest.mark.asyncio
    async def test_permissive_on_llm_exception(self):
        """Should keep posts when LLM assessment raises exceptions (be permissive)."""
        from unittest.mock import patch, AsyncMock

        content_filter = ContentFilter(
            config=ContentFilterConfig(min_word_count=10, min_paragraphs=1),
            model="gpt-4o-mini",  # Enable LLM mode
        )

        # Create a post with 50+ words to pass quick_reject
        content = """This is a valid article with enough content to pass the quick rejection filter.
        It has proper structure and multiple sentences that provide meaningful information to readers.

        The second paragraph adds more context and detail about the topic at hand.
        We continue with additional sentences to ensure we have well over fifty words total.

        This third paragraph wraps up the content with some final thoughts and conclusions.
        The article should definitely pass all the basic heuristic checks we have in place."""
        post = self._make_post(content)

        # Mock litellm.acompletion to raise an exception (simulating event loop error)
        with patch("tiktokify.filters.content_filter.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=RuntimeError("Event loop error")
            )

            passed, rejected = await content_filter.filter([post])

            # Should be permissive - keep the post even if LLM fails
            assert len(passed) == 1
            assert len(rejected) == 0


class TestRecommendationThreshold:
    """Tests for min_similarity threshold in RecommendationEngine."""

    def _make_posts(self) -> list[Post]:
        """Create test posts with varying content."""
        from datetime import datetime

        return [
            Post(
                url="https://example.com/ml",
                slug="ml-intro",
                metadata=PostMetadata(
                    title="Introduction to Machine Learning",
                    date=datetime.now(),
                    tags=["ml", "ai", "python"],
                ),
                content_text="Machine learning is a subset of AI that focuses on algorithms.",
                content_html="",
            ),
            Post(
                url="https://example.com/dl",
                slug="deep-learning",
                metadata=PostMetadata(
                    title="Deep Learning Fundamentals",
                    date=datetime.now(),
                    tags=["ml", "ai", "neural-networks"],
                ),
                content_text="Deep learning uses neural networks for complex tasks.",
                content_html="",
            ),
            Post(
                url="https://example.com/cooking",
                slug="cooking-tips",
                metadata=PostMetadata(
                    title="Best Cooking Tips",
                    date=datetime.now(),
                    tags=["cooking", "food"],
                ),
                content_text="Here are my favorite cooking techniques for delicious meals.",
                content_html="",
            ),
        ]

    def test_no_threshold_returns_all(self):
        """With threshold=0, should return all top-k results."""
        from tiktokify.recommender import RecommendationEngine

        posts = self._make_posts()
        engine = RecommendationEngine(min_similarity=0.0, top_k=5)
        graph = engine.build_graph(posts)

        # Each post should have recommendations (even if low similarity)
        for post in posts:
            recs = graph.adjacency.get(post.slug, [])
            # Should have 2 recommendations (the other 2 posts)
            assert len(recs) == 2

    def test_high_threshold_filters(self):
        """With high threshold, should filter low-similarity results."""
        from tiktokify.recommender import RecommendationEngine

        posts = self._make_posts()
        # Very high threshold - might filter some recommendations
        engine = RecommendationEngine(min_similarity=0.5, top_k=5)
        graph = engine.build_graph(posts)

        # ML and DL posts might still match due to shared tags
        # But cooking should have fewer matches with ML content
        cooking_recs = graph.adjacency.get("cooking-tips", [])
        # With high threshold, cooking might have fewer or no recommendations
        # to ML/DL content
        for slug, score in cooking_recs:
            assert score >= 0.5  # All should be above threshold
