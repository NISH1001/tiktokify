"""Base classes for content providers."""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl

from tiktokify.models import Post


class ExternalContent(BaseModel):
    """A piece of external content from any source."""

    source: str = Field(description="Source type: 'wikipedia', 'hackernews', 'reddit', etc.")
    title: str
    url: HttpUrl
    description: str = Field(default="", description="Brief description or excerpt")
    relevance: str = Field(default="", description="Why this is relevant to the post")
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")


class ContentProvider(ABC):
    """Abstract base class for external content providers.

    To add a new source:
    1. Create a new file (e.g., hackernews.py)
    2. Subclass ContentProvider
    3. Implement source_type and fetch_for_post
    4. Register in enricher.py
    """

    def __init__(self, max_items: int = 3, verbose: bool = False):
        self.max_items = max_items
        self.verbose = verbose

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Unique identifier for this source (e.g., 'wikipedia', 'hackernews')."""
        pass

    @abstractmethod
    async def fetch_for_post(self, post: Post) -> list[ExternalContent]:
        """Fetch relevant external content for a blog post.

        Args:
            post: The blog post to find related content for

        Returns:
            List of ExternalContent items (up to max_items)
        """
        pass

    async def fetch_for_posts(self, posts: list[Post]) -> dict[str, list[ExternalContent]]:
        """Fetch content for multiple posts.

        Default implementation calls fetch_for_post sequentially.
        Override for batch optimization.

        Returns:
            Dict mapping post slug to list of ExternalContent
        """
        results = {}
        for post in posts:
            results[post.slug] = await self.fetch_for_post(post)
        return results
