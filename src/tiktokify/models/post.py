"""Pydantic models for blog posts and recommendation graph."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ExternalContentItem(BaseModel):
    """Generic external content from any source."""

    source: str = Field(description="Source type: 'wikipedia', 'hackernews', 'reddit', etc.")
    title: str
    url: HttpUrl
    description: str = Field(default="", description="Brief description or excerpt")
    relevance: str = Field(default="", description="Why this is relevant to the post")
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")


class WikipediaSuggestion(BaseModel):
    """A Wikipedia article suggestion for a blog post."""

    title: str
    url: HttpUrl
    relevance: str = Field(description="Brief explanation of why this is relevant")
    extract: str = Field(default="", description="Article summary from Wikipedia API")


class PostMetadata(BaseModel):
    """Metadata extracted from Jekyll post front matter."""

    title: str
    date: datetime
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    subtitle: Optional[str] = None
    header_img: Optional[str] = None
    last_edited_on: Optional[datetime] = None


class Post(BaseModel):
    """Complete representation of a blog post."""

    url: str
    slug: str
    metadata: PostMetadata
    content_text: str = Field(description="Plain text content for TF-IDF")
    content_html: str = Field(default="", description="Full HTML content")
    reading_time_minutes: int = Field(default=1)

    # Populated during enrichment phase
    key_points: list[str] = Field(
        default_factory=list, description="LLM-generated key points/summary"
    )
    similar_posts: list[str] = Field(
        default_factory=list, description="List of similar post slugs"
    )
    similarity_scores: dict[str, float] = Field(
        default_factory=dict, description="slug -> similarity score"
    )
    wikipedia_suggestions: list[WikipediaSuggestion] = Field(default_factory=list)
    external_content: list[ExternalContentItem] = Field(
        default_factory=list, description="Content from external sources (HN, Reddit, etc.)"
    )


class RecommendationGraph(BaseModel):
    """Graph of posts with recommendation adjacency list."""

    posts: dict[str, Post] = Field(description="slug -> Post mapping")
    adjacency: dict[str, list[tuple[str, float]]] = Field(
        default_factory=dict, description="slug -> [(similar_slug, score), ...]"
    )

    def to_json_for_embed(self) -> dict:
        """Serialize for embedding in HTML (minimal, frontend-friendly format)."""
        return {
            "posts": {
                slug: {
                    "title": p.metadata.title,
                    "subtitle": p.metadata.subtitle,
                    "date": p.metadata.date.isoformat(),
                    "categories": p.metadata.categories,
                    "tags": p.metadata.tags,
                    "url": p.url,
                    "headerImg": p.metadata.header_img,
                    "readingTime": p.reading_time_minutes,
                    "keyPoints": p.key_points,
                    "wikipedia": [
                        {
                            "title": w.title,
                            "url": str(w.url),
                            "relevance": w.relevance,
                            "extract": w.extract,
                        }
                        for w in p.wikipedia_suggestions
                    ],
                    "externalContent": [
                        {
                            "source": e.source,
                            "title": e.title,
                            "url": str(e.url),
                            "description": e.description,
                            "relevance": e.relevance,
                            "metadata": e.metadata,
                        }
                        for e in p.external_content
                    ],
                }
                for slug, p in self.posts.items()
            },
            "recommendations": {
                slug: [(s, round(score, 3)) for s, score in recs]
                for slug, recs in self.adjacency.items()
            },
        }
