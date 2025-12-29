"""Hacker News content providers.

Provides two providers:
- HackerNewsProvider: Keyword-based search for stories related to post topics
- HNFrontPageProvider: Current front page stories for general interest
"""

import asyncio
import re

import httpx

from tiktokify.enrichment.base import ContentProvider, ExternalContent
from tiktokify.models import Post


async def fetch_article_excerpt(url: str, max_chars: int = 800) -> str:
    """Fetch and extract text excerpt from an article URL."""
    if not url:
        return ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "TikTokify/1.0 (Mozilla/5.0 compatible)",
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=10.0,
                follow_redirects=True,
            )

            if response.status_code != 200:
                return ""

            html = response.text

            # Remove script, style, nav, header, footer tags
            html = re.sub(r"<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)

            # Extract text from paragraph tags (most content is in <p>)
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.DOTALL | re.IGNORECASE)

            # Clean HTML tags from extracted text
            text_parts = []
            for p in paragraphs:
                clean = re.sub(r"<[^>]+>", " ", p)
                clean = re.sub(r"\s+", " ", clean).strip()
                if len(clean) > 50:  # Skip very short paragraphs
                    text_parts.append(clean)

            if not text_parts:
                # Fallback: extract all text
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:max_chars] + "..." if len(text) > max_chars else text

            excerpt = " ".join(text_parts)
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars].rsplit(" ", 1)[0] + "..."
            return excerpt

    except Exception:
        return ""


class HackerNewsProvider(ContentProvider):
    """Fetch relevant Hacker News discussions for blog posts.

    Uses the Algolia HN Search API to find related stories by keyword.
    """

    HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

    @property
    def source_type(self) -> str:
        return "hackernews"

    async def fetch_for_post(self, post: Post) -> list[ExternalContent]:
        """Search HN for stories related to the post's topics."""
        # Build search query from post metadata
        query_parts = []

        # Use tags (most specific)
        if post.metadata.tags:
            query_parts.extend(post.metadata.tags[:3])

        # Add key terms from title
        title_words = [
            w for w in post.metadata.title.split()
            if len(w) > 4 and w.lower() not in {"about", "using", "with", "from", "that", "this", "what", "when", "where", "which"}
        ]
        query_parts.extend(title_words[:2])

        if not query_parts:
            return []

        query = " ".join(query_parts)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.HN_SEARCH_URL,
                    params={
                        "query": query,
                        "tags": "story",
                        "hitsPerPage": self.max_items * 2,  # Fetch extra for filtering
                    },
                    timeout=10.0,
                )

                if response.status_code != 200:
                    return []

                data = response.json()
                hits = data.get("hits", [])

                # Prepare hits for parallel fetching
                selected_hits = hits[: self.max_items]
                story_urls = [hit.get("url", "") for hit in selected_hits]

                # Fetch all article excerpts in parallel
                excerpts = await asyncio.gather(
                    *[fetch_article_excerpt(url) for url in story_urls],
                    return_exceptions=True,
                )

                results = []
                for hit, excerpt in zip(selected_hits, excerpts):
                    story_id = hit.get("objectID", "")
                    hn_url = f"https://news.ycombinator.com/item?id={story_id}"

                    title = hit.get("title", "")
                    points = hit.get("points", 0)
                    num_comments = hit.get("num_comments", 0)
                    author = hit.get("author", "")
                    story_url = hit.get("url", "")

                    # Handle exceptions from parallel fetch
                    if isinstance(excerpt, Exception) or not excerpt:
                        excerpt = f"{points} points · {num_comments} comments"

                    results.append(
                        ExternalContent(
                            source=self.source_type,
                            title=title,
                            url=hn_url,
                            description=excerpt,
                            relevance=f"Found via search: {query}",
                            metadata={
                                "points": points,
                                "num_comments": num_comments,
                                "author": author,
                                "story_url": story_url,
                            },
                        )
                    )

                return results

        except Exception:
            return []


class HNFrontPageProvider(ContentProvider):
    """Fetch current Hacker News front page stories.

    Uses the Algolia HN API to get stories currently on the front page.
    Good for adding general tech interest content to any blog.
    """

    HN_FRONT_PAGE_URL = "https://hn.algolia.com/api/v1/search"

    @property
    def source_type(self) -> str:
        return "hn-frontpage"

    async def fetch_for_post(self, post: Post) -> list[ExternalContent]:
        """Fetch current front page stories (post-independent)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.HN_FRONT_PAGE_URL,
                    params={
                        "tags": "front_page",
                        "hitsPerPage": self.max_items,
                    },
                    timeout=10.0,
                )

                if response.status_code != 200:
                    return []

                data = response.json()
                hits = data.get("hits", [])

                # Prepare hits for parallel fetching
                selected_hits = hits[: self.max_items]
                story_urls = [hit.get("url", "") for hit in selected_hits]

                # Fetch all article excerpts in parallel
                excerpts = await asyncio.gather(
                    *[fetch_article_excerpt(url) for url in story_urls],
                    return_exceptions=True,
                )

                results = []
                for hit, excerpt in zip(selected_hits, excerpts):
                    story_id = hit.get("objectID", "")
                    hn_url = f"https://news.ycombinator.com/item?id={story_id}"

                    title = hit.get("title", "")
                    points = hit.get("points", 0)
                    num_comments = hit.get("num_comments", 0)
                    author = hit.get("author", "")
                    story_url = hit.get("url", "")

                    # Handle exceptions from parallel fetch
                    if isinstance(excerpt, Exception) or not excerpt:
                        excerpt = f"{points} points · {num_comments} comments"

                    results.append(
                        ExternalContent(
                            source=self.source_type,
                            title=title,
                            url=hn_url,
                            description=excerpt,
                            relevance="Currently on HN front page",
                            metadata={
                                "points": points,
                                "num_comments": num_comments,
                                "author": author,
                                "story_url": story_url,
                            },
                        )
                    )

                return results

        except Exception:
            return []
