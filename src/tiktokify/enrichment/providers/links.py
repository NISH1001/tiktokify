"""Link extractor provider for crawling external links from blog posts."""

import asyncio
import re
from urllib.parse import urljoin, urlparse

import httpx

from tiktokify.enrichment.base import ContentProvider, ExternalContent
from tiktokify.models import Post


async def fetch_link_metadata(url: str, max_excerpt_chars: int = 600) -> tuple[str, str]:
    """Fetch title and excerpt from a URL.

    Returns (title, excerpt) tuple.
    """
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
                return "", ""

            html = response.text

            # Extract title
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = ""
            if title_match:
                title = re.sub(r"<[^>]+>", "", title_match.group(1))
                title = re.sub(r"\s+", " ", title).strip()

            # Try meta description first
            meta_desc = re.search(
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
                html,
                re.IGNORECASE,
            )
            if meta_desc:
                excerpt = meta_desc.group(1).strip()
                return title, excerpt

            # Remove script, style, nav, header, footer tags
            clean_html = re.sub(
                r"<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Extract text from paragraph tags
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", clean_html, flags=re.DOTALL | re.IGNORECASE)

            text_parts = []
            for p in paragraphs:
                clean = re.sub(r"<[^>]+>", " ", p)
                clean = re.sub(r"\s+", " ", clean).strip()
                if len(clean) > 50:
                    text_parts.append(clean)

            excerpt = " ".join(text_parts)
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[:max_excerpt_chars].rsplit(" ", 1)[0] + "..."

            return title, excerpt

    except Exception:
        return "", ""


class LinkedContentProvider(ContentProvider):
    """Extract and crawl external links from blog post content.

    Finds links within the blog post HTML and fetches their content,
    creating a "spider" of related content from the post's references.
    """

    # Domains to skip (social media, generic sites, etc.)
    SKIP_DOMAINS = {
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "youtu.be",
        "github.com",
        "gist.github.com",
        "reddit.com",
        "news.ycombinator.com",
        "google.com",
        "amazon.com",
        "wikipedia.org",  # Already have Wikipedia provider
        "fonts.googleapis.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cloudflare.com",
    }

    @property
    def source_type(self) -> str:
        return "linked"

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract external links from HTML content."""
        # Find all href links
        links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)

        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc.lower()

        external_links = []
        seen = set()

        for link in links:
            # Skip anchor links, mailto, javascript, etc.
            if link.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue

            # Resolve relative URLs
            if link.startswith("/"):
                link = urljoin(base_url, link)
            elif not link.startswith(("http://", "https://")):
                continue

            # Parse and validate
            parsed = urlparse(link)
            domain = parsed.netloc.lower()

            # Skip internal links
            if domain == base_domain or domain.endswith(f".{base_domain}"):
                continue

            # Skip blocked domains
            if any(skip in domain for skip in self.SKIP_DOMAINS):
                continue

            # Skip non-http(s) links
            if parsed.scheme not in ("http", "https"):
                continue

            # Skip duplicates
            normalized = f"{parsed.scheme}://{domain}{parsed.path}"
            if normalized in seen:
                continue
            seen.add(normalized)

            external_links.append(link)

        return external_links

    async def fetch_for_post(self, post: Post) -> list[ExternalContent]:
        """Extract links from post content and fetch their metadata."""
        if not post.content_html:
            return []

        # Extract external links
        links = self._extract_links(post.content_html, post.url)

        if not links:
            return []

        # Limit to max_items and fetch all in parallel
        selected_links = links[: self.max_items]

        metadata_results = await asyncio.gather(
            *[fetch_link_metadata(link) for link in selected_links],
            return_exceptions=True,
        )

        results = []
        for link, meta in zip(selected_links, metadata_results):
            # Handle exceptions
            if isinstance(meta, Exception):
                continue

            title, excerpt = meta

            if not title and not excerpt:
                continue

            # Use URL domain as fallback title
            if not title:
                parsed = urlparse(link)
                title = parsed.netloc

            results.append(
                ExternalContent(
                    source=self.source_type,
                    title=title,
                    url=link,
                    description=excerpt,
                    relevance=f"Referenced in: {post.metadata.title}",
                    metadata={
                        "source_post_slug": post.slug,
                        "link_type": "reference",
                    },
                )
            )

        return results
