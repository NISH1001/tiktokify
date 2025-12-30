"""Async crawler for Jekyll blogs using crawl4ai."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from rich.console import Console

from tiktokify.models import Post, PostMetadata

if TYPE_CHECKING:
    from tiktokify.filters import URLFilter

console = Console()


class SpiderCrawler:
    """Async spider crawler for any website with recursive link discovery."""

    def __init__(
        self,
        base_url: str,
        max_concurrent: int = 5,
        max_depth: int = 1,
        verbose: bool = False,
        url_filter: URLFilter | None = None,
        stealth: bool = True,
        headless: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.verbose = verbose
        self.url_filter = url_filter
        self.stealth = stealth
        self.headless = headless
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.base_domain = urlparse(self.base_url).netloc

    async def crawl(self) -> list[Post]:
        """Main entry point - crawls entire blog and returns posts."""
        if self.stealth:
            # Stealth mode: anti-detection settings for protected sites like Medium
            browser_config = BrowserConfig(
                headless=self.headless,
                verbose=self.verbose,
                user_agent_mode="random",
                extra_args=[
                    "--disable-blink-features=AutomationControlled",
                ],
            )
        else:
            browser_config = BrowserConfig(headless=self.headless, verbose=self.verbose)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Step 1: Discover post URLs
            if self.verbose:
                console.print("[dim]Discovering post URLs...[/dim]")

            post_urls = await self._discover_post_urls(crawler)

            if self.verbose:
                console.print(f"[green]Found {len(post_urls)} posts[/green]")

            # Step 2: Crawl individual posts concurrently
            posts = await self._crawl_posts(crawler, post_urls)

        return posts

    async def _discover_post_urls(self, crawler: AsyncWebCrawler) -> list[str]:
        """Discover all content URLs using spider-style recursive crawling.

        Starts from base URL and follows internal links up to max_depth levels.
        - Depth 1: Only links from seed URL (default)
        - Depth 2: Links from seed + links from those pages
        - etc.
        """
        discovered: set[str] = set()
        visited: set[str] = set()

        async def crawl_page(url: str, depth: int) -> set[str]:
            """Crawl a single page and return new URLs found."""
            if depth > self.max_depth or url in visited:
                return set()

            visited.add(url)
            new_urls: set[str] = set()

            try:
                async with self.semaphore:
                    result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(wait_until="domcontentloaded"),
                    )

                if not result.success:
                    return set()

                # Extract links from crawl4ai
                if result.links:
                    for link in result.links.get("internal", []):
                        href = link.get("href", "") if isinstance(link, dict) else str(link)
                        if self._is_content_url(href, self.base_domain):
                            full_url = href if href.startswith("http") else urljoin(url, href)
                            if full_url not in discovered:
                                new_urls.add(full_url)

                # Also parse HTML directly as fallback
                if result.html:
                    hrefs = re.findall(r'href=["\']([^"\']+)["\']', result.html)
                    for href in hrefs:
                        if self._is_content_url(href, self.base_domain):
                            full_url = href if href.startswith("http") else urljoin(url, href)
                            if full_url not in discovered:
                                new_urls.add(full_url)

                discovered.update(new_urls)

                if self.verbose and new_urls:
                    console.print(f"[dim]Depth {depth}: Found {len(new_urls)} URLs from {url}[/dim]")

            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]Warning: Failed to crawl {url}: {e}[/yellow]")

            return new_urls

        # Start with seed URL
        if self.verbose:
            console.print(f"[dim]Spider crawling with max_depth={self.max_depth}[/dim]")

        # Depth 1: crawl seed URL
        current_urls = await crawl_page(self.base_url, 1)

        # Deeper levels: recursively crawl discovered URLs
        for depth in range(2, self.max_depth + 1):
            if not current_urls:
                break

            if self.verbose:
                console.print(f"[dim]Crawling depth {depth}: {len(current_urls)} URLs to explore[/dim]")

            # Crawl all current URLs in parallel
            tasks = [crawl_page(url, depth) for url in current_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect new URLs for next depth
            next_urls: set[str] = set()
            for result in results:
                if isinstance(result, set):
                    next_urls.update(result)

            current_urls = next_urls

        if self.verbose:
            console.print(f"[dim]Total discovered: {len(discovered)} content URLs[/dim]")

        # Apply URL filter if provided
        discovered_list = list(discovered)
        if self.url_filter:
            passed, rejected = await self.url_filter.filter(discovered_list)
            if self.verbose and rejected:
                console.print(
                    f"[dim]URL filter: kept {len(passed)}, rejected {len(rejected)}[/dim]"
                )
                for url, reason in rejected[:5]:  # Show first 5
                    console.print(f"[dim]  ✗ {url}: {reason}[/dim]")
                if len(rejected) > 5:
                    console.print(f"[dim]  ... and {len(rejected) - 5} more[/dim]")
            return passed

        return discovered_list

    def _is_content_url(self, href: str, base_domain: str) -> bool:
        """Check if URL is internal content (not static asset or utility page).

        This is a simple filter - accepts anything that's:
        1. On the same domain
        2. Not a static asset (css, js, images, fonts)
        3. Not a utility link (mailto, javascript, anchor)
        """
        if not href:
            return False

        # Skip anchors, mailto, javascript
        if href.startswith(("#", "mailto:", "javascript:", "tel:")):
            return False

        # Skip static assets
        static_extensions = (
            ".css", ".js", ".json", ".xml", ".rss", ".atom",
            ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
            ".woff", ".woff2", ".ttf", ".eot", ".otf",
            ".pdf", ".zip", ".tar", ".gz",
            ".mp3", ".mp4", ".webm", ".ogg",
        )
        if any(href.lower().endswith(ext) for ext in static_extensions):
            return False

        # Check if it's an external link
        if href.startswith(("http://", "https://")):
            parsed = urlparse(href)
            if parsed.netloc != base_domain:
                return False

        # Skip the base URL itself (index page)
        path = urlparse(href).path if href.startswith("http") else href
        if path in ("", "/", "/index.html", "/index.htm"):
            return False

        return True

    async def _crawl_posts(
        self, crawler: AsyncWebCrawler, urls: list[str]
    ) -> list[Post]:
        """Crawl all post URLs concurrently with semaphore."""

        async def crawl_one(url: str) -> Post | None:
            async with self.semaphore:
                return await self._crawl_single_post(crawler, url)

        tasks = [crawl_one(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        posts = []
        for i, result in enumerate(results):
            if isinstance(result, Post):
                posts.append(result)
            elif isinstance(result, Exception) and self.verbose:
                console.print(f"[yellow]Failed to crawl {urls[i]}: {result}[/yellow]")

        return posts

    async def _crawl_single_post(
        self, crawler: AsyncWebCrawler, url: str
    ) -> Post | None:
        """Crawl and parse a single post."""
        try:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(wait_until="domcontentloaded"),
            )

            if not result.success:
                return None

            # Extract metadata from HTML
            metadata = self._extract_metadata(result.html, url)

            # Use markdown for clean text (TF-IDF)
            content_text = result.markdown or ""

            # Calculate reading time (~200 words/min)
            word_count = len(content_text.split())
            reading_time = max(1, word_count // 200)

            # Extract slug from URL
            slug = self._extract_slug(url)

            return Post(
                url=url,
                slug=slug,
                metadata=metadata,
                content_text=content_text,
                content_html=result.html or "",
                reading_time_minutes=reading_time,
            )
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Error parsing {url}: {e}[/yellow]")
            return None

    def _extract_metadata(self, html: str, url: str) -> PostMetadata:
        """Extract metadata from rendered HTML using regex (works with various blog themes)."""
        # Try multiple patterns for title
        title = "Untitled"
        title_patterns = [
            # Jekyll Clean Blog theme
            r'<h1[^>]*class="[^"]*post-title[^"]*"[^>]*>([^<]+)</h1>',
            # WordPress/common patterns
            r'<h1[^>]*class="[^"]*entry-title[^"]*"[^>]*>([^<]+)</h1>',
            r'<h1[^>]*class="[^"]*article-title[^"]*"[^>]*>([^<]+)</h1>',
            r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</h1>',
            # Meta og:title
            r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"',
            r'<meta[^>]*name="title"[^>]*content="([^"]+)"',
            # Generic h1
            r"<h1[^>]*>([^<]+)</h1>",
            # Title tag fallback
            r"<title>([^<|]+)",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
                if title:
                    break

        # Try multiple patterns for date
        date = datetime.now()
        date_patterns = [
            # Various date formats
            (r"Posted on (\w+ \d+, \d{4})", "%B %d, %Y"),
            (r'datetime="(\d{4}-\d{2}-\d{2})', "%Y-%m-%d"),
            (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
            (r"(\w+ \d{1,2}, \d{4})", "%B %d, %Y"),
            (r"(\d{1,2} \w+ \d{4})", "%d %B %Y"),
            (r'<time[^>]*>([^<]+)</time>', None),  # Will try multiple formats
        ]
        for pattern, fmt in date_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                if fmt:
                    try:
                        date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # Try common formats
                    for try_fmt in ["%B %d, %Y", "%Y-%m-%d", "%d %B %Y", "%b %d, %Y"]:
                        try:
                            date = datetime.strptime(date_str, try_fmt)
                            break
                        except ValueError:
                            continue

        # Extract date from URL if not found in HTML
        if date == datetime.now():
            url_date = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", url)
            if url_date:
                try:
                    date = datetime(int(url_date.group(1)), int(url_date.group(2)), int(url_date.group(3)))
                except ValueError:
                    pass

        # Tags from various patterns
        tags = []
        tag_patterns = [
            r'<span[^>]*class="[^"]*badge[^"]*"[^>]*>([^<]+)</span>',
            r'<a[^>]*class="[^"]*tag[^"]*"[^>]*>([^<]+)</a>',
            r'rel="tag"[^>]*>([^<]+)</a>',
            r'<span[^>]*class="[^"]*tag[^"]*"[^>]*>([^<]+)</span>',
        ]
        for pattern in tag_patterns:
            found = re.findall(pattern, html, re.IGNORECASE)
            tags.extend([t.strip() for t in found if t.strip()])
        tags = list(set(tags))[:10]  # Dedupe and limit

        # Category from URL
        path = urlparse(url).path
        parts = [p for p in path.strip("/").split("/") if p and not re.match(r"^\d+$", p)]
        # Skip date-like parts and get first meaningful segment
        categories = []
        for part in parts[:-1]:  # Exclude last part (the slug)
            if not re.match(r"^20\d{2}$", part) and part not in ["blog", "posts", "articles"]:
                categories.append(part)
                break

        # Header image from various patterns
        header_img = None
        img_patterns = [
            r'class="[^"]*intro-header[^"]*"[^>]*style="[^"]*url\([\'"]?([^\'")\s]+)',
            r'class="[^"]*featured[^"]*"[^>]*src="([^"]+)"',
            r'<meta[^>]*property="og:image"[^>]*content="([^"]+)"',
            r'class="[^"]*post-image[^"]*"[^>]*src="([^"]+)"',
            r'class="[^"]*hero[^"]*"[^>]*src="([^"]+)"',
        ]
        for pattern in img_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                header_img = match.group(1)
                break

        # Subtitle/description from various patterns
        subtitle = None
        subtitle_patterns = [
            r'<span[^>]*class="[^"]*subheading[^"]*"[^>]*>([^<]+)</span>',
            r'<p[^>]*class="[^"]*subtitle[^"]*"[^>]*>([^<]+)</p>',
            r'<meta[^>]*name="description"[^>]*content="([^"]+)"',
            r'<meta[^>]*property="og:description"[^>]*content="([^"]+)"',
        ]
        for pattern in subtitle_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                subtitle = match.group(1).strip()
                if len(subtitle) > 200:
                    subtitle = subtitle[:197] + "..."
                break

        return PostMetadata(
            title=title,
            date=date,
            categories=categories,
            tags=tags,
            subtitle=subtitle,
            header_img=header_img,
        )

    def _extract_slug(self, url: str) -> str:
        """Extract slug from post URL."""
        path = urlparse(url).path
        # Remove trailing slash and get last meaningful part
        path = path.rstrip("/")
        if not path:
            return "index"
        slug = path.rsplit("/", 1)[-1]
        # Remove common extensions
        for ext in (".html", ".htm", ".php", ".aspx"):
            if slug.endswith(ext):
                slug = slug[:-len(ext)]
                break
        return slug or "page"
