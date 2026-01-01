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
        follow_external: bool = False,
        external_depth: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.verbose = verbose
        self.url_filter = url_filter
        self.stealth = stealth
        self.headless = headless
        self.follow_external = follow_external
        # External depth: how deep to crawl external sites (None = same as max_depth)
        self.external_depth = external_depth if external_depth is not None else max_depth
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.base_domain = urlparse(self.base_url).netloc

    async def crawl(self) -> list[Post]:
        """Main entry point - crawls and returns posts in single pass."""
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
            if self.verbose:
                console.print("[dim]Discovering and crawling posts...[/dim]")

            posts = await self._crawl_all(crawler)

            if self.verbose:
                console.print(f"[green]Found {len(posts)} posts[/green]")

        return posts

    def _is_internal_url(self, url: str) -> bool:
        """Check if URL belongs to the base domain."""
        parsed = urlparse(url)
        return parsed.netloc == self.base_domain or not parsed.netloc

    async def _crawl_all(self, crawler: AsyncWebCrawler) -> list[Post]:
        """Single-pass crawl: discover links + build Posts together.

        Depth semantics:
        - depth=0: Only crawl seed URL, don't follow any links
        - depth=1: Seed + 1 level of links
        - depth=2: Seed + 2 levels of links
        """
        posts: dict[str, Post] = {}  # url -> Post (avoid duplicates)
        visited: set[str] = set()

        async def crawl_page(url: str, current_depth: int, is_external: bool) -> list[tuple[str, int, bool]]:
            """Crawl a page and return discovered URLs with their depth info."""
            if url in visited:
                return []

            # Check depth limits
            max_allowed = self.external_depth if is_external else self.max_depth
            if current_depth > max_allowed:
                return []

            visited.add(url)

            try:
                async with self.semaphore:
                    result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(wait_until="domcontentloaded"),
                    )

                if not result.success:
                    return []

                # Build Post immediately (single-pass: no second fetch needed)
                post = self._build_post(result, url)
                if post:
                    posts[url] = post

                # Only extract links if we haven't reached max depth
                if current_depth < max_allowed:
                    new_urls = self._extract_links(result, url, current_depth, visited)
                    if self.verbose and new_urls:
                        ext_info = " (external)" if is_external else ""
                        console.print(f"[dim]Depth {current_depth}{ext_info}: Found {len(new_urls)} links from {url}[/dim]")
                    return new_urls

                return []

            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]Warning: Failed to crawl {url}: {e}[/yellow]")
                return []

        # Start with seed URL
        if self.verbose:
            ext_info = f", external_depth={self.external_depth}" if self.follow_external else ""
            console.print(f"[dim]Spider crawling with max_depth={self.max_depth}{ext_info}[/dim]")

        # Crawl seed URL at depth 0
        current_urls = await crawl_page(self.base_url, 0, False)

        # BFS: crawl discovered URLs level by level
        max_possible_depth = max(self.max_depth, self.external_depth) if self.follow_external else self.max_depth
        for depth in range(1, max_possible_depth + 1):
            if not current_urls:
                break

            if self.verbose:
                console.print(f"[dim]Depth {depth}: {len(current_urls)} URLs to crawl[/dim]")

            # Crawl all current URLs in parallel
            tasks = [crawl_page(url, depth, is_ext) for url, depth, is_ext in current_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect new URLs for next depth
            next_urls: list[tuple[str, int, bool]] = []
            for result in results:
                if isinstance(result, list):
                    next_urls.extend(result)

            current_urls = next_urls

        if self.verbose:
            console.print(f"[dim]Total crawled: {len(posts)} posts[/dim]")

        # Apply URL filter if provided
        post_list = list(posts.values())
        if self.url_filter:
            urls = [p.url for p in post_list]
            passed_urls, rejected = await self.url_filter.filter(urls)
            passed_set = set(passed_urls)
            if self.verbose and rejected:
                console.print(
                    f"[dim]URL filter: kept {len(passed_urls)}, rejected {len(rejected)}[/dim]"
                )
                for url, reason in rejected[:5]:  # Show first 5
                    console.print(f"[dim]  ✗ {url}: {reason}[/dim]")
                if len(rejected) > 5:
                    console.print(f"[dim]  ... and {len(rejected) - 5} more[/dim]")
            post_list = [p for p in post_list if p.url in passed_set]

        return post_list

    def _build_post(self, result, url: str) -> Post | None:
        """Build Post from CrawlResult."""
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

    def _extract_links(
        self, result, url: str, current_depth: int, visited: set[str]
    ) -> list[tuple[str, int, bool]]:
        """Extract links from crawl result for recursive crawling.

        Returns list of (url, next_depth, is_external) tuples.
        """
        new_urls: list[tuple[str, int, bool]] = []
        seen: set[str] = set()

        def add_url(href: str) -> None:
            if not self._is_content_url(href, self.base_domain):
                return
            full_url = href if href.startswith("http") else urljoin(url, href)
            if full_url in visited or full_url in seen:
                return
            seen.add(full_url)
            is_external = not self._is_internal_url(full_url)
            new_urls.append((full_url, current_depth + 1, is_external))

        # From crawl4ai links
        if result.links:
            for link in result.links.get("internal", []):
                href = link.get("href", "") if isinstance(link, dict) else str(link)
                add_url(href)

        # From HTML fallback
        if result.html:
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', result.html)
            for href in hrefs:
                add_url(href)

        return new_urls

    def _is_content_url(self, href: str, base_domain: str) -> bool:
        """Check if URL is content (not static asset or utility page).

        This is a simple filter - accepts anything that's:
        1. On the same domain (or any domain if follow_external=True)
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

        # Check if it's an external link (only filter if not following external)
        if not self.follow_external:
            if href.startswith(("http://", "https://")):
                parsed = urlparse(href)
                if parsed.netloc != base_domain:
                    return False

        # Skip the base URL itself (index page)
        path = urlparse(href).path if href.startswith("http") else href
        if path in ("", "/", "/index.html", "/index.htm"):
            return False

        return True

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
