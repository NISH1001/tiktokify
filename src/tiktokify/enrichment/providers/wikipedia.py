"""Wikipedia content provider."""

import asyncio
from urllib.parse import unquote, urlparse

from curl_cffi import requests as curl_requests

from tiktokify.enrichment.base import ContentProvider, ExternalContent
from tiktokify.models import Post


class WikipediaProvider(ContentProvider):
    """Fetch relevant Wikipedia articles for blog posts.

    Uses Wikipedia REST API to fetch article summaries.
    Requires LLM to first suggest relevant articles (see PostEnricher).
    """

    @property
    def source_type(self) -> str:
        return "wikipedia"

    async def fetch_for_post(self, post: Post) -> list[ExternalContent]:
        """Fetch Wikipedia extracts for pre-suggested articles."""
        results = []

        for suggestion in post.wikipedia_suggestions[: self.max_items]:
            title = self._extract_title_from_url(str(suggestion.url)) or suggestion.title
            extract = await self._fetch_extract(title)

            results.append(
                ExternalContent(
                    source=self.source_type,
                    title=suggestion.title,
                    url=suggestion.url,
                    description=extract,
                    relevance=suggestion.relevance,
                    metadata={"extract": extract},
                )
            )

        return results

    async def _fetch_extract(self, title: str, max_chars: int = 1500) -> str:
        """Fetch article extract from Wikipedia API using curl_cffi.

        Uses curl_cffi with browser impersonation to bypass Wikipedia's
        bot detection which blocks standard Python HTTP clients.
        """
        title = title.strip()
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")

        try:
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: curl_requests.get(url, impersonate="chrome", timeout=10),
            )

            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "")
                if len(extract) > max_chars:
                    extract = extract[:max_chars].rsplit(" ", 1)[0] + "..."
                return extract
        except Exception:
            pass

        return ""

    def _extract_title_from_url(self, url: str) -> str:
        """Extract Wikipedia article title from URL."""
        parsed = urlparse(url)
        if "wikipedia.org" in parsed.netloc:
            path = parsed.path
            if path.startswith("/wiki/"):
                title = path[6:]
                title = unquote(title)
                title = title.replace("_", " ")
                return title
        return ""
