"""Wikipedia content provider."""

from urllib.parse import unquote, urlparse

import httpx

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
        """Fetch article extract from Wikipedia API."""
        title = title.strip()
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                    },
                    timeout=10.0,
                    follow_redirects=True,
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
