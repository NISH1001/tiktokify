"""URL filter for pre-crawl filtering of junk URLs."""

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from tiktokify.filters.base import BaseFilter


class URLFilter(BaseFilter[str]):
    """Pre-crawl URL filtering based on patterns.

    Filters out URLs that match known junk patterns before crawling,
    saving resources and reducing noise in the pipeline.
    """

    # Default junk patterns (generic, not platform-specific)
    # NOTE: These are checked AFTER stripping tracking params
    DEFAULT_PATTERNS: list[str] = [
        # Auth pages
        r"/sign(up|in|out)/?$",
        r"/log(in|out)/?$",
        r"/register/?$",
        r"/auth/",
        r"/oauth/",
        r"/sso/",
        r"/callback/?$",

        # Social actions
        r"/share/?$",
        r"/like/?$",
        r"/follow/?$",
        r"/subscribe/?$",
        r"/unsubscribe/?$",
        r"/clap/?$",

        # User profile sub-pages (followers/following pages)
        r"/@[\w-]+/(followers|following|claps|highlights)/?$",
        r"/u(ser)?/[\w-]+/(followers|following)/?$",
        r"/profile/[\w-]+/(followers|following)/?$",
        r"/followers/?$",
        r"/following/?$",

        # Responses/comments pages
        r"/responses/?$",
        r"/comments/?$",

        # Feed/utility pages
        r"/feed/?$",
        r"/rss/?$",
        r"/atom/?$",
        r"/api/",
        r"/webhook/",
        r"/_next/",
        r"/__",

        # Common non-content pages
        r"/search/?$",
        r"/explore/?$",
        r"/trending/?$",
        r"/popular/?$",
        r"/latest/?$",
        r"/notifications/?$",
        r"/settings/?$",
        r"/preferences/?$",
        r"/account/?$",
        r"/billing/?$",
        r"/pricing/?$",
        r"/plans/?$",
        r"/upgrade/?$",
        r"/premium/?$",
        r"/pro/?$",

        # Newsletter/email
        r"/newsletter/?$",
        r"/email-preferences/?$",

        # Legal pages
        r"/privacy(-policy)?/?$",
        r"/terms(-of-service|-of-use)?/?$",
        r"/cookie(-policy)?/?$",
        r"/gdpr/?$",
        r"/legal/?$",
        r"/tos/?$",

        # Help/support
        r"/help/?$",
        r"/support/?$",
        r"/faq/?$",
        r"/contact/?$",
    ]

    # Query params to strip (tracking/analytics)
    TRACKING_PARAMS: set[str] = {
        # UTM params
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        # Common tracking
        "ref", "source", "campaign", "mc_cid", "mc_eid",
        # Social tracking
        "fbclid", "gclid", "msclkid", "twclid",
        # Medium-specific
        "sk",
    }

    def __init__(self, extra_patterns: list[str] | None = None):
        """Initialize URL filter.

        Args:
            extra_patterns: Additional regex patterns to filter out
        """
        self.patterns = self.DEFAULT_PATTERNS + (extra_patterns or [])
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    @property
    def name(self) -> str:
        return "URLFilter"

    async def filter(self, urls: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Filter URLs before crawling.

        Args:
            urls: List of URLs to filter

        Returns:
            Tuple of (passed_urls, rejected_urls_with_reasons)
        """
        passed: list[str] = []
        rejected: list[tuple[str, str]] = []
        seen: set[str] = set()  # Dedup after stripping params

        for url in urls:
            # Skip non-HTTP URLs (android-app://, ios-app://, mailto:, etc.)
            if not url.startswith(("http://", "https://")):
                rejected.append((url, "Not an HTTP URL"))
                continue

            # Strip tracking params first
            clean_url = self._strip_tracking_params(url)

            # Skip duplicates (same URL after param stripping)
            if clean_url in seen:
                continue
            seen.add(clean_url)

            if reason := self._matches_junk(clean_url):
                rejected.append((url, reason))
            else:
                passed.append(clean_url)

        return passed, rejected

    def _strip_tracking_params(self, url: str) -> str:
        """Strip tracking/analytics query parameters from URL.

        Args:
            url: URL to clean

        Returns:
            URL with tracking params removed
        """
        try:
            parsed = urlparse(url)
            if not parsed.query:
                return url

            # Parse query string and filter out tracking params
            params = parse_qs(parsed.query, keep_blank_values=True)
            clean_params = {
                k: v for k, v in params.items()
                if k.lower() not in self.TRACKING_PARAMS
                and not k.lower().startswith("utm_")
                and not k.lower().startswith("mc_")
            }

            # Rebuild URL
            clean_query = urlencode(clean_params, doseq=True) if clean_params else ""
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                "",  # Remove fragment too
            ))
        except Exception:
            return url

    def _matches_junk(self, url: str) -> str | None:
        """Check if URL matches any junk pattern.

        Args:
            url: URL to check (should already have tracking params stripped)

        Returns:
            Rejection reason if matches, None otherwise
        """
        for pattern in self._compiled:
            if pattern.search(url):
                return f"Matches junk pattern: {pattern.pattern}"
        return None

    def add_pattern(self, pattern: str) -> None:
        """Add a new pattern to the filter at runtime.

        Args:
            pattern: Regex pattern to add
        """
        self.patterns.append(pattern)
        self._compiled.append(re.compile(pattern, re.IGNORECASE))
