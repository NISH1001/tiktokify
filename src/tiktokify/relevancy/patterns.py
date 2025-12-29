"""URL pattern definitions for relevancy classification."""

import re
from enum import Enum
from urllib.parse import parse_qs, urlparse


class RelevancyResult(Enum):
    """Result of pattern-based classification."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    MAYBE = "maybe"  # Needs LLM analysis


# Patterns that definitively indicate NON-CONTENT pages
NOT_RELEVANT_PATH_PATTERNS = [
    # Tag/category listing pages
    r"^/tags?/?$",
    r"^/tags?/[^/]+/?$",
    r"^/categor(y|ies)/?$",
    r"^/categor(y|ies)/[^/]+/?$",
    # Archive/listing pages
    r"^/archive/?$",
    r"^/archives?/?$",
    r"^/archives?/\d{4}/?$",
    r"^/archives?/\d{4}/\d{2}/?$",
    # Utility pages
    r"^/about/?$",
    r"^/about-me/?$",
    r"^/about-us/?$",
    r"^/contact/?$",
    r"^/contact-us/?$",
    r"^/search/?$",
    r"^/sitemap/?$",
    r"^/sitemap\.xml$",
    r"^/rss/?$",
    r"^/feed/?$",
    r"^/atom/?$",
    r"^/subscribe/?$",
    r"^/newsletter/?$",
    # Pagination roots
    r"^/page/?$",
    r"^/page/\d+/?$",
    r"^/pages?/?$",
    # Auth/account pages
    r"^/login/?$",
    r"^/signin/?$",
    r"^/signup/?$",
    r"^/register/?$",
    r"^/logout/?$",
    r"^/account/?$",
    r"^/profile/?$",
    r"^/settings/?$",
    # Legal pages
    r"^/privacy/?$",
    r"^/privacy-policy/?$",
    r"^/terms/?$",
    r"^/terms-of-service/?$",
    r"^/tos/?$",
    r"^/cookies/?$",
    r"^/cookie-policy/?$",
    # Other utility
    r"^/404/?$",
    r"^/500/?$",
    r"^/error/?$",
    r"^/thanks/?$",
    r"^/thank-you/?$",
]

# Query params that indicate pagination (non-content)
PAGINATION_PARAMS = {"page", "p", "pg", "offset", "start"}


def classify_by_url(url: str) -> tuple[RelevancyResult, str]:
    """
    Classify URL relevancy using pattern matching.

    Returns:
        Tuple of (result, reason)
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    query_params = parse_qs(parsed.query)

    # Check path patterns for non-content pages
    for pattern in NOT_RELEVANT_PATH_PATTERNS:
        if re.match(pattern, path, re.IGNORECASE):
            return RelevancyResult.NOT_RELEVANT, f"URL matches meta page pattern: {pattern}"

    # Check pagination query params
    for param in PAGINATION_PARAMS:
        if param in query_params:
            return RelevancyResult.NOT_RELEVANT, f"URL contains pagination parameter: {param}"

    # Check for /page/N suffix on otherwise content URLs
    if re.search(r"/page/\d+/?$", path):
        return RelevancyResult.NOT_RELEVANT, "URL is a paginated listing page"

    # Heuristics for likely content pages
    # - Has date in URL path (common for blog posts)
    if re.search(r"/\d{4}/\d{2}/\d{2}/", path) or re.search(r"/\d{4}/\d{2}/", path):
        return RelevancyResult.RELEVANT, "URL contains date pattern (likely blog post)"

    # - Has /post/, /posts/, /blog/, /article/ in path
    if re.search(r"/(posts?|blog|articles?)/[^/]+", path, re.IGNORECASE):
        return RelevancyResult.RELEVANT, "URL contains content path prefix"

    # - Ends in .html (common for static site generators, org-roam, braindumps)
    if path.endswith(".html"):
        return RelevancyResult.RELEVANT, "URL ends in .html (likely content page)"

    # - Path has a meaningful slug (long segment with hyphens or underscores)
    segments = [s for s in path.split("/") if s]
    if segments:
        last_segment = segments[-1].replace(".html", "")
        # Long slugs with word separators are usually content
        if len(last_segment) > 15 and ("-" in last_segment or "_" in last_segment):
            return RelevancyResult.RELEVANT, "URL has long slug (likely content)"

    # Uncertain - could be content or meta page
    return RelevancyResult.MAYBE, "URL pattern inconclusive"
