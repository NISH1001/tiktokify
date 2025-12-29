"""Spider crawler module for fetching website content."""

from .blog_crawler import SpiderCrawler

# Backward compatibility alias
JekyllBlogCrawler = SpiderCrawler

__all__ = ["SpiderCrawler", "JekyllBlogCrawler"]
