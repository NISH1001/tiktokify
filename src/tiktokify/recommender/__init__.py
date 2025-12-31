"""Recommendation engine module."""

from .base import BaseSimilarity
from .embeddings import EmbeddingSimilarity
from .engine import RecommendationEngine
from .metadata import MetadataSimilarity
from .tfidf import TFIDFSimilarity

__all__ = [
    "BaseSimilarity",
    "EmbeddingSimilarity",
    "MetadataSimilarity",
    "RecommendationEngine",
    "TFIDFSimilarity",
]
