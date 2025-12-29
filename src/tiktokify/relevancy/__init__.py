"""Relevancy classification module."""

from .classifier import RelevancyClassifier
from .patterns import RelevancyResult, classify_by_url

__all__ = [
    "RelevancyClassifier",
    "RelevancyResult",
    "classify_by_url",
]
