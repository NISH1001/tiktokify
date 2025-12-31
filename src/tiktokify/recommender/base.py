"""Base similarity interface and shared utilities."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from tiktokify.models import Post


def compute_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Shared utility for all vector-based similarity methods.

    Args:
        vectors: (n_docs, n_features) array of document vectors.

    Returns:
        (n_docs, n_docs) similarity matrix.
    """
    return sklearn_cosine(vectors)


def get_top_k_from_matrix(
    similarity_matrix: np.ndarray,
    slugs: list[str],
    query_idx: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Extract top-k similar items from similarity matrix.

    Args:
        similarity_matrix: (n, n) pairwise similarity matrix.
        slugs: List of document slugs in same order as matrix.
        query_idx: Index of query document.
        top_k: Number of results to return.

    Returns:
        List of (slug, score) tuples, excluding the query itself.
    """
    scores = similarity_matrix[query_idx]
    top_indices = np.argsort(scores)[::-1]
    results = []
    for idx in top_indices:
        if idx != query_idx and len(results) < top_k:
            results.append((slugs[idx], float(scores[idx])))
    return results


class BaseSimilarity(ABC):
    """Abstract base class for similarity implementations.

    All similarity methods must implement:
    - fit(): Build the similarity index from posts (async for API-based methods)
    - get_similar(): Retrieve top-k similar posts for a query
    - name: Human-readable identifier
    """

    @abstractmethod
    async def fit(self, posts: list[Post]) -> None:
        """Fit the similarity model on posts (async for API-based methods)."""
        pass

    @abstractmethod
    def get_similar(self, slug: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k similar posts for a given slug."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this similarity method."""
        pass
