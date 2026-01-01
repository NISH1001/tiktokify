"""TF-IDF based content similarity."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tiktokify.models import Post

from .base import BaseSimilarity, compute_cosine_similarity, get_top_k_from_matrix


class TFIDFSimilarity(BaseSimilarity):
    """Content-based similarity using TF-IDF."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=1,
            max_df=0.9,
        )
        self._similarity_matrix: np.ndarray | None = None
        self.slugs: list[str] = []

    @property
    def name(self) -> str:
        return "tfidf"

    async def fit(self, posts: list[Post]) -> None:
        """Fit TF-IDF on post content (sync internally, async interface)."""
        self.slugs = [p.slug for p in posts]

        # Handle edge case: need at least 2 posts for similarity
        if len(posts) < 2:
            self._similarity_matrix = np.zeros((len(posts), len(posts)))
            return

        texts = [p.content_text for p in posts]
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self._similarity_matrix = compute_cosine_similarity(tfidf_matrix)

    def get_similar(self, slug: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k similar posts for a given slug."""
        if slug not in self.slugs or self._similarity_matrix is None:
            return []
        idx = self.slugs.index(slug)
        return get_top_k_from_matrix(self._similarity_matrix, self.slugs, idx, top_k)
