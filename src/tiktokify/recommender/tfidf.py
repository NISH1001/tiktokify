"""TF-IDF based content similarity."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tiktokify.models import Post


class TFIDFSimilarity:
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
        self.tfidf_matrix: np.ndarray | None = None
        self.slugs: list[str] = []

    def fit(self, posts: list[Post]) -> None:
        """Fit TF-IDF on post content."""
        self.slugs = [p.slug for p in posts]
        texts = [p.content_text for p in posts]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def get_similarity_matrix(self) -> np.ndarray:
        """Return full cosine similarity matrix."""
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit() first")
        return cosine_similarity(self.tfidf_matrix)

    def get_similar(self, slug: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k similar posts for a given slug."""
        if slug not in self.slugs:
            return []

        idx = self.slugs.index(slug)
        sim_matrix = self.get_similarity_matrix()
        scores = sim_matrix[idx]

        # Get top-k (excluding self)
        top_indices = np.argsort(scores)[::-1][1 : top_k + 1]
        return [(self.slugs[i], float(scores[i])) for i in top_indices]
