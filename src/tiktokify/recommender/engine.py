"""Combined recommendation engine."""

from tiktokify.models import Post, RecommendationGraph

from .metadata import MetadataSimilarity
from .tfidf import TFIDFSimilarity


class RecommendationEngine:
    """Hybrid recommendation combining content and metadata similarity."""

    def __init__(
        self,
        content_weight: float = 0.6,
        metadata_weight: float = 0.4,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ):
        self.content_weight = content_weight
        self.metadata_weight = metadata_weight
        self.top_k = top_k
        self.min_similarity = min_similarity

        self.tfidf = TFIDFSimilarity()
        self.metadata = MetadataSimilarity()

    def build_graph(self, posts: list[Post]) -> RecommendationGraph:
        """Build complete recommendation graph."""
        # Fit both models
        self.tfidf.fit(posts)
        self.metadata.fit(posts)

        posts_dict = {p.slug: p for p in posts}
        adjacency: dict[str, list[tuple[str, float]]] = {}

        for post in posts:
            # Get similarities from both sources
            content_sims = dict(self.tfidf.get_similar(post.slug, self.top_k * 2))
            metadata_sims = dict(self.metadata.get_similar(post.slug, self.top_k * 2))

            # Combine scores
            all_slugs = set(content_sims.keys()) | set(metadata_sims.keys())
            combined: list[tuple[str, float]] = []

            for slug in all_slugs:
                c_score = content_sims.get(slug, 0)
                m_score = metadata_sims.get(slug, 0)
                combined_score = (
                    self.content_weight * c_score + self.metadata_weight * m_score
                )
                combined.append((slug, combined_score))

            # Sort and filter by min_similarity, then take top_k
            combined.sort(key=lambda x: x[1], reverse=True)
            above_threshold = [
                (slug, score)
                for slug, score in combined
                if score >= self.min_similarity
            ]
            top_recommendations = above_threshold[: self.top_k]
            adjacency[post.slug] = top_recommendations

            # Update post object with recommendations
            post.similar_posts = [s for s, _ in top_recommendations]
            post.similarity_scores = dict(top_recommendations)

        return RecommendationGraph(posts=posts_dict, adjacency=adjacency)
