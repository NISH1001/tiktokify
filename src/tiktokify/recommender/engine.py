"""Combined recommendation engine."""

from tiktokify.models import Post, RecommendationGraph

from .embeddings import EmbeddingSimilarity
from .metadata import MetadataSimilarity
from .tfidf import TFIDFSimilarity


class RecommendationEngine:
    """Hybrid recommendation combining content, metadata, and semantic similarity."""

    def __init__(
        self,
        content_weight: float = 0.6,
        metadata_weight: float = 0.4,
        embedding_weight: float = 0.0,
        embedding_model: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        max_concurrent: int = 5,
        verbose: bool = False,
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity

        self.tfidf = TFIDFSimilarity()
        self.metadata = MetadataSimilarity()
        self.embeddings: EmbeddingSimilarity | None = None

        # Initialize embedding similarity if model specified
        if embedding_model:
            self.embeddings = EmbeddingSimilarity(
                model=embedding_model,
                max_concurrent=max_concurrent,
                verbose=verbose,
            )
            # Normalize weights to sum to 1.0
            total = content_weight + metadata_weight + embedding_weight
            self.content_weight = content_weight / total
            self.metadata_weight = metadata_weight / total
            self.embedding_weight = embedding_weight / total
        else:
            # No embeddings - normalize content and metadata only
            total = content_weight + metadata_weight
            self.content_weight = content_weight / total
            self.metadata_weight = metadata_weight / total
            self.embedding_weight = 0.0

    async def build_graph(self, posts: list[Post]) -> RecommendationGraph:
        """Build complete recommendation graph (async for embedding support)."""
        # Fit all similarity models (async)
        await self.tfidf.fit(posts)
        await self.metadata.fit(posts)
        if self.embeddings:
            await self.embeddings.fit(posts)

        posts_dict = {p.slug: p for p in posts}
        adjacency: dict[str, list[tuple[str, float]]] = {}

        for post in posts:
            # Get similarities from all sources
            content_sims = dict(self.tfidf.get_similar(post.slug, self.top_k * 2))
            metadata_sims = dict(self.metadata.get_similar(post.slug, self.top_k * 2))
            embedding_sims: dict[str, float] = {}
            if self.embeddings:
                embedding_sims = dict(self.embeddings.get_similar(post.slug, self.top_k * 2))

            # Combine scores from all sources
            all_slugs = set(content_sims.keys()) | set(metadata_sims.keys()) | set(embedding_sims.keys())
            combined: list[tuple[str, float]] = []

            for slug in all_slugs:
                c_score = content_sims.get(slug, 0)
                m_score = metadata_sims.get(slug, 0)
                e_score = embedding_sims.get(slug, 0)
                combined_score = (
                    self.content_weight * c_score
                    + self.metadata_weight * m_score
                    + self.embedding_weight * e_score
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
