"""Tag and category based similarity."""

from tiktokify.models import Post


class MetadataSimilarity:
    """Tag and category based Jaccard similarity."""

    def __init__(
        self,
        tag_weight: float = 0.7,
        category_weight: float = 0.3,
    ):
        self.tag_weight = tag_weight
        self.category_weight = category_weight
        self.posts: dict[str, Post] = {}

    def fit(self, posts: list[Post]) -> None:
        """Store posts for similarity computation."""
        self.posts = {p.slug: p for p in posts}

    def compute_similarity(self, slug1: str, slug2: str) -> float:
        """Compute Jaccard-like similarity between two posts."""
        p1, p2 = self.posts.get(slug1), self.posts.get(slug2)
        if not p1 or not p2:
            return 0.0

        # Tag similarity (Jaccard index)
        tags1, tags2 = set(p1.metadata.tags), set(p2.metadata.tags)
        tag_union = tags1 | tags2
        tag_sim = len(tags1 & tags2) / len(tag_union) if tag_union else 0

        # Category similarity (exact match)
        cats1, cats2 = set(p1.metadata.categories), set(p2.metadata.categories)
        cat_union = cats1 | cats2
        cat_sim = len(cats1 & cats2) / len(cat_union) if cat_union else 0

        return self.tag_weight * tag_sim + self.category_weight * cat_sim

    def get_similar(self, slug: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k similar posts based on metadata."""
        if slug not in self.posts:
            return []

        scores = [
            (other_slug, self.compute_similarity(slug, other_slug))
            for other_slug in self.posts
            if other_slug != slug
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
