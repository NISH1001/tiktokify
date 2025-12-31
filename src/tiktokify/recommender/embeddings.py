"""Embedding-based semantic similarity."""

import asyncio
from functools import lru_cache

import numpy as np

from tiktokify.models import Post

from .base import BaseSimilarity, compute_cosine_similarity, get_top_k_from_matrix


@lru_cache(maxsize=1024)
def _cached_embed_local(model_name: str, text: str) -> tuple:
    """Cache embeddings by (model, text). Returns tuple for hashability.

    Note: Model loading is cached internally by sentence-transformers.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return tuple(model.encode([text])[0].tolist())


class EmbeddingSimilarity(BaseSimilarity):
    """Semantic similarity via embeddings (local or API).

    Auto-detection:
    - Models starting with 'sentence-transformers/' use local sentence-transformers
    - Other models use litellm API (OpenAI, Voyage, etc.)

    Examples:
        # Local (free, no API key)
        EmbeddingSimilarity(model="sentence-transformers/all-MiniLM-L6-v2")

        # API (requires OPENAI_API_KEY)
        EmbeddingSimilarity(model="text-embedding-3-small")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_concurrent: int = 5,
        verbose: bool = False,
    ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.use_local = model.startswith("sentence-transformers/")
        self._similarity_matrix: np.ndarray | None = None
        self.slugs: list[str] = []

        if self.use_local:
            # Strip prefix for sentence-transformers model name
            self.model_name = model.replace("sentence-transformers/", "")

    @property
    def name(self) -> str:
        return f"embedding:{self.model}"

    async def fit(self, posts: list[Post]) -> None:
        """Generate embeddings and compute similarity matrix."""
        self.slugs = [p.slug for p in posts]
        texts = [self._prepare_text(p) for p in posts]

        if self.use_local:
            # Use cached local embeddings (sync, but fast due to caching)
            embeddings = [list(_cached_embed_local(self.model_name, t)) for t in texts]
        else:
            # Async API embeddings
            embeddings = await self._embed_via_api(texts)

        self._similarity_matrix = compute_cosine_similarity(np.array(embeddings))

    def get_similar(self, slug: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k similar posts for a given slug."""
        if slug not in self.slugs or self._similarity_matrix is None:
            return []
        idx = self.slugs.index(slug)
        return get_top_k_from_matrix(self._similarity_matrix, self.slugs, idx, top_k)

    def _prepare_text(self, post: Post) -> str:
        """Prepare rich text for embedding (title + tags + content excerpt).

        Combines metadata with content for better semantic representation.
        """
        parts = [post.metadata.title]
        if post.metadata.subtitle:
            parts.append(post.metadata.subtitle)
        if post.metadata.tags:
            parts.append("Tags: " + ", ".join(post.metadata.tags))
        # Truncate content for token limits (most embedding models have 8k limit)
        parts.append(post.content_text[:6000])
        return "\n\n".join(parts)

    async def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via litellm (async, rate limited)."""
        import litellm

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def embed_one(text: str) -> list[float]:
            async with semaphore:
                response = await litellm.aembedding(model=self.model, input=[text])
                return response.data[0]["embedding"]

        return await asyncio.gather(*[embed_one(t) for t in texts])
