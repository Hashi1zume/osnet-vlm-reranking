from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    identity: str
    osnet_score: float


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class OSNetRetriever:
    """Ranks gallery embeddings while keeping the encoder independent of search policy."""

    def __init__(self, default_k: int = 10):
        # Keeping K in the retriever makes candidate-budget changes explicit and testable.
        self.default_k = default_k

    def retrieve(self, query_embedding, gallery_embeddings, k: int | None = None):
        candidate_limit = self.default_k if k is None else k
        ranked = [
            Candidate(identity, _cosine(query_embedding, embedding))
            for identity, embedding in gallery_embeddings.items()
        ]
        return sorted(ranked, key=lambda item: item.osnet_score, reverse=True)[:candidate_limit]


def retrieve_top_k(query_embedding, gallery_embeddings, k=10):
    """Compatibility wrapper for callers that do not need a configured retriever."""
    return OSNetRetriever(default_k=k).retrieve(query_embedding, gallery_embeddings)
