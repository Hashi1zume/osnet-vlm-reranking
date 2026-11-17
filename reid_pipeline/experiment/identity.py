class IdentityAssigner:
    """Maps transient detector tracks to stable retrieval identities."""

    def __init__(self, minimum_score: float = 0.75):
        self.minimum_score = minimum_score
        self._assignments = {}

    def assign(self, track_id, reranked_candidates):
        if track_id in self._assignments:
            return self._assignments[track_id]
        candidate = reranked_candidates[0] if reranked_candidates else None
        # A new identity is safer than an unsupported merge; VLM must beat the threshold.
        identity = candidate[0] if candidate and candidate[2] >= self.minimum_score else f"track-{track_id}"
        self._assignments[track_id] = identity
        return identity
