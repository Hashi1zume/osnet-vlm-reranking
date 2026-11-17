from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class _IdentityMemory:
    embeddings: deque
    last_frame: int


class TemporalIdentityGallery:
    """Stores multiple observations per identity for robust re-entry matching."""

    def __init__(self, similarity_threshold: float = 0.75, max_samples: int = 12):
        self.similarity_threshold = similarity_threshold
        self.max_samples = max_samples
        self._identities: dict[str, _IdentityMemory] = {}

    def register(self, identity: str, embedding: np.ndarray, frame: int):
        self._identities[identity] = _IdentityMemory(deque(maxlen=self.max_samples), frame)
        self.update(identity, embedding, frame)

    def update(self, identity: str, embedding: np.ndarray, frame: int):
        if identity not in self._identities:
            self.register(identity, embedding, frame)
            return
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm:
            self._identities[identity].embeddings.append(vector / norm)
            self._identities[identity].last_frame = frame

    def samples(self, identity: str):
        return tuple(self._identities[identity].embeddings)

    def match(self, embedding: np.ndarray, frame: int):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not norm:
            return None, 0.0
        vector = vector / norm
        ranked = []
        for identity, memory in self._identities.items():
            # Max similarity keeps one clear view useful even when other samples are occluded.
            score = max(float(np.dot(vector, sample)) for sample in memory.embeddings)
            ranked.append((score, identity))
        if not ranked:
            return None, 0.0
        score, identity = max(ranked)
        return (identity, score) if score >= self.similarity_threshold else (None, score)
