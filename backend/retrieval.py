import json

import numpy as np

from backend.constants import CHUNKS_PATH, INDEX_PATH, SUBJECT_BOOST
from backend.embeddings import embed_query
from backend.types import Chunk


class Retriever:
    def __init__(self) -> None:
        self.index = np.load(INDEX_PATH)  # rows are unit-normalized by embed_documents
        self.chunks = [Chunk.model_validate(c) for c in json.loads(CHUNKS_PATH.read_text())]
        self.subjects = np.array([c.metadata.subject for c in self.chunks])
        self.centroids = {s: self._centroid(s) for s in sorted(set(self.subjects))}
        # soft boost, not a hard filter: misclassifying the subject only costs the
        # boost, correct-subject chunks that score well on raw similarity still win.
        # Precomputed once per subject so search() does no per-query string compare.
        self.boosts = {
            s: np.where(self.subjects == s, SUBJECT_BOOST, 0.0).astype(np.float32) for s in self.centroids
        }

    def _centroid(self, subject: str) -> np.ndarray:
        mean = self.index[self.subjects == subject].mean(axis=0)
        return mean / np.linalg.norm(mean)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_vector = np.array(embed_query(query), dtype=np.float32)  # also unit-normalized
        predicted_subject = max(self.centroids, key=lambda s: query_vector @ self.centroids[s])

        raw_scores = self.index @ query_vector
        ranked_scores = raw_scores + self.boosts[predicted_subject]

        # rank by the boosted score, but return the raw score -- callers use it as an
        # out-of-scope gate, which should reflect genuine similarity, not a subject
        # bonus meant only to break ties among already-plausible candidates
        top_indices = np.argsort(-ranked_scores)[:k]
        return [(self.chunks[i], float(raw_scores[i])) for i in top_indices]
