import json

import numpy as np

from backend.constants import CHUNKS_PATH, INDEX_PATH
from backend.embeddings import embed_query
from backend.types import Chunk


class Retriever:
    def __init__(self) -> None:
        self.index = np.load(INDEX_PATH)  # rows are unit-normalized by embed_documents
        self.chunks = [Chunk.model_validate(c) for c in json.loads(CHUNKS_PATH.read_text())]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_vector = np.array(embed_query(query), dtype=np.float32)  # also unit-normalized
        scores = self.index @ query_vector
        top_indices = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
