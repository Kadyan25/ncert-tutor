from sentence_transformers import SentenceTransformer

from backend.constants import EMBEDDING_MODEL

# Local embeddings: no API key, no quota, no cost. bge models expect a query-side
# instruction prefix for asymmetric search; document-side text is embedded as-is.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage in the retrieval index. Unit-normalized."""

    return _model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question for similarity search against the index. Unit-normalized."""

    embedding = _model.encode([_QUERY_PREFIX + text], normalize_embeddings=True)[0]
    return embedding.tolist()
