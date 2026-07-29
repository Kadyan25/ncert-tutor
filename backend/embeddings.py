import os
import ssl

# Some local AV/proxy HTTPS interception (e.g. Avast Web Shield) presents a root cert
# that is valid but predates the Basic-Constraints-criticality rule OpenSSL 3.2+ now
# enforces under VERIFY_X509_STRICT. Opt-in only (RELAX_SSL_VERIFICATION=true) since
# most machines don't need this and it shouldn't be a silent default. Must run before
# anything creates an SSL context (the Hugging Face Hub download client included), so
# it comes before that import.
if os.getenv("RELAX_SSL_VERIFICATION", "false").lower() == "true":
    _original_create_default_context = ssl.create_default_context

    def _relaxed_default_context(*args: object, **kwargs: object) -> ssl.SSLContext:
        context = _original_create_default_context(*args, **kwargs)
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return context

    ssl.create_default_context = _relaxed_default_context

from sentence_transformers import SentenceTransformer  # noqa: E402

from backend.constants import EMBEDDING_MODEL  # noqa: E402

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
