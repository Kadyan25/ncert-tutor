import os

from backend.constants import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OUT_OF_SCOPE_MESSAGE,
    OUT_OF_SCOPE_SIMILARITY_THRESHOLD,
    TOP_K,
)
from backend.ollama_client import chat
from backend.prompt import build_messages
from backend.retrieval import Retriever
from backend.types import ChatRequest, ChatResponse, Citation


class RAGAgent:
    """Retrieves grounded NCERT context, then answers with a local Ollama model."""

    def __init__(self) -> None:
        self.model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.retriever = Retriever()

    def answer(self, request: ChatRequest) -> ChatResponse:
        question = request.messages[-1].content
        results = self.retriever.search(question, k=TOP_K)

        if not results or results[0][1] < OUT_OF_SCOPE_SIMILARITY_THRESHOLD:
            return ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, model=self.model)

        chunks = [chunk for chunk, _ in results]
        messages = build_messages(request.messages, chunks)
        answer = chat(messages, model=self.model, base_url=self.base_url)

        citations = [
            Citation(
                subject=chunk.metadata.subject,
                book=chunk.metadata.book,
                section=chunk.metadata.section,
                page=chunk.metadata.page,
            )
            for chunk in chunks
        ]
        return ChatResponse(answer=answer, model=self.model, citations=citations)
