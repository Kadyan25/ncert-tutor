import os

from backend.anthropic_client import chat as anthropic_chat
from backend.constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GENERATION_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OUT_OF_SCOPE_MESSAGE,
    OUT_OF_SCOPE_SIMILARITY_THRESHOLD,
    TOP_K,
)
from backend.gemini_client import chat as gemini_chat
from backend.ollama_client import chat as ollama_chat
from backend.prompt import build_messages
from backend.retrieval import Retriever
from backend.types import ChatRequest, ChatResponse, Citation


class RAGAgent:
    """Retrieves grounded NCERT context, then answers via the configured generation provider."""

    def __init__(self) -> None:
        self.provider = os.getenv("GENERATION_PROVIDER", DEFAULT_GENERATION_PROVIDER)
        self.retriever = Retriever()

        if self.provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
            self.api_key = os.environ["ANTHROPIC_API_KEY"]
        elif self.provider == "gemini":
            self.model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            self.api_key = os.environ["GOOGLE_API_KEY"]
        else:
            self.model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "anthropic":
            return anthropic_chat(messages, model=self.model, api_key=self.api_key)
        if self.provider == "gemini":
            return gemini_chat(messages, model=self.model, api_key=self.api_key)
        return ollama_chat(messages, model=self.model, base_url=self.base_url)

    def answer(self, request: ChatRequest) -> ChatResponse:
        question = request.messages[-1].content
        results = self.retriever.search(question, k=TOP_K)

        if not results or results[0][1] < OUT_OF_SCOPE_SIMILARITY_THRESHOLD:
            return ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, model=self.model)

        chunks = [chunk for chunk, _ in results]
        messages = build_messages(request.messages, chunks)
        answer = self.generate(messages)

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
