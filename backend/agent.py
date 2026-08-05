import os

from backend.constants import (
    DEFAULT_GEMINI_MODEL,
    OUT_OF_SCOPE_MESSAGE,
    OUT_OF_SCOPE_SIMILARITY_THRESHOLD,
    TOP_K,
)
from backend.gemini_client import chat as gemini_chat
from backend.gemini_client import chat_with_tools as gemini_chat_with_tools
from backend.prompt import build_agent_messages, build_messages, format_context
from backend.retrieval import Retriever
from backend.types import Chunk, ChatRequest, ChatResponse, Citation


class OutOfScope(Exception):
    """Raised by the follow-up search tool callback when nothing clears the
    retrieval threshold."""


def _citations(chunks: list[Chunk]) -> list[Citation]:
    return [
        Citation(
            subject=chunk.metadata.subject,
            book=chunk.metadata.book,
            section=chunk.metadata.section,
            page=chunk.metadata.page,
        )
        for chunk in chunks
    ]


class RAGAgent:
    """First turn: deterministic retrieval from the raw question, no tool call.
    Follow-up turns: a tool-calling Gemini turn where the model composes its own
    search query and may lean on its own reasoning to elaborate on already-grounded
    context — see constants.SYSTEM_PROMPT."""

    def __init__(self) -> None:
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.retriever = Retriever()

    def answer(self, request: ChatRequest) -> ChatResponse:
        if len(request.messages) == 1:
            return self._answer_first_turn(request)
        return self._answer_follow_up(request)

    def _answer_first_turn(self, request: ChatRequest) -> ChatResponse:
        question = request.messages[-1].content
        results = self.retriever.search(question, k=TOP_K)

        if not results or results[0][1] < OUT_OF_SCOPE_SIMILARITY_THRESHOLD:
            return ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, model=self.model)

        chunks = [chunk for chunk, _ in results]
        messages = build_messages(request.messages, chunks)
        answer = gemini_chat(messages, model=self.model, api_key=self.api_key)
        return ChatResponse(answer=answer, model=self.model, citations=_citations(chunks))

    def _answer_follow_up(self, request: ChatRequest) -> ChatResponse:
        used_chunks: list[Chunk] = []

        def search(query: str) -> str:
            results = self.retriever.search(query, k=TOP_K)
            if not results or results[0][1] < OUT_OF_SCOPE_SIMILARITY_THRESHOLD:
                raise OutOfScope
            used_chunks.extend(chunk for chunk, _ in results)
            return format_context(used_chunks)

        messages = build_agent_messages(request.messages)
        try:
            answer = gemini_chat_with_tools(messages, model=self.model, api_key=self.api_key, search_fn=search)
        except OutOfScope:
            return ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, model=self.model)

        return ChatResponse(answer=answer, model=self.model, citations=_citations(used_chunks))
