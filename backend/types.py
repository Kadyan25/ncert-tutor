from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


class ChunkMetadata(BaseModel):
    subject: str
    book: str
    section: str
    source_file: str
    page: int


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class Citation(BaseModel):
    subject: str
    book: str
    section: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    model: str
    citations: list[Citation] = Field(default_factory=list)
