from pydantic import BaseModel, Field

from backend.types import Citation


class EvalCase(BaseModel):
    id: str
    prompt: str


class EvalResult(BaseModel):
    id: str
    answer: str
    model: str
    citations: list[Citation] = Field(default_factory=list)
