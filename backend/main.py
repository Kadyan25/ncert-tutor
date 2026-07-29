from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agent import RAGAgent
from backend.types import ChatRequest, ChatResponse


load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="NCERT Tutor Assignment")

# Local-only tool: the static frontend is served from a different origin (or file://,
# which browsers report as a null origin), so it needs CORS opened up to reach /chat.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = RAGAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": agent.model}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    return agent.answer(request)
