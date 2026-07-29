from backend.constants import SYSTEM_PROMPT
from backend.types import Chunk, Message


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk.metadata
        source = f"[{i}] {m.subject.title()} — {m.book}, {m.section}, p.{m.page}"
        blocks.append(f"{source}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(history: list[Message], chunks: list[Chunk]) -> list[dict[str, str]]:
    system_content = f"{SYSTEM_PROMPT}\n\nReference material:\n{format_context(chunks)}"
    messages = [{"role": "system", "content": system_content}]
    messages.extend({"role": m.role, "content": m.content} for m in history)
    return messages


def split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Splits the system message out for providers whose API takes it separately."""

    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    conversation = [m for m in messages if m["role"] != "system"]
    return system, conversation
