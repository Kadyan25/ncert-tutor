import httpx


def chat(messages: list[dict[str, str]], model: str, base_url: str) -> str:
    response = httpx.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
