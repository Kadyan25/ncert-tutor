import anthropic

from backend.prompt import split_system


def chat(messages: list[dict[str, str]], model: str, api_key: str) -> str:
    system, conversation = split_system(messages)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=conversation,
    )
    return response.content[0].text.strip()
