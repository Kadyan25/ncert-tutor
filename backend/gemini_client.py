from google import genai
from google.genai import types as genai_types

from backend.prompt import split_system


def chat(messages: list[dict[str, str]], model: str, api_key: str) -> str:
    system, conversation = split_system(messages)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in conversation
    ]

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(system_instruction=system),
    )
    return response.text.strip()
