from collections.abc import Callable

from google import genai
from google.genai import types as genai_types

from backend.prompt import split_system

SEARCH_TOOL_NAME = "search_ncert_textbook"

_SEARCH_TOOL = genai_types.Tool(
    function_declarations=[
        genai_types.FunctionDeclaration(
            name=SEARCH_TOOL_NAME,
            description=(
                "Search the Class 12 CBSE NCERT Mathematics, Physics, and Chemistry "
                "textbooks for content relevant to a question. Always call this "
                "first for any syllabus question, including follow-ups."
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "query": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="A clear, self-contained, English search query.",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)

_clients: dict[str, genai.Client] = {}


def _client_for(api_key: str) -> genai.Client:
    """Reuse one Client per key instead of creating a fresh httpx connection pool
    on every request — churning clients under repeated load was the likely cause
    of severe request hangs seen in production testing."""

    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def _find_function_call(response: genai_types.GenerateContentResponse) -> genai_types.FunctionCall | None:
    for part in response.candidates[0].content.parts:
        if part.function_call is not None:
            return part.function_call
    return None


def _to_contents(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, object]]]:
    system, conversation = split_system(messages)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in conversation
    ]
    return system, contents


def chat(messages: list[dict[str, str]], model: str, api_key: str) -> str:
    """First turn: chunks are already attached in the system message (see
    prompt.build_messages) — a plain single-shot call, no tool offered."""

    system, contents = _to_contents(messages)
    response = _client_for(api_key).models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(system_instruction=system),
    )
    return response.text.strip()


def chat_with_tools(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    search_fn: Callable[[str], str],
) -> str:
    """Follow-up turns: the model composes its own search query, we execute
    search_fn (retrieval + scope check, owned by the caller) and hand the result
    back for a final grounded answer. search_fn may raise to signal out-of-scope —
    that propagates to the caller uncaught, short-circuiting before a second call."""

    system, contents = _to_contents(messages)
    client = _client_for(api_key)
    config = genai_types.GenerateContentConfig(system_instruction=system, tools=[_SEARCH_TOOL])

    response = client.models.generate_content(model=model, contents=contents, config=config)
    call = _find_function_call(response)
    if call is None:
        return response.text.strip()

    tool_result = search_fn(call.args["query"])

    contents.append(response.candidates[0].content)
    contents.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_function_response(name=call.name, response={"result": tool_result})],
        )
    )
    final = client.models.generate_content(model=model, contents=contents, config=config)
    return final.text.strip()
