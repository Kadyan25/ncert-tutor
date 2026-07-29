import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from backend.types import ChatResponse
from evals.constants import BASE_URL, EVALS_DIR, PROMPTS_PATH, REPORTS_DIR
from evals.types import EvalCase, EvalResult


def load_cases(prompts_path: Path) -> list[EvalCase]:
    with prompts_path.open() as prompt_file:
        payload = json.load(prompt_file)
    return [EvalCase.model_validate(case) for case in payload["cases"]]


def run_case(client: httpx.Client, case: EvalCase) -> EvalResult:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": case.prompt}]},
    )
    response.raise_for_status()
    answer = ChatResponse.model_validate(response.json())
    return EvalResult(id=case.id, answer=answer.answer, model=answer.model, citations=answer.citations)


def main() -> None:
    prompts_path = EVALS_DIR / sys.argv[1] if len(sys.argv) > 1 else PROMPTS_PATH
    report_prefix = prompts_path.stem

    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        client.get("/health").raise_for_status()
        results = [run_case(client, case) for case in load_cases(prompts_path)]

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"{report_prefix}_{timestamp}.json"
    report_path.write_text(json.dumps([result.model_dump() for result in results], indent=2))
    print(f"Wrote {len(results)} eval results to {report_path}")


if __name__ == "__main__":
    main()
