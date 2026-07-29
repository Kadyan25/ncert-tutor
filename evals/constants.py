import os
from pathlib import Path
from typing import Final

BASE_URL: Final[str] = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
EVALS_DIR: Final[Path] = Path(__file__).parent
PROMPTS_PATH: Final[Path] = EVALS_DIR / "prompts.json"
REPORTS_DIR: Final[Path] = EVALS_DIR / "reports"
