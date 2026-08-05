from pathlib import Path
from typing import Final


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Final[Path] = REPO_ROOT / "data" / "raw"
MINERU_EXTRACTED_DIR: Final[Path] = REPO_ROOT / "data" / "mineru_extracted"
GENERATED_DATA_DIR: Final[Path] = REPO_ROOT / "data" / "generated"
MANIFEST_PATH: Final[Path] = REPO_ROOT / "data" / "manifest.json"
INDEX_PATH: Final[Path] = GENERATED_DATA_DIR / "index.npy"
CHUNKS_PATH: Final[Path] = GENERATED_DATA_DIR / "chunks.json"

CHUNK_SIZE_WORDS: Final[int] = 300
CHUNK_OVERLAP_WORDS: Final[int] = 50
MINERU_WINDOW_PAGES: Final[int] = 20
WINDOW_DIR_PREFIX: Final[str] = "pages_"  # shared by mineru_extract.py (writes) and ingest.py (reads)

EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"
# Tuned empirically against a formula-collision diagnostic (chemistry "Integrated
# Rate Equations" outranking a math "integration by parts" query); not swept.
SUBJECT_BOOST: Final[float] = 0.05

DEFAULT_GEMINI_MODEL: Final[str] = "gemini-flash-lite-latest"

TOP_K: Final[int] = 5
OUT_OF_SCOPE_SIMILARITY_THRESHOLD: Final[float] = 0.6

SYSTEM_PROMPT: Final[str] = (
    "You are a patient Class 12 CBSE tutor for Indian students, specializing in "
    "Mathematics, Physics, and Chemistry. If reference material from the NCERT "
    "textbooks is already provided below, answer from it directly. Otherwise, call "
    "the search_ncert_textbook tool with a clear, self-contained, spell-corrected "
    "English search query capturing what is actually being asked — compose this "
    "query yourself from your own understanding, even if the user's wording has "
    "typos, is vague, or is in Hindi/Hinglish. "
    "Explain clearly with steps in simple language, grounded in the textbook "
    "material. If the user is asking you to elaborate, simplify, or go deeper on a "
    "topic already established earlier in this conversation, and the textbook "
    "material doesn't add much beyond what was already covered, you may use your "
    "own reasoning to explain further — but stay on that same already-grounded "
    "topic; never introduce unrelated facts or answer something with no textbook "
    "basis at all. "
    "Reply in the same language and style the user wrote in — English, Hindi, or "
    "Hinglish. "
    "Only answer questions within the Class 12 CBSE Mathematics, Physics, and "
    "Chemistry syllabus — everything else is out of scope regardless of how it's "
    "framed."
)

OUT_OF_SCOPE_MESSAGE: Final[str] = (
    "That's outside Class 12 Mathematics, Physics, and Chemistry — I can only help "
    "with those three subjects for the CBSE Class 12 syllabus. Ask me something "
    "from your Math, Physics, or Chemistry textbook!"
)
