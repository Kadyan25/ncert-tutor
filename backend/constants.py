from pathlib import Path
from typing import Final


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Final[Path] = REPO_ROOT / "data" / "raw"
GENERATED_DATA_DIR: Final[Path] = REPO_ROOT / "data" / "generated"
MANIFEST_PATH: Final[Path] = REPO_ROOT / "data" / "manifest.json"
INDEX_PATH: Final[Path] = GENERATED_DATA_DIR / "index.npy"
CHUNKS_PATH: Final[Path] = GENERATED_DATA_DIR / "chunks.json"

CHUNK_SIZE_WORDS: Final[int] = 300
CHUNK_OVERLAP_WORDS: Final[int] = 50

EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"

DEFAULT_OLLAMA_MODEL: Final[str] = "gemma2:2b"
DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"

TOP_K: Final[int] = 5
OUT_OF_SCOPE_SIMILARITY_THRESHOLD: Final[float] = 0.6

SYSTEM_PROMPT: Final[str] = (
    "You are a patient Class 12 CBSE tutor for Indian students, specializing in "
    "Mathematics, Physics, and Chemistry. Answer strictly using the reference "
    "material provided below, drawn from the official NCERT textbooks. Explain "
    "clearly with steps in simple language. If the reference material does not "
    "contain enough information to answer, say so honestly rather than guessing. "
    "Only answer questions within the Class 12 CBSE Mathematics, Physics, and "
    "Chemistry syllabus."
)

OUT_OF_SCOPE_MESSAGE: Final[str] = (
    "That's outside Class 12 Mathematics, Physics, and Chemistry — I can only help "
    "with those three subjects for the CBSE Class 12 syllabus. Ask me something "
    "from your Math, Physics, or Chemistry textbook!"
)
