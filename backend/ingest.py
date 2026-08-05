"""Rerunnable ingestion: data/mineru_extracted/ -> data/generated/{index.npy, chunks.json}.

Requires MinerU extraction to have already run (backend/mineru_extract.py --
slow, one-time, hours for the full corpus; see requirements-ingest.txt). This
step itself is fast and safe to rerun any time chunking/embedding logic
changes:
    PYTHONPATH=. python -m backend.ingest
"""

import json
import re
from pathlib import Path

import numpy as np

from backend.constants import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    GENERATED_DATA_DIR,
    INDEX_PATH,
    MANIFEST_PATH,
    MINERU_EXTRACTED_DIR,
    WINDOW_DIR_PREFIX,
)
from backend.embeddings import embed_documents
from backend.types import Chunk, ChunkMetadata

SKIP_BLOCK_TYPES = {"header", "footer", "page_number"}
WINDOW_START_RE = re.compile(rf"{WINDOW_DIR_PREFIX}(\d+)-\d+")


def load_manifest() -> dict[str, dict[str, str]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {book["ncert_code"]: book for book in manifest["books"]}


def section_label(suffix: str) -> str:
    if suffix.isdigit():
        return f"Chapter {int(suffix)}"
    if suffix == "ps":
        return "Prelims"
    if suffix == "an":
        return "Answers"
    if re.fullmatch(r"a\d", suffix):
        return f"Appendix {suffix[1]}"
    return suffix.upper()


def _block_text(block: dict) -> str:
    if block["type"] == "table":
        return re.sub(r"<[^>]+>", " ", block.get("table_body", ""))
    return block.get("text", "")


def extract_pages(stem_dir: Path) -> list[tuple[int, str]]:
    """Merge every MinerU page-window for one PDF into {page_number: text}.
    page_idx in each window's content_list.json is window-relative, so the
    true page number is the window's start offset (parsed from its directory
    name) plus page_idx, 1-indexed."""
    pages: dict[int, list[str]] = {}
    for window_dir in sorted(stem_dir.glob(f"{WINDOW_DIR_PREFIX}*")):
        match = WINDOW_START_RE.match(window_dir.name)
        if not match:
            continue
        window_start = int(match.group(1))
        content_list = window_dir / stem_dir.name / "auto" / f"{stem_dir.name}_content_list.json"
        if not content_list.exists():
            continue
        for block in json.loads(content_list.read_text(encoding="utf-8")):
            if block["type"] in SKIP_BLOCK_TYPES:
                continue
            text = _block_text(block)
            if text.strip():
                pages.setdefault(window_start + block["page_idx"] + 1, []).append(text)
    return [(page, re.sub(r"\s+", " ", " ".join(pages[page])).strip()) for page in sorted(pages)]


def chunk_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + CHUNK_SIZE_WORDS]))
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
    return chunks


def build_chunks() -> list[Chunk]:
    manifest = load_manifest()
    chunks: list[Chunk] = []
    for stem_dir in sorted(d for d in MINERU_EXTRACTED_DIR.iterdir() if d.is_dir() and d.name != "_logs"):
        code, suffix = stem_dir.name[:5], stem_dir.name[5:]
        if suffix == "ps" or code not in manifest:
            continue  # prelims: committee/preface boilerplate, no subject content
        book = manifest[code]
        section = section_label(suffix)
        pages = extract_pages(stem_dir)
        if not pages:
            print(f"WARNING: {stem_dir.name} produced 0 pages -- check its windows extracted cleanly")
        for page_number, page_text in pages:
            for chunk_text in chunk_words(page_text):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            subject=book["subject"],
                            book=book["book"],
                            section=section,
                            source_file=f"{stem_dir.name}.pdf",
                            page=page_number,
                        ),
                    )
                )
    return chunks


def main() -> None:
    chunks = build_chunks()
    print(f"Extracted {len(chunks)} chunks from {MINERU_EXTRACTED_DIR}")

    vectors = np.array(embed_documents([chunk.text for chunk in chunks]), dtype=np.float32)
    print(f"Embedded {len(chunks)} chunks -> shape {vectors.shape}")

    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, vectors)
    CHUNKS_PATH.write_text(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
    print(f"Wrote {INDEX_PATH} and {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
