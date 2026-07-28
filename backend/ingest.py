"""Rerunnable ingestion: data/raw/*.pdf -> data/generated/{index.npy, chunks.json}.

Rebuild from scratch any time with:
    PYTHONPATH=. python -m backend.ingest
"""

import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from backend.constants import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    GENERATED_DATA_DIR,
    INDEX_PATH,
    MANIFEST_PATH,
    RAW_DATA_DIR,
)
from backend.embeddings import embed_documents
from backend.types import Chunk, ChunkMetadata


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


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


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
    for pdf_path in sorted(RAW_DATA_DIR.glob("*/*.pdf")):
        code, suffix = pdf_path.stem[:5], pdf_path.stem[5:]
        if suffix == "ps":
            continue  # prelims: committee/preface boilerplate, no subject content
        book = manifest[code]
        section = section_label(suffix)
        for page_number, page_text in extract_pages(pdf_path):
            for chunk_text in chunk_words(page_text):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            subject=book["subject"],
                            book=book["book"],
                            section=section,
                            source_file=pdf_path.name,
                            page=page_number,
                        ),
                    )
                )
    return chunks


def main() -> None:
    chunks = build_chunks()
    print(f"Extracted {len(chunks)} chunks from {RAW_DATA_DIR}")

    vectors = np.array(embed_documents([chunk.text for chunk in chunks]), dtype=np.float32)
    print(f"Embedded {len(chunks)} chunks -> shape {vectors.shape}")

    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, vectors)
    CHUNKS_PATH.write_text(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
    print(f"Wrote {INDEX_PATH} and {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
