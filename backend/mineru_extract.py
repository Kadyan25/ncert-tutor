"""One-time, resumable extraction: data/raw/*.pdf -> data/mineru_extracted/.

Slow (CPU-only vision pipeline, hours for the full corpus) -- run once, kept
separate from the fast ingest.py rebuild. Windowed into MINERU_WINDOW_PAGES-page
batches to stay under this machine's memory ceiling (a single 67-page batch
OOM'd once during development); safe to interrupt and re-run, already-done
windows are skipped. Needs backend/requirements-ingest.txt installed.

Run with:
    PYTHONPATH=. python -m backend.mineru_extract
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from pypdf import PdfReader

from backend.constants import (
    MINERU_EXTRACTED_DIR,
    MINERU_WINDOW_PAGES,
    RAW_DATA_DIR,
    REPO_ROOT,
    WINDOW_DIR_PREFIX,
)

CACHE_ROOT = REPO_ROOT / ".mineru-cache"


def _mineru_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    env["MODELSCOPE_CACHE"] = str(CACHE_ROOT / "modelscope")
    env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    env["MINERU_MODEL_SOURCE"] = "huggingface"
    return env


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _windows(pdf_path: Path) -> list[tuple[int, int]]:
    n_pages = len(PdfReader(pdf_path).pages)
    return [
        (start, min(start + MINERU_WINDOW_PAGES - 1, n_pages - 1))
        for start in range(0, n_pages, MINERU_WINDOW_PAGES)
    ]


def _window_dir(stem: str, start: int, end: int) -> Path:
    return MINERU_EXTRACTED_DIR / stem / f"{WINDOW_DIR_PREFIX}{start:03d}-{end:03d}"


def _window_done(outdir: Path, stem: str) -> bool:
    # ingest.py reads content_list.json, not the .md -- check the file it actually
    # depends on, so a partial MinerU write can't be marked done-forever here while
    # ingest.py silently contributes zero pages from it.
    return (outdir / stem / "auto" / f"{stem}_content_list.json").exists()


def _run_window(pdf_path: Path, start: int, end: int, log_dir: Path) -> bool:
    stem = pdf_path.stem
    outdir = _window_dir(stem, start, end)
    if _window_done(outdir, stem):
        return True
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mineru.cli.client",
        "-p", str(pdf_path), "-o", str(outdir),
        "-b", "pipeline", "-s", str(start), "-e", str(end),
    ]
    with open(log_dir / f"{stem}_p{start:03d}-{end:03d}.log", "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, env=_mineru_env(), stdout=f, stderr=subprocess.STDOUT)
    return result.returncode == 0 and _window_done(outdir, stem)


def main() -> None:
    pdf_paths = [p for p in sorted(RAW_DATA_DIR.glob("*/*.pdf")) if not p.stem.endswith("ps")]
    windows = [(p, s, e) for p in pdf_paths for s, e in _windows(p)]
    log_dir = MINERU_EXTRACTED_DIR / "_logs"

    done = sum(_window_done(_window_dir(p.stem, s, e), p.stem) for p, s, e in windows)
    _log(f"START: {len(windows)} windows across {len(pdf_paths)} PDFs, {done} already done")

    ok, failed = 0, 0
    for i, (pdf_path, start, end) in enumerate(windows, start=1):
        if _run_window(pdf_path, start, end, log_dir):
            ok += 1
        else:
            failed += 1
            _log(f"FAILED {pdf_path.stem} pages {start}-{end} ({i}/{len(windows)}) -- see {log_dir}")

    _log(f"ALL DONE: {ok} ok, {failed} failed, {len(windows)} total")


if __name__ == "__main__":
    main()
