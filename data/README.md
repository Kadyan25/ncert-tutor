# NCERT PDFs

Download the six NCERT Class 12 books (Math I/II, Physics I/II, Chemistry I/II) from
[ncert.nic.in/textbook.php](https://ncert.nic.in/textbook.php): select Class XII, pick
the subject and book, then use "Download complete book."

That download is a **zip of individual chapter PDFs**, not one merged file — this is
intentional to keep per-chapter citations. Extract each zip into its own folder:

```
data/raw/
  mathematics_part_1/   (from Mathematics Part-I's zip)
  mathematics_part_2/
  physics_part_1/
  physics_part_2/
  chemistry_part_1/
  chemistry_part_2/
```

`manifest.json` has the exact NCERT codes and subject/book names `backend/ingest.py`
matches against — the folder names above just need to contain the extracted chapter
PDFs directly (no further renaming needed, ingestion reads the filename prefix).

Run `PYTHONPATH=. python -m backend.ingest` afterward to build the retrieval index —
see the root README for details.
