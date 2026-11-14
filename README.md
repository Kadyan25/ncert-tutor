# NCERT Class 12 Tutor

A local-first RAG tutor for Class 12 CBSE Mathematics, Physics, and Chemistry. It
answers strictly from the official NCERT textbooks, cites its sources, and declines
anything outside the syllabus.

## Architecture

```
data/raw/*.pdf  --[ingest CLI]-->  data/generated/{index.npy, chunks.json}

user question --[embed: local bge-small]--> query vector
              --[cosine similarity vs index.npy]--> top-5 chunks + metadata
              --[similarity below threshold?]--> refuse/redirect
              --[else: build grounded prompt]--> [generate: configurable provider]
              --> answer + citations
```

- **Ingestion** (`backend/ingest.py`): extracts text per NCERT chapter PDF, chunks it
  (~300 words, 50-word overlap), embeds every chunk locally, and writes a numpy index
  + JSON metadata sidecar. Rerunnable end-to-end from `data/raw/` alone.
- **Retrieval** (`backend/retrieval.py`): embeds the incoming question with the same
  local model, does brute-force cosine similarity against the stored index (fast
  enough at ~1900 chunks — no vector DB needed), returns the top-5 chunks.
- **Scope guard** (`backend/agent.py`): if the best retrieved chunk scores below a
  calibrated similarity threshold, the question is treated as out-of-syllabus and
  answered with a fixed refusal — no generation call is made at all.
- **Generation** (`backend/agent.py` + `*_client.py`): builds a prompt from the
  retrieved chunks and calls a configurable provider (see below). Citations are
  attached directly from the retrieved chunks' metadata, not parsed out of the
  model's answer — it can't hallucinate a wrong citation this way.
- **Frontend** (`frontend/`): a dark-themed chat UI, plain HTML/CSS/JS, no build step.

## Setup

**macOS/Linux:**

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

**Windows (PowerShell):**

```powershell
python -m venv backend/.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
```

If `Activate.ps1` is blocked, PowerShell's script execution policy needs to allow it
first: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Windows (cmd.exe):**

```cmd
python -m venv backend/.venv
backend\.venv\Scripts\activate.bat
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
```

Edit `backend/.env` and set `GOOGLE_API_KEY` — see [Model policy](#model-policy)
below. All commands below (`python -m ...`) work identically across all three shells.

### Get the data

Download the six NCERT Class 12 PDFs (see `data/README.md` for exact links/naming)
into `data/raw/{subject}_part_{n}/`. Then build the retrieval index:

```bash
python -m backend.ingest
```

This downloads a small local embedding model on first run (~130MB, one-time) and
writes `data/generated/index.npy` + `chunks.json`. Takes ~10 minutes on CPU for the
full corpus; rerun any time to rebuild from scratch.

### Run the server

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

## Chat API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain definite integrals."}]}'
```

Response includes `answer`, `model`, and `citations` (book/chapter/page per source
chunk used). There are two distinct refusal paths, and only one leaves `citations`
empty: a similarity score below the threshold means nothing relevant was found and
the response is a hard refusal with empty citations. A score that clears the
threshold but doesn't actually contain the answer still returns citations — those
retrieved chunks are genuinely relevant to the topic, the model just honestly said
they don't cover the specific question — see `evals/reports/red_team_eval.json`'s
`ignore-grounding-instruction` and `fake-citation-request` cases for real examples of
this second path.

## Model policy

Generation is **Gemini Flash-Lite only** (`GOOGLE_API_KEY`, free tier available).
**Embeddings are always local** (`bge-small-en-v1.5`), independent of generation —
see [Tradeoffs](#tradeoffs) for why.

Many RAG starter templates default to OpenAI; this one doesn't. Earlier versions also
supported local Ollama and Claude Haiku as swappable generation backends during development — both were
tested and validated (see [Tradeoffs](#tradeoffs) for the reasoning), then removed
once Gemini-only was confirmed sufficient going forward. `ollama_client.py` /
`anthropic_client.py` are gone from the codebase; the historical eval reports
comparing them are still in `evals/reports/` for reference.

## Frontend

Optional dark-themed chat UI, no build step:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` (backend must be running). See `frontend/README.md`.

## Evals

With the backend running:

```bash
python -m evals.run_evals
```

Reports are written to `evals/reports/visible_eval_<timestamp>.json` (gitignored —
auto-generated). Labeled runs are checked in:

- `evals/reports/gemini_flash_lite_eval.json` — the shipped default's actual results.
- `evals/reports/gemma2_2b_eval.json` / `qwen2.5_3b_eval.json` — the local-model
  comparison that picked `gemma2:2b` over Qwen (see Tradeoffs).
- `evals/reports/red_team_eval.json` — output from a second prompt set,
  `evals/red_team_prompts.json`, covering the jailbreak/injection attempts described
  in Limitations. Run it the same way: `python -m evals.run_evals red_team_prompts.json`.

## Tradeoffs

**Embeddings stay local (`bge-small-en-v1.5`), never hosted**, even though generation
uses a hosted provider by default. This was tested, not assumed: compared local
`bge-small`, local `bge-large`, and Google's `gemini-embedding-001` across three
separate tests (a 91-chunk sample, the full 1890-chunk corpus, and a 1740-chunk
Gemini-embedded subset after working around free-tier quota limits). `bge-small` won
every time on the metric that matters most here — separation between in-scope and
out-of-scope similarity scores (~0.22 margin vs Gemini's ~0.13) — while being free,
fastest, and carrying zero quota risk. It's the better retrieval model for this task,
not a compromise. Query embedding also draws from the same shared free-tier quota
pool as bulk document embedding, which would make retrieval a live-runtime risk in a
real chat app if it were hosted — local has zero exposure to that.

**Generation is Gemini Flash-Lite only, not local Ollama or Claude Haiku.** This
project started fully local: `gemma2:2b` was picked over `qwen2.5:3b-instruct` after
running the full eval suite against both — Qwen wrote raw LaTeX that renders as
broken text without a LaTeX-aware frontend, and fabricated a specific section-number
citation that didn't appear anywhere in the actual retrieved source text, which is a
real trust problem for a tutor bot. But local generation is genuinely bottlenecked by
consumer GPU hardware (~19-20s per answer, confirmed via direct A/B testing that
swapping embedding provider changes total latency by under 0.5s — generation is
the entire bottleneck). Hosted generation cut that by 8-9x. Gemini Flash-Lite and
Claude Haiku were tested head-to-head through the actual API (same question, same
retrieval, only generation differed): Gemini was ~31% faster (3.85s vs 5.6s), both
equally well-grounded and correctly cited — Haiku was tested as a speed comparison
point, never the default. Both `ollama_client.py` and `anthropic_client.py` were kept
as fully working, documented alternatives for a while (multi-key
resilience mattered under free-tier constraints at the time), then removed once
Gemini-only was confirmed sufficient — see `evals/reports/gemma2_2b_eval.json` /
`qwen2.5_3b_eval.json` for the historical local-model comparison that's no longer
live code but is still useful evidence.

**No vector database.** Brute-force cosine similarity over a numpy array is simpler
to run and debug, and fast enough at this corpus size (~1900 chunks, sub-millisecond
search). Would reconsider at a much larger corpus.

**Chapter-level PDFs, not merged into one file per book.** NCERT's "complete book"
downloads are actually zips of individual chapter PDFs — kept them that way, which
gives chapter-level citation granularity for free instead of needing to track it
separately.

## Limitations

- **Citation granularity stops at chapter + page**, not sub-section. No stored
  "Section 1.9" style label — retrieval works on chunk-text similarity only.
- **No OCR.** If any NCERT page were a scanned image rather than extractable text,
  ingestion would silently skip it (not observed in practice with the actual NCERT
  PDFs, which are text-based).
- **Bare `$` LaTeX delimiters are ambiguous with currency.** The frontend renders
  math via KaTeX (`$...$`, `$$...$$`), which works well for the actual model output,
  but two unrelated `$` signs in the same message (e.g. a price mention) would get
  misread as one math expression. Not a realistic risk for a Math/Physics/Chemistry
  tutor, so left as-is rather than adding a fragile heuristic to guard against it.
- **Out-of-scope detection is a similarity threshold**, calibrated empirically
  (0.6, from observed score distributions), backed by an explicit system-prompt
  refusal instruction. Red-teamed with jailbreak/override phrasing, roleplay, fake
  system tags, fake citations, translation bypass, and urgency flooding — see
  `evals/red_team_prompts.json` and its results in
  `evals/reports/red_team_eval.json` (12/12 held, zero real answers to off-topic
  requests). A few additional multi-turn attacks (fake conversation history claiming
  prior permission was granted) were tested manually, not captured in this harness
  since it's single-turn only — one inconsistent, low-severity finding there: fake
  history occasionally got a response to acknowledge an off-topic word by name
  without ever answering it.
- **Single provider, single point of failure.** Generation depends entirely on
  Gemini's API being reachable and within quota — no local fallback exists anymore
  (traded away deliberately for a simpler, leaner codebase; see WORKLOG for the
  reasoning). A prior version supported Ollama as a zero-key fallback; removed once
  no longer needed.

## Worklog

See `WORKLOG.md` for AI tools used and the reasoning behind the key decisions above.
