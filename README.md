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

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate  # Windows: backend\.venv\Scripts\activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` — see [Model policy](#model-policy) below for what each
`GENERATION_PROVIDER` value needs.

### Get the data

Download the six NCERT Class 12 PDFs (see `data/README.md` for exact links/naming)
into `data/raw/{subject}_part_{n}/`. Then build the retrieval index:

```bash
PYTHONPATH=. python -m backend.ingest
```

This downloads a small local embedding model on first run (~130MB, one-time) and
writes `data/generated/index.npy` + `chunks.json`. Takes ~10 minutes on CPU for the
full corpus; rerun any time to rebuild from scratch.

### Run the server

```bash
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

## Chat API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain definite integrals."}]}'
```

Response includes `answer`, `model`, and `citations` (book/chapter/page per source
chunk used — empty for refused out-of-scope questions).

## Model policy

`GENERATION_PROVIDER` in `backend/.env` selects the answering model. **Embeddings are
always local** (`bge-small-en-v1.5`) regardless of this setting — see
[Tradeoffs](#tradeoffs) for why.

| Value | Model | Needs |
|---|---|---|
| `gemini` (default) | Gemini Flash-Lite | `GOOGLE_API_KEY` (free tier available) |
| `ollama` | `gemma2:2b` (local) | Nothing — [Ollama](https://ollama.com) installed + `ollama pull gemma2:2b` |
| `anthropic` | Claude Haiku | `ANTHROPIC_API_KEY` |

The assignment starter defaults to OpenAI; none of these three do. `ollama` needs no
API key at all, satisfying "local-only is enough" literally. `gemini`/`anthropic` are
there for faster hosted responses with your own key — see [Tradeoffs](#tradeoffs) for
why generation defaults to hosted while embeddings stay local either way.

## Frontend

Optional dark-themed chat UI, no build step:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` (backend must be running). See `frontend/README.md`.

## Evals

With the backend running:

```bash
PYTHONPATH=. python -m evals.run_evals
```

Reports are written to `evals/reports/visible_eval_<timestamp>.json` (gitignored —
auto-generated). Two labeled comparison runs are checked in:
`evals/reports/gemma2_2b_eval.json` and `evals/reports/qwen2.5_3b_eval.json`, from
comparing local models before adding hosted providers (see Tradeoffs).

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

**Generation defaults to a hosted provider (Gemini Flash-Lite), not local Ollama.**
This project started fully local: `gemma2:2b` was picked over `qwen2.5:3b-instruct`
after running the full eval suite against both — Qwen wrote raw LaTeX that renders as
broken text without a LaTeX-aware frontend, and fabricated a specific section-number
citation that didn't appear anywhere in the actual retrieved source text, which is a
real trust problem for a tutor bot. But local generation is genuinely bottlenecked by
consumer GPU hardware (~19-20s per answer, confirmed via direct A/B testing that
swapping embedding provider changes total latency by under 0.5s — generation is
the entire bottleneck). Hosted generation cut that by 8-9x. Gemini Flash-Lite and
Claude Haiku were tested head-to-head through the actual API (same question, same
retrieval, only generation differed): Gemini was ~31% faster (3.85s vs 5.6s), both
equally well-grounded and correctly cited — Haiku was tested as a speed comparison
point, not chosen as the default. Since no API keys are committed to this repo
either way, whoever runs this brings their own key and quota, so the free-tier risk
that affected embeddings during development doesn't carry over to a real deployment
the same way. `ollama`/`gemma2:2b` remains fully wired up as a zero-key, zero-cost
fallback (`GENERATION_PROVIDER=ollama`).

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
  refusal instruction. Red-teamed with ~15 attack attempts (jailbreak/override
  phrasing, roleplay, fake system tags, fake citations, conversation-history
  injection, translation bypass, urgency flooding) — zero full breaches. One
  inconsistent, low-severity finding: fake conversation history occasionally got a
  response to acknowledge an off-topic word by name without ever answering it.
- **Local model ceiling**: if `GENERATION_PROVIDER=ollama`, a 2-3B local model has a
  real reasoning ceiling on genuinely novel multi-step problems, though it performed
  well on the visible eval suite including a partial-fractions integral.

## Worklog

See `WORKLOG.md` for AI tools used and the reasoning behind the key decisions above.
