# Worklog

## AI tools used

Built this with Claude Code (Anthropic), used for planning, writing the code, and
running/debugging things locally (ingestion runs, model benchmarks, browser testing
of the frontend). I made the calls on architecture and tradeoffs; Claude did the
implementation and a lot of the testing legwork.

## What I built

A RAG tutor over the Class 12 NCERT Math/Physics/Chemistry books. Pipeline is:
extract text from the PDFs, chunk it, embed the chunks, store them in a local index.
At query time, embed the question, pull the closest chunks, feed them to a model to
answer from, return the answer with citations. If nothing relevant comes back above
a similarity threshold, it refuses instead of guessing.

## Key decisions

**Embeddings stay local, not hosted.** I tried Google's embedding API first since the
starter leans hosted-model-friendly. Kept hitting quota walls on the free tier —
inconsistent, sometimes failed on the very first request, sometimes got through a few
hundred chunks before stopping. Rather than fight that, I switched to a small local
embedding model (bge-small) and actually benchmarked it against Gemini's embeddings on
the same data, three separate times at increasing scale. Local won every time on the
thing that actually matters here — how cleanly it separates real subject questions
from off-topic ones. Not a compromise, genuinely the better option for this.

**Generation is hosted, not local — but swappable.** Started fully local with Ollama
(gemma2:2b), picked over a slightly bigger local model after running the eval set
against both — the other one wrote LaTeX that would look broken without a proper
renderer, and it once cited a section number that doesn't exist in the actual source
text, which is a bad trait for something answering NCERT questions. But local
generation on my hardware is slow, ~20 seconds an answer. Timed it properly and hosted
models (tried both Gemini and Claude Haiku) come back in a few seconds instead. Since
the assignment says using another model is fine as long as the tradeoff's explained,
I built it so the generation backend is just a config switch — Gemini by default, but
Ollama and Haiku both work if you set the env var. No API keys are committed anywhere,
you use your own.

**No vector database.** At this size (under 2000 chunks) a numpy array and a brute
force similarity search is plenty fast. Didn't see a reason to add a dependency for
that.

**Out-of-scope handling has two layers.** A similarity threshold catches clearly
unrelated questions before any generation call happens. A system prompt instruction
catches the borderline cases where retrieval finds something loosely related but the
actual question still can't be honestly answered from it — model says so instead of
making something up. Tried to break this with a bunch of prompt injection and jailbreak
attempts (fake system messages, fake conversation history claiming permission was
granted, smuggling off-topic requests inside physics-sounding wording, asking it to
ignore its instructions outright). Held up in every case I tried.

**Frontend is plain HTML/CSS/JS, no build step.** Wanted something that looks decent
without adding a whole toolchain for a bonus feature. Added KaTeX for math rendering
since the hosted models write real LaTeX in their answers.

## What I'd do differently with more time

- Citations only go down to chapter/page, not the specific section. Would need either
  parsing the NCERT table of contents or a separate labeling pass.
- The out-of-scope threshold is tuned from observed scores on my test questions, not
  a large labeled set. Works well on everything I tried, but isn't mathematically
  guaranteed.
- Frontend markdown rendering is basic — headers and math work now, but nothing fancier.
