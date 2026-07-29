# Frontend

A dark-themed chat UI for the tutor. Plain HTML/CSS/JS — no build step, no npm
dependencies. Talks to the backend's `POST /chat`.

## Run it

With the backend running on `http://localhost:8000` (see the root README), serve this
folder with any static server, e.g.:

```bash
python -m http.server 5500 --directory frontend
```

Then open `http://localhost:5500`.

You can also open `frontend/index.html` directly in a browser (`file://`) — CORS is
already configured on the backend to allow this.

## Notes

- `API_URL` in `app.js` is hardcoded to `http://localhost:8000/chat`; change it if you
  run the backend on a different port.
- The backend returns the full answer in one response (no streaming endpoint), so the
  UI simulates a token-by-token reveal client-side for better perceived responsiveness.
- Citations render as pills under each answer (book · chapter · page), deduplicated by
  source since overlapping chunks can retrieve the same page twice.
- Math (`$...$`, `$$...$$`) renders via [KaTeX](https://katex.org), loaded from a CDN
  — the only network dependency in an otherwise build-step-free frontend. Degrades
  gracefully if unreachable: raw LaTeX text shows instead of a broken page.
