# VideoSense

VideoSense is a Python app for semantic video search.

## Features

- Upload one or many videos and index sampled frames.
- Generate image and text embeddings with Gemini.
- Search moments with natural language.
- Search both visual moments and spoken transcript chunks.
- Jump to matching timestamps in the video player.
- Pluggable vector backend: in-memory, ChromaDB, or Qdrant (local or Qdrant Cloud) with graceful fallback.
- Reuse previously indexed videos via content-based identity.
- See best match per video for broad library queries.
- Filter weak matches with a configurable minimum similarity threshold.
- Use scene-aware frame selection to reduce visual embedding waste.

## Local setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Install dependencies:
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```
3. Create local env file:
   ```bash
   cp .env.example .env
   ```
4. Set `GEMINI_API_KEY` in `.env`.

For transcript indexing, install `ffmpeg` locally so VideoSense can extract audio.

## Run

```bash
.venv/bin/streamlit run app.py
```

## Vector backends

The persistent store is pluggable via `VIDEOSENSE_VECTOR_BACKEND`
(`auto` | `qdrant` | `chroma` | `memory`). All backends sit behind one
`IndexStore` interface, so the app-side hybrid (visual + transcript) fusion is
unchanged:

- **memory** — no persistence; the index lives only for the session.
- **chroma** — local ChromaDB under `.videosense/chroma/` (set `VIDEOSENSE_ENABLE_PERSISTENCE=true`).
- **qdrant** — Qdrant, local or cloud (set `VIDEOSENSE_QDRANT_URL`, plus `VIDEOSENSE_QDRANT_API_KEY` for Qdrant Cloud).
- **auto** (default) — Qdrant when a URL is set and reachable, otherwise Chroma (if persistence is on), otherwise in-memory.

If Qdrant is configured but unreachable (e.g. a suspended free-tier cluster),
the app **falls back** to Chroma/in-memory via a startup health check, so it
never hard-fails.

### Local Qdrant

```bash
docker compose up -d qdrant
# then in .env:
#   VIDEOSENSE_VECTOR_BACKEND=qdrant
#   VIDEOSENSE_QDRANT_URL=http://localhost:6333
```

### Qdrant Cloud (free tier)

1. Create a free cluster at https://cloud.qdrant.io and copy its URL + API key.
2. Set `VIDEOSENSE_QDRANT_URL` and `VIDEOSENSE_QDRANT_API_KEY` in `.env`.
3. Free clusters suspend after ~1 week idle. The `qdrant-keepalive` GitHub
   Action pings the cluster twice a week — add repo secrets `QDRANT_URL` and
   `QDRANT_API_KEY` to enable it.

## Local Data

VideoSense stores indexed app data under `.videosense/`:

- `.videosense/videos/` for durable uploaded video files
- `.videosense/audio/` for extracted audio
- `.videosense/library.json` for indexed video metadata
- `.videosense/chroma/` when the ChromaDB backend is enabled
- `.videosense/qdrant/` when running local Qdrant via docker compose

## Tests

```bash
.venv/bin/pytest -q
```
