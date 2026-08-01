# VideoSense

[![tests](https://github.com/kikugo/VideoSense/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kikugo/VideoSense/actions/workflows/tests.yml)
[![qdrant-keepalive](https://github.com/kikugo/VideoSense/actions/workflows/qdrant-keepalive.yml/badge.svg)](https://github.com/kikugo/VideoSense/actions/workflows/qdrant-keepalive.yml)

[Live demo](https://videosense.streamlit.app) — hit **Load sample video** and search it. No upload or signup needed.

VideoSense is a Python app for semantic video search.

The bundled clip is NASA's Artemis I recap (public domain, `examples/sample/`). It was chosen because it has narration for the transcript channel and visually distinct scenes, so "rocket on the launch pad", "mission control room" and "someone speaking at a podium" return genuinely different moments rather than three views of the same shot. Because the vector backend persists, the first visitor to load it pays the indexing cost and everyone after reuses the index by content hash.

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

## A bug worth writing down

Ranking fuses the visual and transcript channels with Reciprocal Rank Fusion. RRF scores are rank-based, so at `rrf_k=60` they sit around 0.016–0.033 no matter how good the match is — they order results, they do not measure them.

The UI was filtering on that fused score against the "Minimum score" slider, which defaults to **0.30**. Nothing can clear 0.30 on a scale that tops out near 0.03, so **every search returned "No strong matches found"** while retrieval underneath was working fine: raw cosine similarities for the same queries ran 0.33–0.73. The displayed match strength had the same problem and read "2%" for every hit.

The fix keeps RRF for ordering but carries the best raw cosine through fusion as a separate `similarity` field, and filters and displays on that. Two regression tests pin it, and both fail if the plumbing is removed.
- Use scene-aware frame selection to reduce visual embedding waste.

## Storage and the keepalive

Qdrant Cloud is the primary index. Chroma is the fallback, and an in-memory store is
the last resort, chosen at startup in `src/store_factory.py` by a health check.

The free tier suspends a cluster after a period of inactivity, and a read-only health
check does not reset that timer. A cluster here was suspended in June 2026 while every
read ping came back green. A suspended cluster answers 404 on every path, so it also
does not look like an outage.

The fix is a scheduled write: a cron upserts and deletes a point twice a week, which
does reset the timer. GitHub disables a cron after 60 days without repository
activity, so a second job re-enables it. If the cluster is unreachable anyway, the app
falls back to Chroma rather than failing.

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

## Deploy (Streamlit Community Cloud)

The app runs on [Streamlit Community Cloud](https://share.streamlit.io): point
it at this repo (`main`, main file `app.py`) and pick **Python 3.12** under
*Advanced settings* (3.14 lacks wheels for some deps).

- **System packages** come from `packages.txt` (`ffmpeg` for audio extraction,
  `libgl1` for OpenCV). `requirements.txt` uses `opencv-python-headless` so
  OpenCV runs on the headless server.
- **Secrets**: add them in the app's *Settings → Secrets* editor using
  `.streamlit/secrets.toml.example` as the template. `src/runtime_env.py`
  bridges `st.secrets` into the environment that `AppConfig.from_env()` reads,
  and real env vars always win over secrets so local runs are unaffected.
- At minimum set `GEMINI_API_KEY`. For the persistent vector backend, also set
  `VIDEOSENSE_QDRANT_URL` and `VIDEOSENSE_QDRANT_API_KEY`; without them the app
  falls back to Chroma/in-memory automatically.

## Vector backends

The persistent store is pluggable via `VIDEOSENSE_VECTOR_BACKEND`
(`auto` | `qdrant` | `chroma` | `memory`). All backends sit behind one
`IndexStore` interface, so the app-side hybrid (visual + transcript) fusion is
unchanged:

- **memory**: no persistence; the index lives only for the session.
- **chroma**: local ChromaDB under `.videosense/chroma/` (set `VIDEOSENSE_ENABLE_PERSISTENCE=true`).
- **qdrant**: Qdrant, local or cloud (set `VIDEOSENSE_QDRANT_URL`, plus `VIDEOSENSE_QDRANT_API_KEY` for Qdrant Cloud).
- **auto** (default): Qdrant when a URL is set and reachable, otherwise Chroma (if persistence is on), otherwise in-memory.

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
3. Free clusters suspend after about a week of no activity. The
   `qdrant-keepalive` GitHub Action keeps one warm by writing and then deleting
   a throwaway point twice a week (a read-only ping doesn't reset the idle
   timer). Add repo secrets `QDRANT_URL` and `QDRANT_API_KEY` to enable it.
   If the ping keeps failing, the action opens a GitHub issue with the
   diagnosis; a push to `main` re-enables the cron if GitHub disabled it
   after a quiet spell.

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
