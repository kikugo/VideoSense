# VideoSense

VideoSense is a Python app for semantic video search.

## Features

- Upload one or many videos and index sampled frames.
- Generate image and text embeddings with Gemini.
- Search moments with natural language.
- Search both visual moments and spoken transcript chunks.
- Jump to matching timestamps in the video player.
- Optional persistent index storage with ChromaDB.
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

## Local Data

VideoSense stores indexed app data under `.videosense/`:

- `.videosense/videos/` for durable uploaded video files
- `.videosense/audio/` for extracted audio
- `.videosense/library.json` for indexed video metadata
- `.videosense/chroma/` when persistent ChromaDB is enabled

## Tests

```bash
.venv/bin/pytest -q
```
