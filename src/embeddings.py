from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from src.models import FrameRecord


async def embed_frames_with_retry(
    frames: list[FrameRecord],
    embed_image_fn: Callable[[str], Awaitable[list[float]]],
    concurrency: int,
    max_retries: int,
    base_backoff_sec: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[FrameRecord], list[str]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    failures: list[str] = []
    total = len(frames)
    done = 0
    lock = asyncio.Lock()

    async def _embed_one(frame: FrameRecord) -> None:
        nonlocal done
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    frame.embedding = await embed_image_fn(frame.thumbnail_b64)
                    break
                except Exception:
                    if attempt >= max_retries:
                        failures.append(frame.frame_id)
                        frame.embedding = None
                        break
                    wait_for = base_backoff_sec * (2**attempt)
                    if wait_for > 0:
                        await asyncio.sleep(wait_for)

            async with lock:
                done += 1
                if on_progress:
                    on_progress(done, total)

    await asyncio.gather(*[_embed_one(frame) for frame in frames])
    return frames, failures


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = 'gemini-embedding-2-preview') -> None:
        if not api_key:
            raise ValueError('GEMINI_API_KEY is required.')

        self.model = model
        from google import genai  # Lazy import keeps tests lightweight.

        self._client = genai.Client(api_key=api_key)

    async def embed_text(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_text_sync, text)

    async def embed_image_base64(self, image_b64: str) -> list[float]:
        return await asyncio.to_thread(self._embed_image_sync, image_b64)

    def _embed_text_sync(self, text: str) -> list[float]:
        response = self._client.models.embed_content(model=self.model, contents=text)
        return list(response.embeddings[0].values)

    def _embed_image_sync(self, image_b64: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=[
                {
                    'inlineData': {
                        'data': image_b64,
                        'mimeType': 'image/jpeg',
                    }
                }
            ],
        )
        return list(response.embeddings[0].values)
