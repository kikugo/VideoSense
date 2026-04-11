from __future__ import annotations

import numpy as np

from src.embeddings import embed_frames_with_retry
from src.index_store import IndexStore
from src.models import FrameRecord, SearchResult
from src.search import rank_results


async def index_frames_into_store(
    video_id: str,
    frames: list[FrameRecord],
    store: IndexStore,
    embed_image_fn,
    concurrency: int,
    max_retries: int,
    base_backoff_sec: float,
    on_progress=None,
) -> tuple[list[FrameRecord], list[str]]:
    indexed_frames, failures = await embed_frames_with_retry(
        frames=frames,
        embed_image_fn=embed_image_fn,
        concurrency=concurrency,
        max_retries=max_retries,
        base_backoff_sec=base_backoff_sec,
        on_progress=on_progress,
    )
    store.save(video_id, indexed_frames)
    return indexed_frames, failures


async def search_frames(query: str, store: IndexStore, embed_text_fn, top_k: int) -> list[SearchResult]:
    query_vector = np.array(await embed_text_fn(query), dtype=float)
    frames = store.load_all()
    return rank_results(query_vector, frames, top_k=top_k)
