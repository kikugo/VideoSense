from __future__ import annotations

import asyncio

import numpy as np

from src.index_store import InMemoryIndexStore
from src.models import FrameRecord
from src.pipeline import index_frames_into_store, search_frames


class FakeEmbedder:
    async def embed_image_base64(self, image_b64: str) -> list[float]:
        return [len(image_b64), 1.0]

    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


async def _run_index_case() -> None:
    store = InMemoryIndexStore()
    embedder = FakeEmbedder()

    frames = [
        FrameRecord('v1', 'f1', 0.0, 'aaa', None),
        FrameRecord('v1', 'f2', 2.0, 'aaaa', None),
    ]

    indexed_frames, failures = await index_frames_into_store(
        video_id='v1',
        frames=frames,
        store=store,
        embed_image_fn=embedder.embed_image_base64,
        concurrency=2,
        max_retries=1,
        base_backoff_sec=0.0,
    )

    assert failures == []
    assert indexed_frames[0].embedding is not None
    assert store.load('v1') is not None


async def _run_search_case() -> None:
    store = InMemoryIndexStore()
    embedder = FakeEmbedder()

    store.save(
        'v1',
        [
            FrameRecord('v1', 'f1', 0.0, 'aaaaaa', [6.0, 1.0]),
            FrameRecord('v1', 'f2', 2.0, 'aa', [2.0, 1.0]),
        ],
    )

    results = await search_frames(
        query='aaaaa',
        store=store,
        embed_text_fn=embedder.embed_text,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].frame.frame_id == 'f1'
    assert np.isfinite(results[0].similarity)


def test_index_frames_into_store_embeds_and_saves() -> None:
    asyncio.run(_run_index_case())


def test_search_frames_returns_ranked_results() -> None:
    asyncio.run(_run_search_case())
