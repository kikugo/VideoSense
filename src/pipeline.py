from __future__ import annotations

import numpy as np

from src.embeddings import embed_frames_with_retry
from src.index_store import IndexStore
from src.models import FrameRecord, SearchResult, TranscriptChunk, UnifiedSearchResult
from src.retrieval import fuse_ranked_results
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
    return store.query_visual(query_vector, top_k=top_k)


async def embed_transcripts_into_store(
    video_id: str,
    chunks: list[TranscriptChunk],
    store: IndexStore,
    embed_text_fn,
) -> list[TranscriptChunk]:
    for chunk in chunks:
        chunk.embedding = await embed_text_fn(chunk.text)
    store.save_transcripts(video_id, chunks)
    return chunks


async def search_library(query: str, store: IndexStore, embed_text_fn, top_k: int, config) -> list[UnifiedSearchResult]:
    query_vector = np.array(await embed_text_fn(query), dtype=float)
    visual = store.query_visual(query_vector, top_k=top_k)
    transcripts = store.query_transcripts(query_vector, top_k=top_k)
    return fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': config.visual_weight, 'transcript': config.transcript_weight},
        rrf_k=config.rrf_k,
    )[:top_k]
