from __future__ import annotations

import numpy as np

from src.index_store import InMemoryIndexStore
from src.models import TranscriptChunk


def _chunk(chunk_id: str, embedding: list[float]) -> TranscriptChunk:
    return TranscriptChunk(
        video_id='video-1',
        chunk_id=chunk_id,
        start_sec=1.0,
        end_sec=3.0,
        text='hello world',
        embedding=embedding,
    )


def test_in_memory_store_saves_and_loads_transcript_chunks():
    store = InMemoryIndexStore()
    chunks = [_chunk('t1', [1.0, 0.0]), _chunk('t2', [0.0, 1.0])]

    store.save_transcripts('video-1', chunks)

    loaded = store.load_transcripts('video-1')
    assert loaded is not None
    assert [chunk.chunk_id for chunk in loaded] == ['t1', 't2']


def test_in_memory_store_queries_transcripts():
    store = InMemoryIndexStore()
    store.save_transcripts('video-1', [_chunk('t1', [1.0, 0.0]), _chunk('t2', [0.0, 1.0])])

    results = store.query_transcripts(np.array([1.0, 0.0]), top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == 't1'
