from __future__ import annotations

import math

import numpy as np

from src.models import FrameRecord, SearchResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError('Vectors must have the same shape.')

    dot_product = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def rank_results(query_embedding: np.ndarray, frames: list[FrameRecord], top_k: int = 3) -> list[SearchResult]:
    scored: list[SearchResult] = []

    for frame in frames:
        if not frame.embedding:
            continue
        similarity = cosine_similarity(query_embedding, np.array(frame.embedding, dtype=float))
        scored.append(SearchResult(frame=frame, similarity=similarity))

    scored.sort(key=lambda item: item.similarity, reverse=True)
    return scored[:top_k]


def format_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(math.floor(seconds)))
    minutes = seconds_int // 60
    remainder = seconds_int % 60
    return f'{minutes}:{remainder:02d}'
