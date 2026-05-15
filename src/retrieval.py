from __future__ import annotations

from src.models import SearchResult, TranscriptSearchResult, UnifiedSearchResult


def _bucket_key(video_id: str, timestamp: float, bucket_sec: float = 1.0) -> tuple[str, int]:
    return video_id, int(timestamp // bucket_sec)


def dedupe_by_time_bucket(results: list[SearchResult], bucket_sec: float = 1.0) -> list[SearchResult]:
    best: dict[tuple[str, int], SearchResult] = {}
    for result in results:
        key = _bucket_key(result.frame.video_id, result.frame.timestamp_sec, bucket_sec)
        current = best.get(key)
        if current is None or result.similarity > current.similarity:
            best[key] = result
    return sorted(best.values(), key=lambda item: item.similarity, reverse=True)


def fuse_ranked_results(
    visual_results: list[SearchResult],
    transcript_results: list[TranscriptSearchResult],
    weights: dict[str, float],
    rrf_k: int,
) -> list[UnifiedSearchResult]:
    fused: dict[tuple[str, int], UnifiedSearchResult] = {}

    for rank, result in enumerate(visual_results, start=1):
        key = _bucket_key(result.frame.video_id, result.frame.timestamp_sec)
        score = weights.get('visual', 1.0) / (rrf_k + rank)
        fused[key] = UnifiedSearchResult(
            video_id=result.frame.video_id,
            start_sec=result.frame.timestamp_sec,
            end_sec=result.frame.timestamp_sec,
            score=score,
            channel='visual',
            frame=result.frame,
        )

    for rank, result in enumerate(transcript_results, start=1):
        key = _bucket_key(result.chunk.video_id, result.chunk.start_sec)
        score = weights.get('transcript', 1.0) / (rrf_k + rank)
        current = fused.get(key)
        if current:
            current.score += score
            current.channel = 'both'
            current.transcript = result.chunk
            current.end_sec = max(current.end_sec, result.chunk.end_sec)
        else:
            fused[key] = UnifiedSearchResult(
                video_id=result.chunk.video_id,
                start_sec=result.chunk.start_sec,
                end_sec=result.chunk.end_sec,
                score=score,
                channel='transcript',
                transcript=result.chunk,
            )

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)
