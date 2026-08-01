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


def _visual_inside_span(
    fused: dict[tuple[str, int], UnifiedSearchResult],
    chunk,
) -> UnifiedSearchResult | None:
    """Best visual hit whose frame falls inside a transcript chunk's span.

    A frame and the chunk that was being spoken over it are the same moment, but
    they bucket to different keys: frames are bucketed per second while chunks
    run about ten seconds. Without this lookup the two compete for a slot instead
    of merging, so `channel='both'` almost never happened and one modality
    crowded the other out of the results.
    """
    candidates = [
        entry
        for entry in fused.values()
        if entry.channel == 'visual'
        and entry.video_id == chunk.video_id
        and chunk.start_sec <= entry.start_sec <= chunk.end_sec
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.similarity)


def select_with_modality_coverage(
    results: list[UnifiedSearchResult], limit: int
) -> list[UnifiedSearchResult]:
    """Take the top `limit`, but do not let one channel shut the other out.

    RRF scores are rank-based and the channel weights multiply them, so a
    transcript weight above the visual weight makes the *worst* transcript
    outrank the *best* frame (1.15/63 > 1.0/61). The top-k was therefore always
    entirely transcript whenever the transcript channel returned enough hits,
    which is a poor showing for a search tool whose headline is visual search.

    Ranking is otherwise untouched: this only swaps the lowest-ranked selected
    result for the highest-ranked one carrying the missing modality.
    """
    selected = results[:limit]
    if limit <= 1 or not selected:
        return selected

    def has_frame(item: UnifiedSearchResult) -> bool:
        return item.frame is not None

    if any(has_frame(item) for item in selected):
        return selected

    promotion = next((item for item in results[limit:] if has_frame(item)), None)
    if promotion is None:
        return selected

    return selected[:-1] + [promotion]


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
            similarity=result.similarity,
            frame=result.frame,
        )

    for rank, result in enumerate(transcript_results, start=1):
        key = _bucket_key(result.chunk.video_id, result.chunk.start_sec)
        score = weights.get('transcript', 1.0) / (rrf_k + rank)
        # Prefer a frame sitting inside this chunk's span over an exact bucket
        # hit: they are the same moment, and merging them yields one result with
        # both a thumbnail and the spoken line.
        current = _visual_inside_span(fused, result.chunk) or fused.get(key)
        if current:
            current.score += score
            current.channel = 'both'
            current.transcript = result.chunk
            current.end_sec = max(current.end_sec, result.chunk.end_sec)
            # A moment matched by both channels keeps the stronger evidence.
            current.similarity = max(current.similarity, result.similarity)
        else:
            fused[key] = UnifiedSearchResult(
                video_id=result.chunk.video_id,
                start_sec=result.chunk.start_sec,
                end_sec=result.chunk.end_sec,
                score=score,
                channel='transcript',
                similarity=result.similarity,
                transcript=result.chunk,
            )

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)
