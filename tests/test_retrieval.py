from __future__ import annotations

from src.models import FrameRecord, SearchResult, TranscriptChunk, TranscriptSearchResult
from src.retrieval import dedupe_by_time_bucket, fuse_ranked_results


def test_fuse_ranked_results_merges_visual_and_transcript_hits():
    visual = [
        SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.8),
        SearchResult(FrameRecord('v2', 'f2', 20.0, 'thumb', [1.0]), 0.7),
    ]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', 't1', 10.2, 12.0, 'spoken', [1.0]), 0.9)
    ]

    fused = fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15},
        rrf_k=60,
    )

    assert fused[0].video_id == 'v1'
    assert fused[0].channel == 'both'
    assert fused[0].start_sec == 10.0


def test_dedupe_by_time_bucket_keeps_best_score():
    visual = [
        SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.6),
        SearchResult(FrameRecord('v1', 'f2', 10.4, 'thumb', [1.0]), 0.9),
    ]

    deduped = dedupe_by_time_bucket(visual, bucket_sec=1.0)

    assert len(deduped) == 1
    assert deduped[0].frame.frame_id == 'f2'
