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


def test_fused_results_carry_raw_similarity_not_just_rrf_score():
    """The UI filters on a 0-1 similarity threshold, so fusion must expose one.

    RRF rank scores sit near 0.016-0.033 at rrf_k=60. Filtering those against the
    default 0.30 threshold discarded every result, so search returned "no strong
    matches" for every query even though retrieval was working.
    """
    visual = [SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.46)]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v2', 't1', 30.0, 32.0, 'spoken', [1.0]), 0.62)
    ]

    fused = fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15},
        rrf_k=60,
    )

    by_video = {item.video_id: item for item in fused}
    assert by_video['v1'].similarity == 0.46
    assert by_video['v2'].similarity == 0.62
    # The RRF score is still what orders the list, and is on its own small scale.
    assert all(item.score < 0.05 for item in fused)
    # A realistic threshold must keep these rather than drop them.
    assert [item for item in fused if item.similarity >= 0.30] == fused


def test_fused_similarity_takes_the_stronger_channel_when_both_match():
    visual = [SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.40)]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', 't1', 10.2, 12.0, 'spoken', [1.0]), 0.75)
    ]

    fused = fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15},
        rrf_k=60,
    )

    assert fused[0].channel == 'both'
    assert fused[0].similarity == 0.75


def test_dedupe_by_time_bucket_keeps_best_score():
    visual = [
        SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.6),
        SearchResult(FrameRecord('v1', 'f2', 10.4, 'thumb', [1.0]), 0.9),
    ]

    deduped = dedupe_by_time_bucket(visual, bucket_sec=1.0)

    assert len(deduped) == 1
    assert deduped[0].frame.frame_id == 'f2'
