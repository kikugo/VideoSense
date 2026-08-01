from __future__ import annotations

from dataclasses import dataclass

from src.models import FrameRecord, SearchResult, TranscriptChunk, TranscriptSearchResult, match_strength
from src.retrieval import (
    dedupe_by_time_bucket,
    fuse_ranked_results,
    select_with_modality_coverage,
)


@dataclass
class _StaleResult:
    """Mimics a result built by an older src package that predates `similarity`."""

    video_id: str
    score: float


def test_match_strength_reports_unknown_rather_than_zero_for_stale_objects():
    """A stale deploy must not be indistinguishable from a zero-similarity match.

    Streamlit can run a new app.py against an already-imported older src package.
    Returning 0.0 here would filter every result out, which looks exactly like the
    bug this field was added to fix; None lets callers keep the result instead.
    """
    assert match_strength(_StaleResult(video_id='v1', score=0.019)) is None


def test_match_strength_reads_similarity_when_present():
    visual = [SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.42)]

    fused = fuse_ranked_results(
        visual_results=visual, transcript_results=[], weights={'visual': 1.0}, rrf_k=60
    )

    assert match_strength(fused[0]) == 0.42


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


def test_frame_merges_with_the_transcript_chunk_spoken_over_it():
    """A frame at 42s and the chunk covering 38-48s are one moment, not two.

    Frames bucket per second, chunks run about ten seconds, so before span-aware
    merging these landed in different buckets and competed for a slot instead of
    producing a single result carrying both a thumbnail and the spoken line.
    """
    visual = [SearchResult(FrameRecord('v1', 'f1', 42.1, 'thumb', [1.0]), 0.46)]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', 't1', 38.0, 48.0, 'spoken', [1.0]), 0.62)
    ]

    fused = fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15},
        rrf_k=60,
    )

    assert len(fused) == 1
    assert fused[0].channel == 'both'
    assert fused[0].frame is not None
    assert fused[0].transcript is not None
    assert fused[0].similarity == 0.62


def test_transcripts_cannot_shut_visuals_out_of_the_results():
    """With transcript_weight > visual_weight the worst transcript outranks the
    best frame (1.15/63 > 1.0/61), so the top-k was always entirely transcript.
    """
    visual = [
        SearchResult(FrameRecord('v1', f'f{i}', 100.0 + i, 'thumb', [1.0]), 0.45)
        for i in range(3)
    ]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', f't{i}', i * 10.0, i * 10.0 + 5.0, 'spoken', [1.0]), 0.60)
        for i in range(3)
    ]

    fused = fuse_ranked_results(
        visual_results=visual,
        transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15},
        rrf_k=60,
    )
    assert all(item.frame is None for item in fused[:3]), 'precondition: transcripts block'

    covered = select_with_modality_coverage(fused, 3)

    assert len(covered) == 3
    assert any(item.frame is not None for item in covered), 'a visual hit must survive'
    assert covered[0] is fused[0], 'the top result keeps its rank'


def test_modality_coverage_leaves_already_mixed_results_alone():
    visual = [SearchResult(FrameRecord('v1', 'f1', 42.1, 'thumb', [1.0]), 0.46)]
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', 't1', 38.0, 48.0, 'spoken', [1.0]), 0.62)
    ]
    fused = fuse_ranked_results(
        visual_results=visual, transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15}, rrf_k=60,
    )

    assert select_with_modality_coverage(fused, 3) == fused


def test_modality_coverage_is_a_noop_when_nothing_visual_matched():
    transcripts = [
        TranscriptSearchResult(TranscriptChunk('v1', f't{i}', i * 10.0, i * 10.0 + 5.0, 'spoken', [1.0]), 0.6)
        for i in range(4)
    ]
    fused = fuse_ranked_results(
        visual_results=[], transcript_results=transcripts,
        weights={'visual': 1.0, 'transcript': 1.15}, rrf_k=60,
    )

    covered = select_with_modality_coverage(fused, 3)

    assert len(covered) == 3
    assert all(item.frame is None for item in covered)


def test_dedupe_by_time_bucket_keeps_best_score():
    visual = [
        SearchResult(FrameRecord('v1', 'f1', 10.0, 'thumb', [1.0]), 0.6),
        SearchResult(FrameRecord('v1', 'f2', 10.4, 'thumb', [1.0]), 0.9),
    ]

    deduped = dedupe_by_time_bucket(visual, bucket_sec=1.0)

    assert len(deduped) == 1
    assert deduped[0].frame.frame_id == 'f2'
