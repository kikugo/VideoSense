from src.models import FrameRecord, SearchResult
from src.search import group_top_result_per_video


def test_group_top_result_per_video_keeps_best_match_per_video():
    results = [
        SearchResult(FrameRecord('v1', 'a', 0.0, 'x', [0.1]), 0.90),
        SearchResult(FrameRecord('v1', 'b', 2.0, 'x', [0.1]), 0.80),
        SearchResult(FrameRecord('v2', 'c', 1.0, 'x', [0.1]), 0.85),
    ]

    grouped = group_top_result_per_video(results)

    assert len(grouped) == 2
    assert grouped[0].frame.video_id == 'v1'
    assert grouped[1].frame.video_id == 'v2'
