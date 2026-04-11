import numpy as np
import pytest

from src.search import cosine_similarity, rank_results, format_timestamp
from src.models import FrameRecord


def test_cosine_similarity_returns_expected_value():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_norm():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0


def test_rank_results_sorts_and_limits_top_k():
    query = np.array([1.0, 0.0])
    frames = [
        FrameRecord(video_id='v1', frame_id='f1', timestamp_sec=0.0, thumbnail_b64='x', embedding=[1.0, 0.0]),
        FrameRecord(video_id='v1', frame_id='f2', timestamp_sec=2.0, thumbnail_b64='x', embedding=[0.8, 0.2]),
        FrameRecord(video_id='v1', frame_id='f3', timestamp_sec=4.0, thumbnail_b64='x', embedding=[0.1, 0.9]),
    ]

    results = rank_results(query, frames, top_k=2)

    assert len(results) == 2
    assert results[0].frame.frame_id == 'f1'
    assert results[1].frame.frame_id == 'f2'
    assert results[0].similarity >= results[1].similarity


def test_format_timestamp_renders_minutes_and_seconds():
    assert format_timestamp(0) == '0:00'
    assert format_timestamp(65) == '1:05'
