import base64

import numpy as np

from src.video_processing import build_sample_timestamps, frame_to_jpeg_base64, compute_video_id


def test_build_sample_timestamps_uses_interval_and_includes_zero():
    timestamps = build_sample_timestamps(duration_sec=6.1, interval_sec=2.0)
    assert timestamps == [0.0, 2.0, 4.0, 6.0]


def test_build_sample_timestamps_handles_short_video():
    timestamps = build_sample_timestamps(duration_sec=0.6, interval_sec=2.0)
    assert timestamps == [0.0]


def test_frame_to_jpeg_base64_returns_decodable_string():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded = frame_to_jpeg_base64(frame)

    decoded = base64.b64decode(encoded)
    assert decoded[:2] == b'\xff\xd8'


def test_compute_video_id_is_deterministic_for_same_path():
    a = compute_video_id('/tmp/video-a.mp4')
    b = compute_video_id('/tmp/video-a.mp4')
    c = compute_video_id('/tmp/video-b.mp4')

    assert a == b
    assert a != c
