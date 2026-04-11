from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.video_processing import compute_video_id, extract_sampled_frames


def _make_test_video(path: Path, frame_count: int = 10, fps: float = 1.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    if not writer.isOpened():
        raise RuntimeError('Could not create test video.')

    for i in range(frame_count):
        frame = np.full((64, 64, 3), fill_value=i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_sampled_frames_returns_interval_records(tmp_path: Path):
    video_path = tmp_path / 'sample.avi'
    _make_test_video(video_path, frame_count=10, fps=1.0)

    frames = extract_sampled_frames(
        video_path=str(video_path),
        video_id=compute_video_id(str(video_path)),
        interval_sec=2.0,
    )

    assert len(frames) == 5
    assert frames[0].timestamp_sec == 0.0
    assert frames[1].timestamp_sec == 2.0
    assert all(item.thumbnail_b64 for item in frames)
    assert all(item.embedding is None for item in frames)
