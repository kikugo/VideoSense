from __future__ import annotations

from scenedetect import ContentDetector, detect

from src.video_processing import build_sample_timestamps


def detect_scene_timestamps(video_path: str, threshold: float, max_frames: int) -> list[float]:
    scenes = detect(video_path, ContentDetector(threshold=threshold))
    timestamps: list[float] = []
    for start_time, end_time in scenes:
        midpoint = (start_time.get_seconds() + end_time.get_seconds()) / 2.0
        timestamps.append(round(midpoint, 3))
    return timestamps[:max_frames]


def select_visual_timestamps(
    video_path: str,
    duration_sec: float,
    strategy: str,
    interval_sec: float,
    threshold: float,
    max_frames: int,
) -> list[float]:
    if strategy == 'scene':
        try:
            timestamps = detect_scene_timestamps(
                video_path=video_path,
                threshold=threshold,
                max_frames=max_frames,
            )
            if timestamps:
                return timestamps
        except Exception:
            pass

    return build_sample_timestamps(duration_sec=duration_sec, interval_sec=interval_sec)[:max_frames]
