from __future__ import annotations

from src.scenes import detect_scene_timestamps, select_visual_timestamps


def test_select_visual_timestamps_prefers_scene_strategy(monkeypatch):
    monkeypatch.setattr('src.scenes.detect_scene_timestamps', lambda **_: [1.0, 4.0])

    timestamps = select_visual_timestamps(
        video_path='video.mp4',
        duration_sec=10.0,
        strategy='scene',
        interval_sec=2.0,
        threshold=27.0,
        max_frames=10,
    )

    assert timestamps == [1.0, 4.0]


def test_select_visual_timestamps_falls_back_to_interval(monkeypatch):
    def fail_detection(**_):
        raise RuntimeError('scene detection failed')

    monkeypatch.setattr('src.scenes.detect_scene_timestamps', fail_detection)

    timestamps = select_visual_timestamps(
        video_path='video.mp4',
        duration_sec=5.0,
        strategy='scene',
        interval_sec=2.0,
        threshold=27.0,
        max_frames=10,
    )

    assert timestamps == [0.0, 2.0, 4.0]


def test_detect_scene_timestamps_limits_results(monkeypatch):
    class FakeScene:
        def __init__(self, start: float, end: float):
            self._start = start
            self._end = end

        def get_seconds(self):
            return self._start

    scenes = [(FakeScene(0.0, 0.0), FakeScene(4.0, 4.0)), (FakeScene(4.0, 4.0), FakeScene(10.0, 10.0))]
    monkeypatch.setattr('src.scenes.detect', lambda *_args, **_kwargs: scenes)

    timestamps = detect_scene_timestamps(video_path='video.mp4', threshold=27.0, max_frames=1)

    assert timestamps == [2.0]
