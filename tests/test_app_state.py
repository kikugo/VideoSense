from __future__ import annotations

from src.app_state import ensure_defaults, set_playback_target


def test_ensure_defaults_populates_required_keys():
    state = {}

    ensure_defaults(state)

    assert state['video_name_by_id'] == {}
    assert state['video_path_by_id'] == {}
    assert state['search_results'] == []
    assert state['playback_start_time'] == 0


def test_set_playback_target_updates_video_and_time():
    state = {}
    ensure_defaults(state)

    set_playback_target(state, video_id='video-1', start_time=12.8)

    assert state['playback_video_id'] == 'video-1'
    assert state['playback_start_time'] == 12
