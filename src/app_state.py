from __future__ import annotations

from collections.abc import MutableMapping


def ensure_defaults(state: MutableMapping) -> None:
    state.setdefault('active_video_path', None)
    state.setdefault('active_video_id', None)
    state.setdefault('active_video_name', None)
    state.setdefault('video_name_by_id', {})
    state.setdefault('video_path_by_id', {})
    state.setdefault('search_results', [])
    state.setdefault('last_index_failures', [])
    state.setdefault('playback_video_id', None)
    state.setdefault('playback_start_time', 0)


def set_playback_target(state: MutableMapping, video_id: str, start_time: float) -> None:
    state['playback_video_id'] = video_id
    state['playback_start_time'] = int(start_time)
