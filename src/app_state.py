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


def demo_rejection_reason(
    indexed_count: int,
    upload_bytes: int,
    max_videos: int,
    max_upload_mb: int,
) -> str | None:
    """Why this upload should be refused on the public demo, or None to allow it.

    Indexing holds frames in memory, and the hosted demo shares a small container,
    so one oversized upload degrades the app for everyone. Size is checked before
    the session quota because it is the more specific complaint.
    """
    upload_mb = upload_bytes / (1024 * 1024)
    if upload_mb > max_upload_mb:
        return (
            f'Demo limit: {max_upload_mb} MB per video, this one is {upload_mb:.0f} MB. '
            'Clone the repo to run it without limits.'
        )
    if indexed_count >= max_videos:
        return (
            f'Demo limit: {max_videos} videos per session. '
            'Clone the repo to run it without limits.'
        )
    return None


def set_playback_target(state: MutableMapping, video_id: str, start_time: float) -> None:
    state['playback_video_id'] = video_id
    state['playback_start_time'] = int(start_time)
