from __future__ import annotations

from src.app_state import demo_rejection_reason, ensure_defaults, set_playback_target


def test_ensure_defaults_populates_required_keys():
    state = {}

    ensure_defaults(state)

    assert state['video_name_by_id'] == {}
    assert state['video_path_by_id'] == {}
    assert state['search_results'] == []
    assert state['playback_start_time'] == 0


def test_demo_rejection_reason_allows_upload_within_limits():
    assert demo_rejection_reason(
        indexed_count=1, upload_bytes=50 * 1024 * 1024, max_videos=3, max_upload_mb=200
    ) is None


def test_demo_rejection_reason_blocks_when_session_quota_reached():
    reason = demo_rejection_reason(
        indexed_count=3, upload_bytes=1024, max_videos=3, max_upload_mb=200
    )

    assert reason is not None
    assert '3' in reason


def test_demo_rejection_reason_blocks_oversized_file():
    reason = demo_rejection_reason(
        indexed_count=0, upload_bytes=300 * 1024 * 1024, max_videos=3, max_upload_mb=200
    )

    assert reason is not None
    assert '200' in reason


def test_demo_rejection_reason_reports_size_before_quota():
    # an oversized file is the more specific complaint, so it wins
    reason = demo_rejection_reason(
        indexed_count=5, upload_bytes=300 * 1024 * 1024, max_videos=3, max_upload_mb=200
    )

    assert '200 MB' in reason


def test_set_playback_target_updates_video_and_time():
    state = {}
    ensure_defaults(state)

    set_playback_target(state, video_id='video-1', start_time=12.8)

    assert state['playback_video_id'] == 'video-1'
    assert state['playback_start_time'] == 12
