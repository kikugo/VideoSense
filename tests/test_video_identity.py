from src.video_processing import compute_video_id_from_bytes


def test_compute_video_id_from_bytes_is_stable():
    payload = b'same-content'
    assert compute_video_id_from_bytes(payload) == compute_video_id_from_bytes(payload)


def test_compute_video_id_from_bytes_changes_with_content():
    a = compute_video_id_from_bytes(b'a')
    b = compute_video_id_from_bytes(b'b')
    assert a != b
