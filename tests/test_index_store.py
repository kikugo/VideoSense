from src.index_store import InMemoryIndexStore
from src.models import FrameRecord


def _sample_frame(frame_id: str) -> FrameRecord:
    return FrameRecord(
        video_id='video-1',
        frame_id=frame_id,
        timestamp_sec=2.0,
        thumbnail_b64='abc',
        embedding=[0.1, 0.2],
    )


def test_in_memory_store_saves_and_loads_frames():
    store = InMemoryIndexStore()
    frames = [_sample_frame('f1'), _sample_frame('f2')]

    store.save('video-1', frames)

    loaded = store.load('video-1')
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].frame_id == 'f1'


def test_in_memory_store_returns_all_frames():
    store = InMemoryIndexStore()
    store.save('video-1', [_sample_frame('f1')])
    store.save('video-2', [FrameRecord('video-2', 'f3', 1.0, 'x', [1.0, 1.0])])

    all_frames = store.load_all()
    ids = {frame.frame_id for frame in all_frames}

    assert ids == {'f1', 'f3'}
