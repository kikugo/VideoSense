from pathlib import Path

from src.index_store import ChromaIndexStore
from src.models import FrameRecord


def test_chroma_store_saves_and_loads(tmp_path: Path):
    store = ChromaIndexStore(persist_directory=str(tmp_path / 'chroma'))

    frames = [
        FrameRecord('video-a', 'f1', 0.0, 'thumb1', [0.1, 0.2, 0.3]),
        FrameRecord('video-a', 'f2', 2.0, 'thumb2', [0.2, 0.2, 0.3]),
    ]
    store.save('video-a', frames)

    loaded = store.load('video-a')
    assert loaded is not None
    assert len(loaded) == 2
    assert {item.frame_id for item in loaded} == {'f1', 'f2'}


def test_chroma_store_load_all_returns_multiple_videos(tmp_path: Path):
    store = ChromaIndexStore(persist_directory=str(tmp_path / 'chroma'))
    store.save('video-a', [FrameRecord('video-a', 'a1', 0.0, 'x', [0.1, 0.2])])
    store.save('video-b', [FrameRecord('video-b', 'b1', 1.0, 'x', [0.3, 0.4])])

    all_frames = store.load_all()
    assert {frame.video_id for frame in all_frames} == {'video-a', 'video-b'}
