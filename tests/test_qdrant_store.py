import numpy as np

from src.models import FrameRecord, TranscriptChunk
from src.qdrant_store import QdrantIndexStore


def _store() -> QdrantIndexStore:
    # In-memory mode: no server needed, each store is isolated.
    return QdrantIndexStore(location=':memory:')


def test_qdrant_store_saves_and_loads_frames():
    store = _store()
    frames = [
        FrameRecord('video-a', 'f1', 0.0, 'thumb1', [0.1, 0.2, 0.3]),
        FrameRecord('video-a', 'f2', 2.0, 'thumb2', [0.2, 0.2, 0.3]),
    ]
    store.save('video-a', frames)

    loaded = store.load('video-a')
    assert loaded is not None
    assert {item.frame_id for item in loaded} == {'f1', 'f2'}
    by_id = {f.frame_id: f for f in loaded}
    assert by_id['f1'].thumbnail_b64 == 'thumb1'
    assert by_id['f1'].timestamp_sec == 0.0
    assert by_id['f2'].embedding == [0.2, 0.2, 0.3]


def test_qdrant_store_load_all_returns_multiple_videos():
    store = _store()
    store.save('video-a', [FrameRecord('video-a', 'a1', 0.0, 'x', [0.1, 0.2])])
    store.save('video-b', [FrameRecord('video-b', 'b1', 1.0, 'x', [0.3, 0.4])])

    all_frames = store.load_all()
    assert {frame.video_id for frame in all_frames} == {'video-a', 'video-b'}


def test_qdrant_store_reindex_replaces_frames_for_video():
    store = _store()
    store.save('video-a', [FrameRecord('video-a', 'f1', 0.0, 'x', [0.1, 0.2])])
    store.save('video-a', [FrameRecord('video-a', 'f2', 1.0, 'y', [0.3, 0.4])])

    loaded = store.load('video-a')
    assert loaded is not None
    assert {f.frame_id for f in loaded} == {'f2'}


def test_qdrant_query_visual_ranks_closest_first():
    store = _store()
    store.save(
        'video-a',
        [
            FrameRecord('video-a', 'near', 0.0, 'x', [1.0, 0.0, 0.0]),
            FrameRecord('video-a', 'far', 1.0, 'y', [0.0, 1.0, 0.0]),
        ],
    )

    results = store.query_visual(np.array([0.9, 0.1, 0.0]), top_k=2)
    assert len(results) == 2
    assert results[0].frame.frame_id == 'near'
    assert results[0].similarity >= results[1].similarity


def test_qdrant_query_visual_empty_when_no_data():
    store = _store()
    assert store.query_visual(np.array([0.1, 0.2]), top_k=3) == []


def test_qdrant_store_saves_loads_and_queries_transcripts():
    store = _store()
    chunks = [
        TranscriptChunk('video-a', 'c1', 0.0, 2.0, 'hello world', [1.0, 0.0]),
        TranscriptChunk('video-a', 'c2', 2.0, 4.0, 'goodbye', [0.0, 1.0]),
    ]
    store.save_transcripts('video-a', chunks)

    loaded = store.load_transcripts('video-a')
    assert loaded is not None
    assert {c.chunk_id for c in loaded} == {'c1', 'c2'}

    results = store.query_transcripts(np.array([0.95, 0.05]), top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == 'c1'
    assert results[0].chunk.text == 'hello world'


def test_qdrant_load_missing_video_returns_none():
    store = _store()
    assert store.load('nope') is None
    assert store.load_transcripts('nope') is None
