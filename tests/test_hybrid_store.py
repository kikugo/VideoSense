from src.index_store import HybridIndexStore, InMemoryIndexStore
from src.models import FrameRecord


class FakePersistentStore(InMemoryIndexStore):
    pass


def test_hybrid_store_writes_to_both_backends():
    memory = InMemoryIndexStore()
    persistent = FakePersistentStore()
    store = HybridIndexStore(memory_store=memory, persistent_store=persistent)

    frame = FrameRecord('v1', 'f1', 0.0, 'thumb', [0.1, 0.2])
    store.save('v1', [frame])

    assert memory.load('v1') is not None
    assert persistent.load('v1') is not None


def test_hybrid_store_falls_back_to_persistent_load():
    memory = InMemoryIndexStore()
    persistent = FakePersistentStore()
    store = HybridIndexStore(memory_store=memory, persistent_store=persistent)

    frame = FrameRecord('v1', 'f1', 0.0, 'thumb', [0.1, 0.2])
    persistent.save('v1', [frame])

    loaded = store.load('v1')
    assert loaded is not None
    assert loaded[0].frame_id == 'f1'
    assert memory.load('v1') is not None
