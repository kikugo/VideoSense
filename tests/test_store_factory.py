from types import SimpleNamespace

from src.store_factory import build_persistent_store


def _config(**overrides):
    base = dict(
        vector_backend='auto',
        qdrant_url='',
        qdrant_api_key='',
        enable_persistence=False,
        persistence_dir='.videosense/chroma',
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeStore:
    def __init__(self, name: str, healthy: bool = True):
        self.name = name
        self._healthy = healthy

    def health_check(self) -> None:
        if not self._healthy:
            raise ConnectionError('qdrant unreachable')


def _factories(qdrant: _FakeStore, chroma: _FakeStore):
    return dict(
        qdrant_factory=lambda **_: qdrant,
        chroma_factory=lambda **_: chroma,
    )


def test_memory_backend_returns_none():
    store = build_persistent_store(
        _config(vector_backend='memory'),
        **_factories(_FakeStore('q'), _FakeStore('c')),
    )
    assert store is None


def test_qdrant_backend_healthy_returns_qdrant():
    store = build_persistent_store(
        _config(vector_backend='qdrant', qdrant_url='http://x:6333'),
        **_factories(_FakeStore('q'), _FakeStore('c')),
    )
    assert store.name == 'q'


def test_qdrant_unreachable_falls_back_to_chroma_when_persistence_on():
    store = build_persistent_store(
        _config(vector_backend='qdrant', qdrant_url='http://x:6333', enable_persistence=True),
        **_factories(_FakeStore('q', healthy=False), _FakeStore('c')),
    )
    assert store.name == 'c'


def test_qdrant_unreachable_returns_none_when_persistence_off():
    store = build_persistent_store(
        _config(vector_backend='qdrant', qdrant_url='http://x:6333', enable_persistence=False),
        **_factories(_FakeStore('q', healthy=False), _FakeStore('c')),
    )
    assert store is None


def test_chroma_backend_with_persistence_returns_chroma():
    store = build_persistent_store(
        _config(vector_backend='chroma', enable_persistence=True),
        **_factories(_FakeStore('q'), _FakeStore('c')),
    )
    assert store.name == 'c'


def test_auto_without_qdrant_url_uses_chroma_when_persistence_on():
    store = build_persistent_store(
        _config(vector_backend='auto', qdrant_url='', enable_persistence=True),
        **_factories(_FakeStore('q'), _FakeStore('c')),
    )
    assert store.name == 'c'


def test_auto_with_qdrant_url_healthy_uses_qdrant():
    store = build_persistent_store(
        _config(vector_backend='auto', qdrant_url='http://x:6333'),
        **_factories(_FakeStore('q'), _FakeStore('c')),
    )
    assert store.name == 'q'
