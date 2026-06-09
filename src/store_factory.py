from __future__ import annotations

from typing import Callable, Optional

from src.index_store import ChromaIndexStore, IndexStore
from src.qdrant_store import QdrantIndexStore


def _default_qdrant_factory(**kwargs) -> IndexStore:
    return QdrantIndexStore(**kwargs)


def _default_chroma_factory(**kwargs) -> IndexStore:
    return ChromaIndexStore(**kwargs)


def build_persistent_store(
    config,
    *,
    qdrant_factory: Callable[..., IndexStore] = _default_qdrant_factory,
    chroma_factory: Callable[..., IndexStore] = _default_chroma_factory,
) -> Optional[IndexStore]:
    """Pick the persistent backend for a HybridIndexStore.

    - ``memory``: no persistence (in-memory only).
    - ``qdrant`` / ``auto`` (with a url): use Qdrant if it is reachable; if the
      health check fails, fall back to Chroma (when persistence is enabled) so a
      suspended free-tier cluster can never produce a broken demo.
    - ``chroma`` / ``auto`` (no url) / Qdrant unreachable: use Chroma when
      persistence is enabled, otherwise no persistent store.
    """
    backend = (getattr(config, 'vector_backend', 'auto') or 'auto').strip().lower()

    if backend == 'memory':
        return None

    wants_qdrant = backend == 'qdrant' or (backend == 'auto' and bool(config.qdrant_url))
    if wants_qdrant and config.qdrant_url:
        try:
            store = qdrant_factory(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
            store.health_check()  # type: ignore[attr-defined]
            return store
        except Exception:
            # Unreachable/misconfigured Qdrant: degrade gracefully below.
            pass

    if config.enable_persistence:
        return chroma_factory(persist_directory=config.persistence_dir)
    return None
