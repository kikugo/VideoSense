from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import FrameRecord


class IndexStore(ABC):
    @abstractmethod
    def save(self, video_id: str, frames: list[FrameRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, video_id: str) -> list[FrameRecord] | None:
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[FrameRecord]:
        raise NotImplementedError


class InMemoryIndexStore(IndexStore):
    def __init__(self) -> None:
        self._frames_by_video_id: dict[str, list[FrameRecord]] = {}

    def save(self, video_id: str, frames: list[FrameRecord]) -> None:
        self._frames_by_video_id[video_id] = list(frames)

    def load(self, video_id: str) -> list[FrameRecord] | None:
        frames = self._frames_by_video_id.get(video_id)
        if frames is None:
            return None
        return list(frames)

    def load_all(self) -> list[FrameRecord]:
        merged: list[FrameRecord] = []
        for frames in self._frames_by_video_id.values():
            merged.extend(frames)
        return merged


class ChromaIndexStore(IndexStore):
    def __init__(self, persist_directory: str, collection_name: str = 'videosense_frames') -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def save(self, video_id: str, frames: list[FrameRecord]) -> None:
        existing = self._collection.get(where={'video_id': video_id}, include=[])
        existing_ids = existing.get('ids', []) if existing else []
        if existing_ids:
            self._collection.delete(ids=existing_ids)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | float]] = []
        documents: list[str] = []

        for frame in frames:
            if frame.embedding is None:
                continue
            ids.append(f'{video_id}:{frame.frame_id}')
            embeddings.append(frame.embedding)
            metadatas.append(
                {
                    'video_id': frame.video_id,
                    'frame_id': frame.frame_id,
                    'timestamp_sec': frame.timestamp_sec,
                }
            )
            documents.append(frame.thumbnail_b64)

        if ids:
            self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def load(self, video_id: str) -> list[FrameRecord] | None:
        result = self._collection.get(where={'video_id': video_id}, include=['embeddings', 'metadatas', 'documents'])
        ids = result.get('ids', [])
        if not ids:
            return None
        return self._to_frame_records(result)

    def load_all(self) -> list[FrameRecord]:
        result = self._collection.get(include=['embeddings', 'metadatas', 'documents'])
        ids = result.get('ids', [])
        if not ids:
            return []
        return self._to_frame_records(result)

    def _to_frame_records(self, result: dict) -> list[FrameRecord]:
        records: list[FrameRecord] = []

        ids = result.get('ids', [])
        embeddings = result.get('embeddings', [])
        metadatas = result.get('metadatas', [])
        documents = result.get('documents', [])

        for idx, _ in enumerate(ids):
            metadata = metadatas[idx]
            records.append(
                FrameRecord(
                    video_id=str(metadata['video_id']),
                    frame_id=str(metadata['frame_id']),
                    timestamp_sec=float(metadata['timestamp_sec']),
                    thumbnail_b64=str(documents[idx]),
                    embedding=[float(v) for v in embeddings[idx]],
                )
            )

        return records


class HybridIndexStore(IndexStore):
    def __init__(self, memory_store: InMemoryIndexStore, persistent_store: IndexStore | None = None) -> None:
        self._memory_store = memory_store
        self._persistent_store = persistent_store

    def save(self, video_id: str, frames: list[FrameRecord]) -> None:
        self._memory_store.save(video_id, frames)
        if self._persistent_store:
            self._persistent_store.save(video_id, frames)

    def load(self, video_id: str) -> list[FrameRecord] | None:
        in_memory = self._memory_store.load(video_id)
        if in_memory:
            return in_memory

        if not self._persistent_store:
            return None

        restored = self._persistent_store.load(video_id)
        if restored:
            self._memory_store.save(video_id, restored)
        return restored

    def load_all(self) -> list[FrameRecord]:
        in_memory = self._memory_store.load_all()
        if in_memory:
            return in_memory

        if not self._persistent_store:
            return []

        restored = self._persistent_store.load_all()
        by_video: dict[str, list[FrameRecord]] = {}
        for frame in restored:
            by_video.setdefault(frame.video_id, []).append(frame)
        for video_id, frames in by_video.items():
            self._memory_store.save(video_id, frames)
        return restored
