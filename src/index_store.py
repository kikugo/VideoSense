from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models import FrameRecord, SearchResult, TranscriptChunk, TranscriptSearchResult
from src.search import cosine_similarity, rank_results


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

    @abstractmethod
    def query_visual(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def save_transcripts(self, video_id: str, chunks: list[TranscriptChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_transcripts(self, video_id: str) -> list[TranscriptChunk] | None:
        raise NotImplementedError

    @abstractmethod
    def load_all_transcripts(self) -> list[TranscriptChunk]:
        raise NotImplementedError

    @abstractmethod
    def query_transcripts(self, query_embedding: np.ndarray, top_k: int) -> list[TranscriptSearchResult]:
        raise NotImplementedError


class InMemoryIndexStore(IndexStore):
    def __init__(self) -> None:
        self._frames_by_video_id: dict[str, list[FrameRecord]] = {}
        self._transcripts_by_video_id: dict[str, list[TranscriptChunk]] = {}

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

    def query_visual(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        return rank_results(query_embedding=query_embedding, frames=self.load_all(), top_k=top_k)

    def save_transcripts(self, video_id: str, chunks: list[TranscriptChunk]) -> None:
        self._transcripts_by_video_id[video_id] = list(chunks)

    def load_transcripts(self, video_id: str) -> list[TranscriptChunk] | None:
        chunks = self._transcripts_by_video_id.get(video_id)
        if chunks is None:
            return None
        return list(chunks)

    def load_all_transcripts(self) -> list[TranscriptChunk]:
        merged: list[TranscriptChunk] = []
        for chunks in self._transcripts_by_video_id.values():
            merged.extend(chunks)
        return merged

    def query_transcripts(self, query_embedding: np.ndarray, top_k: int) -> list[TranscriptSearchResult]:
        scored: list[TranscriptSearchResult] = []
        for chunk in self.load_all_transcripts():
            if not chunk.embedding:
                continue
            similarity = cosine_similarity(query_embedding, np.array(chunk.embedding, dtype=float))
            scored.append(TranscriptSearchResult(chunk=chunk, similarity=similarity))
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]


class ChromaIndexStore(IndexStore):
    def __init__(self, persist_directory: str, collection_name: str = 'videosense_frames') -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._transcript_collection = self._client.get_or_create_collection(name='videosense_transcripts')

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

    def query_visual(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['embeddings', 'metadatas', 'documents', 'distances'],
        )
        frames = self._to_frame_records(
            {
                'ids': result.get('ids', [[]])[0],
                'embeddings': result.get('embeddings', [[]])[0],
                'metadatas': result.get('metadatas', [[]])[0],
                'documents': result.get('documents', [[]])[0],
            }
        )
        distances = result.get('distances', [[]])[0]
        return [SearchResult(frame=frame, similarity=1.0 - float(distances[index])) for index, frame in enumerate(frames)]

    def save_transcripts(self, video_id: str, chunks: list[TranscriptChunk]) -> None:
        existing = self._transcript_collection.get(where={'video_id': video_id}, include=[])
        existing_ids = existing.get('ids', []) if existing else []
        if existing_ids:
            self._transcript_collection.delete(ids=existing_ids)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | float]] = []
        documents: list[str] = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            ids.append(f'{video_id}:{chunk.chunk_id}')
            embeddings.append(chunk.embedding)
            metadatas.append(
                {
                    'video_id': chunk.video_id,
                    'chunk_id': chunk.chunk_id,
                    'start_sec': chunk.start_sec,
                    'end_sec': chunk.end_sec,
                }
            )
            documents.append(chunk.text)
        if ids:
            self._transcript_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def load_transcripts(self, video_id: str) -> list[TranscriptChunk] | None:
        result = self._transcript_collection.get(where={'video_id': video_id}, include=['embeddings', 'metadatas', 'documents'])
        ids = result.get('ids', [])
        if not ids:
            return None
        return self._to_transcript_chunks(result)

    def load_all_transcripts(self) -> list[TranscriptChunk]:
        result = self._transcript_collection.get(include=['embeddings', 'metadatas', 'documents'])
        ids = result.get('ids', [])
        if not ids:
            return []
        return self._to_transcript_chunks(result)

    def query_transcripts(self, query_embedding: np.ndarray, top_k: int) -> list[TranscriptSearchResult]:
        result = self._transcript_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['embeddings', 'metadatas', 'documents', 'distances'],
        )
        chunks = self._to_transcript_chunks(
            {
                'ids': result.get('ids', [[]])[0],
                'embeddings': result.get('embeddings', [[]])[0],
                'metadatas': result.get('metadatas', [[]])[0],
                'documents': result.get('documents', [[]])[0],
            }
        )
        distances = result.get('distances', [[]])[0]
        return [TranscriptSearchResult(chunk=chunk, similarity=1.0 - float(distances[index])) for index, chunk in enumerate(chunks)]

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

    def _to_transcript_chunks(self, result: dict) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        ids = result.get('ids', [])
        embeddings = result.get('embeddings', [])
        metadatas = result.get('metadatas', [])
        documents = result.get('documents', [])
        for idx, _ in enumerate(ids):
            metadata = metadatas[idx]
            chunks.append(
                TranscriptChunk(
                    video_id=str(metadata['video_id']),
                    chunk_id=str(metadata['chunk_id']),
                    start_sec=float(metadata['start_sec']),
                    end_sec=float(metadata['end_sec']),
                    text=str(documents[idx]),
                    embedding=[float(v) for v in embeddings[idx]],
                )
            )
        return chunks


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

    def query_visual(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._memory_store.load_all():
            return self._memory_store.query_visual(query_embedding, top_k)
        if self._persistent_store:
            return self._persistent_store.query_visual(query_embedding, top_k)
        return []

    def save_transcripts(self, video_id: str, chunks: list[TranscriptChunk]) -> None:
        self._memory_store.save_transcripts(video_id, chunks)
        if self._persistent_store:
            self._persistent_store.save_transcripts(video_id, chunks)

    def load_transcripts(self, video_id: str) -> list[TranscriptChunk] | None:
        in_memory = self._memory_store.load_transcripts(video_id)
        if in_memory:
            return in_memory
        if not self._persistent_store:
            return None
        restored = self._persistent_store.load_transcripts(video_id)
        if restored:
            self._memory_store.save_transcripts(video_id, restored)
        return restored

    def load_all_transcripts(self) -> list[TranscriptChunk]:
        in_memory = self._memory_store.load_all_transcripts()
        if in_memory:
            return in_memory
        if not self._persistent_store:
            return []
        restored = self._persistent_store.load_all_transcripts()
        by_video: dict[str, list[TranscriptChunk]] = {}
        for chunk in restored:
            by_video.setdefault(chunk.video_id, []).append(chunk)
        for video_id, chunks in by_video.items():
            self._memory_store.save_transcripts(video_id, chunks)
        return restored

    def query_transcripts(self, query_embedding: np.ndarray, top_k: int) -> list[TranscriptSearchResult]:
        if self._memory_store.load_all_transcripts():
            return self._memory_store.query_transcripts(query_embedding, top_k)
        if self._persistent_store:
            return self._persistent_store.query_transcripts(query_embedding, top_k)
        return []
