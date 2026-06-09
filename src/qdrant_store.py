from __future__ import annotations

import uuid

import numpy as np

from src.index_store import IndexStore
from src.models import FrameRecord, SearchResult, TranscriptChunk, TranscriptSearchResult

# Fixed namespace so a given "video_id:item_id" always maps to the same Qdrant
# point id (Qdrant requires uint64 or UUID ids, not arbitrary strings).
_NAMESPACE = uuid.UUID('1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed')

_SCROLL_BATCH = 256


def _point_id(raw: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, raw))


class QdrantIndexStore(IndexStore):
    """IndexStore backed by Qdrant.

    Two collections (frames + transcripts) mirror the Chroma backend, so the
    app-side RRF fusion in ``retrieval.py`` is unchanged. Use ``location`` for
    local/embedded mode (``':memory:'`` or a path) or ``url`` + ``api_key`` for
    Qdrant Cloud.
    """

    def __init__(
        self,
        location: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_prefix: str = 'videosense',
    ) -> None:
        from qdrant_client import QdrantClient

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(location=location or ':memory:')

        self._frames_collection = f'{collection_prefix}_frames'
        self._transcripts_collection = f'{collection_prefix}_transcripts'

    # ----- frames -----

    def save(self, video_id: str, frames: list[FrameRecord]) -> None:
        usable = [frame for frame in frames if frame.embedding is not None]
        self._delete_by_video(self._frames_collection, video_id)
        if not usable:
            return

        from qdrant_client import models

        self._ensure_collection(self._frames_collection, len(usable[0].embedding))
        points = [
            models.PointStruct(
                id=_point_id(f'{video_id}:{frame.frame_id}'),
                vector=list(frame.embedding),
                payload={
                    'video_id': frame.video_id,
                    'frame_id': frame.frame_id,
                    'timestamp_sec': frame.timestamp_sec,
                    'thumbnail_b64': frame.thumbnail_b64,
                    # Raw embedding kept in the payload: Qdrant normalizes the
                    # stored vector under cosine, so we reconstruct the exact
                    # original here for parity with the Chroma backend.
                    'embedding': list(frame.embedding),
                },
            )
            for frame in usable
        ]
        self._client.upsert(collection_name=self._frames_collection, points=points)

    def load(self, video_id: str) -> list[FrameRecord] | None:
        records = self._scroll_frames(video_id)
        return records if records else None

    def load_all(self) -> list[FrameRecord]:
        return self._scroll_frames(None)

    def query_visual(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        hits = self._query(self._frames_collection, query_embedding, top_k)
        return [SearchResult(frame=self._to_frame(hit), similarity=float(hit.score)) for hit in hits]

    # ----- transcripts -----

    def save_transcripts(self, video_id: str, chunks: list[TranscriptChunk]) -> None:
        usable = [chunk for chunk in chunks if chunk.embedding is not None]
        self._delete_by_video(self._transcripts_collection, video_id)
        if not usable:
            return

        from qdrant_client import models

        self._ensure_collection(self._transcripts_collection, len(usable[0].embedding))
        points = [
            models.PointStruct(
                id=_point_id(f'{video_id}:{chunk.chunk_id}'),
                vector=list(chunk.embedding),
                payload={
                    'video_id': chunk.video_id,
                    'chunk_id': chunk.chunk_id,
                    'start_sec': chunk.start_sec,
                    'end_sec': chunk.end_sec,
                    'text': chunk.text,
                    'embedding': list(chunk.embedding),
                },
            )
            for chunk in usable
        ]
        self._client.upsert(collection_name=self._transcripts_collection, points=points)

    def load_transcripts(self, video_id: str) -> list[TranscriptChunk] | None:
        chunks = self._scroll_transcripts(video_id)
        return chunks if chunks else None

    def load_all_transcripts(self) -> list[TranscriptChunk]:
        return self._scroll_transcripts(None)

    def query_transcripts(self, query_embedding: np.ndarray, top_k: int) -> list[TranscriptSearchResult]:
        hits = self._query(self._transcripts_collection, query_embedding, top_k)
        return [TranscriptSearchResult(chunk=self._to_chunk(hit), similarity=float(hit.score)) for hit in hits]

    # ----- helpers -----

    def _ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def _delete_by_video(self, name: str, video_id: str) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(name):
            return
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key='video_id', match=models.MatchValue(value=video_id))]
                )
            ),
        )

    def _query(self, name: str, query_embedding: np.ndarray, top_k: int):
        if not self._client.collection_exists(name):
            return []
        response = self._client.query_points(
            collection_name=name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def _scroll(self, name: str, video_id: str | None):
        from qdrant_client import models

        if not self._client.collection_exists(name):
            return []

        scroll_filter = None
        if video_id is not None:
            scroll_filter = models.Filter(
                must=[models.FieldCondition(key='video_id', match=models.MatchValue(value=video_id))]
            )

        points = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=_SCROLL_BATCH,
                offset=offset,
            )
            points.extend(batch)
            if offset is None:
                break
        return points

    def _scroll_frames(self, video_id: str | None) -> list[FrameRecord]:
        return [self._to_frame(point) for point in self._scroll(self._frames_collection, video_id)]

    def _scroll_transcripts(self, video_id: str | None) -> list[TranscriptChunk]:
        return [self._to_chunk(point) for point in self._scroll(self._transcripts_collection, video_id)]

    @staticmethod
    def _embedding(payload: dict) -> list[float]:
        return [float(value) for value in (payload.get('embedding') or [])]

    def _to_frame(self, point) -> FrameRecord:
        payload = point.payload or {}
        return FrameRecord(
            video_id=str(payload['video_id']),
            frame_id=str(payload['frame_id']),
            timestamp_sec=float(payload['timestamp_sec']),
            thumbnail_b64=str(payload['thumbnail_b64']),
            embedding=self._embedding(payload),
        )

    def _to_chunk(self, point) -> TranscriptChunk:
        payload = point.payload or {}
        return TranscriptChunk(
            video_id=str(payload['video_id']),
            chunk_id=str(payload['chunk_id']),
            start_sec=float(payload['start_sec']),
            end_sec=float(payload['end_sec']),
            text=str(payload['text']),
            embedding=self._embedding(payload),
        )
