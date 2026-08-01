from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoMetadata:
    video_id: str
    name: str
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    indexed_at: str
    visual_frame_count: int
    transcript_chunk_count: int


@dataclass
class FrameRecord:
    video_id: str
    frame_id: str
    timestamp_sec: float
    thumbnail_b64: str
    embedding: Optional[list[float]] = None


@dataclass
class TranscriptChunk:
    video_id: str
    chunk_id: str
    start_sec: float
    end_sec: float
    text: str
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    frame: FrameRecord
    similarity: float


@dataclass
class TranscriptSearchResult:
    chunk: TranscriptChunk
    similarity: float


@dataclass
class UnifiedSearchResult:
    video_id: str
    start_sec: float
    end_sec: float
    score: float
    channel: str
    # `score` is an RRF rank score (roughly 0.016-0.033 at rrf_k=60) and is only
    # meaningful for ordering. `similarity` is the best raw cosine among the
    # channels that matched, which is what a 0-1 threshold can be compared to.
    # Filtering on `score` silently discarded every result.
    similarity: float = 0.0
    frame: Optional[FrameRecord] = None
    transcript: Optional[TranscriptChunk] = None
