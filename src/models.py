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
class SearchResult:
    frame: FrameRecord
    similarity: float
