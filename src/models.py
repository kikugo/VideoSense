from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
