from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str
    embedding_model: str
    frame_interval_sec: float
    top_k: int
    embed_concurrency: int
    embed_max_retries: int
    embed_backoff_sec: float
    enable_persistence: bool
    persistence_dir: str
    min_similarity: float
    frame_strategy: str
    scene_threshold: float
    max_visual_frames: int

    @classmethod
    def from_env(cls) -> 'AppConfig':
        load_dotenv()

        return cls(
            gemini_api_key=os.getenv('GEMINI_API_KEY', '').strip(),
            embedding_model=os.getenv('VIDEOSENSE_EMBEDDING_MODEL', 'gemini-embedding-2-preview').strip(),
            frame_interval_sec=float(os.getenv('VIDEOSENSE_FRAME_INTERVAL_SEC', '2.0')),
            top_k=int(os.getenv('VIDEOSENSE_TOP_K', '3')),
            embed_concurrency=int(os.getenv('VIDEOSENSE_EMBED_CONCURRENCY', '4')),
            embed_max_retries=int(os.getenv('VIDEOSENSE_EMBED_MAX_RETRIES', '2')),
            embed_backoff_sec=float(os.getenv('VIDEOSENSE_EMBED_BACKOFF_SEC', '1.0')),
            enable_persistence=os.getenv('VIDEOSENSE_ENABLE_PERSISTENCE', 'false').strip().lower() in {'1', 'true', 'yes'},
            persistence_dir=os.getenv('VIDEOSENSE_PERSISTENCE_DIR', '.videosense/chroma').strip(),
            min_similarity=float(os.getenv('VIDEOSENSE_MIN_SIMILARITY', '0.3')),
            frame_strategy=os.getenv('VIDEOSENSE_FRAME_STRATEGY', 'scene').strip(),
            scene_threshold=float(os.getenv('VIDEOSENSE_SCENE_THRESHOLD', '27.0')),
            max_visual_frames=int(os.getenv('VIDEOSENSE_MAX_VISUAL_FRAMES', '300')),
        )
