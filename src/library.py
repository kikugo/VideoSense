from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.models import VideoMetadata
from src.video_processing import compute_video_id_from_bytes


def load_catalog(catalog_path: Path) -> dict[str, VideoMetadata]:
    if not catalog_path.exists():
        return {}

    raw = json.loads(catalog_path.read_text(encoding='utf-8'))
    return {video_id: VideoMetadata(**data) for video_id, data in raw.items()}


def save_catalog(catalog_path: Path, catalog: dict[str, VideoMetadata]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {video_id: asdict(metadata) for video_id, metadata in catalog.items()}
    catalog_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def upsert_video_metadata(catalog: dict[str, VideoMetadata], metadata: VideoMetadata) -> None:
    catalog[metadata.video_id] = metadata


def remove_video(catalog: dict[str, VideoMetadata], video_id: str) -> None:
    catalog.pop(video_id, None)


def persist_video_bytes(video_bytes: bytes, original_name: str, library_root: Path) -> Path:
    video_id = compute_video_id_from_bytes(video_bytes)
    suffix = Path(original_name).suffix.lower() or '.mp4'
    output_dir = library_root / 'videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{video_id}{suffix}'
    if not output_path.exists():
        output_path.write_bytes(video_bytes)
    return output_path
