from __future__ import annotations

from pathlib import Path

from src.library import (
    load_catalog,
    persist_video_bytes,
    remove_video,
    save_catalog,
    upsert_video_metadata,
)
from src.models import VideoMetadata


def _metadata(video_id: str, path: Path) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        name='sample.mp4',
        path=str(path),
        duration_sec=10.0,
        fps=30.0,
        frame_count=300,
        indexed_at='2026-05-15T10:00:00Z',
        visual_frame_count=5,
        transcript_chunk_count=2,
    )


def test_catalog_round_trips_video_metadata(tmp_path: Path):
    catalog_path = tmp_path / 'library.json'
    metadata = _metadata('video-1', tmp_path / 'video.mp4')

    save_catalog(catalog_path, {'video-1': metadata})

    loaded = load_catalog(catalog_path)
    assert loaded['video-1'] == metadata


def test_upsert_and_remove_video_metadata(tmp_path: Path):
    catalog: dict[str, VideoMetadata] = {}
    metadata = _metadata('video-1', tmp_path / 'video.mp4')

    upsert_video_metadata(catalog, metadata)
    remove_video(catalog, 'video-1')

    assert catalog == {}


def test_persist_video_bytes_uses_content_id_and_extension(tmp_path: Path):
    output_path = persist_video_bytes(
        video_bytes=b'video-bytes',
        original_name='My Clip.MP4',
        library_root=tmp_path,
    )

    assert output_path.exists()
    assert output_path.parent == tmp_path / 'videos'
    assert output_path.suffix == '.mp4'
    assert output_path.read_bytes() == b'video-bytes'
