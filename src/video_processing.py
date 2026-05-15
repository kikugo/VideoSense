from __future__ import annotations

import base64
import hashlib
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.models import FrameRecord


def build_sample_timestamps(duration_sec: float, interval_sec: float) -> list[float]:
    if interval_sec <= 0:
        raise ValueError('interval_sec must be greater than zero.')

    if duration_sec <= 0:
        return [0.0]

    timestamps: list[float] = []
    current = 0.0
    while current <= duration_sec:
        timestamps.append(round(current, 3))
        current += interval_sec

    if not timestamps:
        return [0.0]

    return timestamps


def frame_to_jpeg_base64(frame_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        raise RuntimeError('Failed to encode frame as JPEG.')

    return base64.b64encode(encoded.tobytes()).decode('utf-8')


def compute_video_id(video_path: str) -> str:
    normalized = str(Path(video_path).resolve())
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()


def compute_video_id_from_bytes(video_bytes: bytes) -> str:
    return hashlib.sha1(video_bytes).hexdigest()


def extract_sampled_frames(
    video_path: str,
    video_id: str,
    interval_sec: float,
    timestamps: list[float] | None = None,
) -> list[FrameRecord]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError('Could not determine video duration from stream metadata.')

    duration_sec = frame_count / fps
    timestamps = timestamps or build_sample_timestamps(duration_sec, interval_sec)

    records: list[FrameRecord] = []
    for index, timestamp in enumerate(timestamps):
        frame_index = int(round(timestamp * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        records.append(
            FrameRecord(
                video_id=video_id,
                frame_id=f'f{index:05d}',
                timestamp_sec=float(timestamp),
                thumbnail_b64=frame_to_jpeg_base64(frame),
                embedding=None,
            )
        )

    capture.release()
    return records


def persist_uploaded_video(uploaded_file, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', uploaded_file.name)
    output_path = base_dir / f'{uuid.uuid4().hex[:8]}_{safe_name}'
    output_path.write_bytes(bytes(uploaded_file.getbuffer()))
    return output_path
