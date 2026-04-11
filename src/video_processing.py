from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import cv2
import numpy as np


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
