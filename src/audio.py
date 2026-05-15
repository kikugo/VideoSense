from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterable

from src.models import TranscriptChunk


def extract_audio_path(video_id: str, library_root: Path) -> Path:
    return library_root / 'audio' / f'{video_id}.wav'


def extract_audio(video_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(video_path),
            '-vn',
            '-acodec',
            'pcm_s16le',
            '-ar',
            '16000',
            '-ac',
            '1',
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def whisper_transcribe(audio_path: Path) -> list[dict[str, float | str]]:
    from faster_whisper import WhisperModel

    model = WhisperModel('base', device='cpu', compute_type='int8')
    segments, _info = model.transcribe(str(audio_path), word_timestamps=False)
    return [{'start': segment.start, 'end': segment.end, 'text': segment.text.strip()} for segment in segments]


def build_transcript_chunks(
    video_id: str,
    segments: Iterable[dict[str, float | str]],
    target_window_sec: float = 10.0,
) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current_text: list[str] = []
    current_start: float | None = None
    current_end = 0.0

    for segment in segments:
        start = float(segment['start'])
        end = float(segment['end'])
        text = str(segment['text']).strip()
        if not text:
            continue

        if current_start is None:
            current_start = start

        if current_text and end - current_start > target_window_sec:
            chunks.append(
                TranscriptChunk(
                    video_id=video_id,
                    chunk_id=f't{len(chunks):05d}',
                    start_sec=current_start,
                    end_sec=current_end,
                    text=' '.join(current_text),
                )
            )
            current_text = []
            current_start = start

        current_text.append(text)
        current_end = end

    if current_text and current_start is not None:
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                chunk_id=f't{len(chunks):05d}',
                start_sec=current_start,
                end_sec=current_end,
                text=' '.join(current_text),
            )
        )

    return chunks


def transcribe_video(
    video_path: Path,
    video_id: str,
    library_root: Path,
    transcriber: Callable[[Path], list[dict[str, float | str]]] = whisper_transcribe,
    audio_extractor: Callable[[Path, Path], None] = extract_audio,
) -> list[TranscriptChunk]:
    audio_path = extract_audio_path(video_id=video_id, library_root=library_root)
    audio_extractor(video_path, audio_path)
    segments = transcriber(audio_path)
    return build_transcript_chunks(video_id=video_id, segments=segments)
