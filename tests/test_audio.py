from __future__ import annotations

from pathlib import Path

from src.audio import build_transcript_chunks, extract_audio_path, transcribe_video


def test_build_transcript_chunks_groups_segments_by_window():
    segments = [
        {'start': 0.0, 'end': 3.0, 'text': 'hello'},
        {'start': 3.0, 'end': 7.0, 'text': 'world'},
        {'start': 13.0, 'end': 16.0, 'text': 'later'},
    ]

    chunks = build_transcript_chunks('video-1', segments, target_window_sec=10.0)

    assert len(chunks) == 2
    assert chunks[0].text == 'hello world'
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 7.0
    assert chunks[1].text == 'later'


def test_extract_audio_path_uses_audio_directory(tmp_path: Path):
    output = extract_audio_path(video_id='video-1', library_root=tmp_path)

    assert output == tmp_path / 'audio' / 'video-1.wav'


def test_transcribe_video_uses_supplied_transcriber(tmp_path: Path):
    def fake_transcriber(_audio_path: Path):
        return [{'start': 1.0, 'end': 2.0, 'text': 'spoken words'}]

    chunks = transcribe_video(
        video_path=tmp_path / 'sample.mp4',
        video_id='video-1',
        library_root=tmp_path,
        transcriber=fake_transcriber,
        audio_extractor=lambda *_args, **_kwargs: None,
    )

    assert chunks[0].video_id == 'video-1'
    assert chunks[0].text == 'spoken words'
