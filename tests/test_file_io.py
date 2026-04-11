from pathlib import Path

from src.video_processing import persist_uploaded_video


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def test_persist_uploaded_video_writes_bytes(tmp_path: Path):
    upload = FakeUpload(name='clip.mp4', data=b'abc123')

    output_path = persist_uploaded_video(upload, base_dir=tmp_path)

    assert output_path.exists()
    assert output_path.read_bytes() == b'abc123'
    assert output_path.name.endswith('_clip.mp4')
