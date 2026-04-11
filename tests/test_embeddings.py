import asyncio

from src.embeddings import embed_frames_with_retry
from src.models import FrameRecord


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed_image_base64(self, thumbnail_b64: str):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('temporary failure')
        return [0.1, 0.2, 0.3]


async def _run_retry_case():
    frame = FrameRecord(video_id='v1', frame_id='f1', timestamp_sec=0.0, thumbnail_b64='abc')
    embedder = FakeEmbedder()

    updated, failures = await embed_frames_with_retry(
        frames=[frame],
        embed_image_fn=embedder.embed_image_base64,
        concurrency=2,
        max_retries=2,
        base_backoff_sec=0.0,
    )

    assert failures == []
    assert updated[0].embedding == [0.1, 0.2, 0.3]


async def _run_failure_case():
    async def always_fail(_: str):
        raise RuntimeError('boom')

    frame = FrameRecord(video_id='v1', frame_id='f1', timestamp_sec=0.0, thumbnail_b64='abc')
    updated, failures = await embed_frames_with_retry(
        frames=[frame],
        embed_image_fn=always_fail,
        concurrency=1,
        max_retries=1,
        base_backoff_sec=0.0,
    )

    assert updated[0].embedding is None
    assert failures == ['f1']


def test_embed_frames_with_retry_recovers_transient_errors():
    asyncio.run(_run_retry_case())


def test_embed_frames_with_retry_reports_failed_frames():
    asyncio.run(_run_failure_case())


async def _run_progress_case():
    calls = []

    async def ok(_: str):
        return [1.0, 2.0]

    frames = [
        FrameRecord(video_id='v1', frame_id='f1', timestamp_sec=0.0, thumbnail_b64='a'),
        FrameRecord(video_id='v1', frame_id='f2', timestamp_sec=2.0, thumbnail_b64='b'),
    ]

    await embed_frames_with_retry(
        frames=frames,
        embed_image_fn=ok,
        concurrency=2,
        max_retries=1,
        base_backoff_sec=0.0,
        on_progress=lambda done, total: calls.append((done, total)),
    )

    assert calls[-1] == (2, 2)


def test_embed_frames_with_retry_reports_progress():
    asyncio.run(_run_progress_case())
