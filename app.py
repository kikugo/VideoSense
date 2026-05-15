from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

from src.app_state import ensure_defaults, set_playback_target
from src.audio import transcribe_video
from src.config import AppConfig
from src.embeddings import GeminiEmbedder
from src.index_store import ChromaIndexStore, HybridIndexStore, InMemoryIndexStore
from src.library import load_catalog, persist_video_bytes, save_catalog, upsert_video_metadata
from src.models import VideoMetadata
from src.pipeline import embed_transcripts_into_store, index_frames_into_store, search_library
from src.scenes import select_visual_timestamps
from src.ui import render_unified_results
from src.video_processing import (
    compute_video_id_from_bytes,
    extract_sampled_frames,
)


def _get_runtime() -> tuple[AppConfig, GeminiEmbedder, HybridIndexStore]:
    if 'runtime' in st.session_state:
        return st.session_state['runtime']

    config = AppConfig.from_env()
    embedder = GeminiEmbedder(api_key=config.gemini_api_key, model=config.embedding_model)

    memory_store = InMemoryIndexStore()
    persistent_store = None
    if config.enable_persistence:
        persistent_store = ChromaIndexStore(persist_directory=config.persistence_dir)

    store = HybridIndexStore(memory_store=memory_store, persistent_store=persistent_store)
    st.session_state['runtime'] = (config, embedder, store)
    return config, embedder, store


def _ensure_session_defaults() -> None:
    ensure_defaults(st.session_state)


def _video_metadata(video_id: str, name: str, path: Path, frames, chunks) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        name=name,
        path=str(path),
        duration_sec=max((frame.timestamp_sec for frame in frames), default=0.0),
        fps=0.0,
        frame_count=0,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        visual_frame_count=len(frames),
        transcript_chunk_count=len(chunks),
    )


def main() -> None:
    st.set_page_config(page_title='VideoSense', page_icon='🎬', layout='wide')
    st.title('VideoSense')
    st.write('Search through your videos with natural language.')

    _ensure_session_defaults()

    try:
        config, embedder, store = _get_runtime()
    except ValueError as exc:
        st.error(str(exc))
        st.info('Set GEMINI_API_KEY in your local environment or .env file, then restart the app.')
        return

    library_root = Path('.videosense')
    catalog_path = library_root / 'library.json'
    catalog = load_catalog(catalog_path)
    for metadata in catalog.values():
        st.session_state['video_name_by_id'][metadata.video_id] = metadata.name
        st.session_state['video_path_by_id'][metadata.video_id] = metadata.path

    with st.sidebar:
        st.subheader('Search Controls')
        top_k = st.slider('Results', min_value=1, max_value=12, value=config.top_k)
        min_score = st.slider('Minimum score', min_value=0.0, max_value=1.0, value=config.min_similarity)
        enable_transcripts = st.toggle('Transcript channel', value=True)
        st.caption(f'{len(catalog)} indexed video(s)')

    uploaded_files = st.file_uploader(
        'Upload one or more videos',
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button('Index videos', type='primary'):
            for uploaded in uploaded_files:
                video_bytes = bytes(uploaded.getbuffer())
                video_path = persist_video_bytes(
                    video_bytes=video_bytes,
                    original_name=uploaded.name,
                    library_root=library_root,
                )
                video_id = compute_video_id_from_bytes(video_bytes)

                st.session_state['active_video_path'] = str(video_path)
                st.session_state['active_video_id'] = video_id
                st.session_state['active_video_name'] = uploaded.name
                st.session_state['video_name_by_id'][video_id] = uploaded.name
                st.session_state['video_path_by_id'][video_id] = str(video_path)
                st.session_state['playback_video_id'] = video_id
                st.session_state['playback_start_time'] = 0

                cached = store.load(video_id)
                if cached:
                    st.info(f'Reused index for {uploaded.name}.')
                    continue

                with st.spinner(f'Extracting visual moments from {uploaded.name}...'):
                    duration_placeholder = 0.0
                    timestamps = select_visual_timestamps(
                        video_path=str(video_path),
                        duration_sec=duration_placeholder,
                        strategy=config.frame_strategy,
                        interval_sec=config.frame_interval_sec,
                        threshold=config.scene_threshold,
                        max_frames=config.max_visual_frames,
                    )
                    frames = extract_sampled_frames(
                        video_path=str(video_path),
                        video_id=video_id,
                        interval_sec=config.frame_interval_sec,
                        timestamps=timestamps,
                    )

                progress = st.progress(0.0, text=f'Indexing {uploaded.name}...')

                def _on_progress(done: int, total: int) -> None:
                    ratio = done / max(1, total)
                    progress.progress(ratio, text=f'Indexing {uploaded.name}... {done}/{total}')

                indexed, failures = asyncio.run(
                    index_frames_into_store(
                        video_id=video_id,
                        frames=frames,
                        store=store,
                        embed_image_fn=embedder.embed_image_base64,
                        concurrency=config.embed_concurrency,
                        max_retries=config.embed_max_retries,
                        base_backoff_sec=config.embed_backoff_sec,
                        on_progress=_on_progress,
                    )
                )
                st.session_state['last_index_failures'] = failures
                st.success(f'{uploaded.name}: indexed {len(indexed) - len(failures)} frame(s).')
                if failures:
                    st.warning(f'{uploaded.name}: failed frames {len(failures)}')

                chunks = []
                if enable_transcripts:
                    try:
                        with st.spinner(f'Transcribing {uploaded.name}...'):
                            chunks = transcribe_video(video_path=video_path, video_id=video_id, library_root=library_root)
                            awaitable = embed_transcripts_into_store(
                                video_id=video_id,
                                chunks=chunks,
                                store=store,
                                embed_text_fn=embedder.embed_text,
                            )
                            asyncio.run(awaitable)
                    except Exception as exc:
                        st.warning(f'{uploaded.name}: transcript indexing skipped ({exc})')

                metadata = _video_metadata(video_id=video_id, name=uploaded.name, path=video_path, frames=indexed, chunks=chunks)
                upsert_video_metadata(catalog, metadata)
                save_catalog(catalog_path, catalog)

        playback_video_id = st.session_state.get('playback_video_id')
        video_path_by_id = st.session_state.get('video_path_by_id', {})
        if playback_video_id and playback_video_id in video_path_by_id:
            playback_path = video_path_by_id[playback_video_id]
            video_bytes = Path(playback_path).read_bytes()
            st.video(video_bytes, start_time=int(st.session_state.get('playback_start_time', 0)))

            query = st.text_input('Search for a moment')
            if st.button('Search') and query.strip():
                results = asyncio.run(
                    search_library(
                        query=query.strip(),
                        store=store,
                        embed_text_fn=embedder.embed_text,
                        top_k=top_k,
                        config=config,
                    )
                )
                filtered = [result for result in results if result.score >= min_score]
                st.session_state['search_results'] = filtered

                if not filtered:
                    st.info(
                        f'No strong matches found at the current similarity threshold ({min_score:.2f}).'
                    )
            selected = render_unified_results(
                st=st,
                results=st.session_state.get('search_results', []),
                video_name_by_id=st.session_state.get('video_name_by_id', {}),
            )
            if selected:
                set_playback_target(st.session_state, video_id=selected[0], start_time=selected[1])


if __name__ == '__main__':
    main()
