from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import streamlit as st

from src.config import AppConfig
from src.embeddings import GeminiEmbedder
from src.index_store import ChromaIndexStore, HybridIndexStore, InMemoryIndexStore
from src.pipeline import index_frames_into_store, search_frames
from src.search import (
    filter_results_by_similarity,
    format_timestamp,
    group_top_result_per_video,
)
from src.video_processing import (
    compute_video_id_from_bytes,
    extract_sampled_frames,
    persist_uploaded_video,
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
    st.session_state.setdefault('active_video_path', None)
    st.session_state.setdefault('active_video_id', None)
    st.session_state.setdefault('active_video_name', None)
    st.session_state.setdefault('video_name_by_id', {})
    st.session_state.setdefault('video_path_by_id', {})
    st.session_state.setdefault('search_results', [])
    st.session_state.setdefault('last_index_failures', [])
    st.session_state.setdefault('playback_video_id', None)
    st.session_state.setdefault('playback_start_time', 0)


def _render_results() -> None:
    results = st.session_state.get('search_results', [])
    if not results:
        return

    st.subheader('Top Matches')
    columns = st.columns(len(results))
    for idx, result in enumerate(results):
        with columns[idx]:
            image_data = base64.b64decode(result.frame.thumbnail_b64)
            st.image(image_data, use_container_width=True)
            st.caption(
                f"{format_timestamp(result.frame.timestamp_sec)} - {round(result.similarity * 100)}%"
            )

            names = st.session_state.get('video_name_by_id', {})
            source = names.get(result.frame.video_id, result.frame.video_id[:8])
            st.caption(f"source: {source}")
            if st.button('Play', key=f"play_{result.frame.video_id}_{result.frame.frame_id}"):
                st.session_state['playback_video_id'] = result.frame.video_id
                st.session_state['playback_start_time'] = int(result.frame.timestamp_sec)


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

    uploaded_files = st.file_uploader(
        'Upload one or more videos',
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button('Index videos', type='primary'):
            temp_root = Path(tempfile.gettempdir()) / 'videosense_uploads'
            for uploaded in uploaded_files:
                video_bytes = bytes(uploaded.getbuffer())
                video_path = persist_uploaded_video(uploaded_file=uploaded, base_dir=temp_root)
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

                with st.spinner(f'Extracting frames from {uploaded.name}...'):
                    frames = extract_sampled_frames(
                        video_path=str(video_path),
                        video_id=video_id,
                        interval_sec=config.frame_interval_sec,
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

        playback_video_id = st.session_state.get('playback_video_id')
        video_path_by_id = st.session_state.get('video_path_by_id', {})
        if playback_video_id and playback_video_id in video_path_by_id:
            playback_path = video_path_by_id[playback_video_id]
            video_bytes = Path(playback_path).read_bytes()
            st.video(video_bytes, start_time=int(st.session_state.get('playback_start_time', 0)))

            query = st.text_input('Search for a moment')
            if st.button('Search') and query.strip():
                results = asyncio.run(
                    search_frames(
                        query=query.strip(),
                        store=store,
                        embed_text_fn=embedder.embed_text,
                        top_k=config.top_k,
                    )
                )
                filtered = filter_results_by_similarity(results, min_similarity=config.min_similarity)
                st.session_state['search_results'] = filtered

                grouped = group_top_result_per_video(filtered)
                if grouped:
                    st.subheader('Best Match Per Video')
                    for item in grouped:
                        names = st.session_state.get('video_name_by_id', {})
                        label = names.get(item.frame.video_id, item.frame.video_id[:8])
                        st.write(
                            f"{label}: {format_timestamp(item.frame.timestamp_sec)} ({round(item.similarity * 100)}%)"
                        )
                else:
                    st.info(
                        f'No strong matches found at the current similarity threshold ({config.min_similarity:.2f}).'
                    )
            _render_results()


if __name__ == '__main__':
    main()
