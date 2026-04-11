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
from src.search import format_timestamp
from src.video_processing import compute_video_id, extract_sampled_frames, persist_uploaded_video


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
    st.session_state.setdefault('search_results', [])
    st.session_state.setdefault('last_index_failures', [])


def _render_results(video_bytes: bytes) -> None:
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

    selected = st.selectbox(
        'Play from match',
        options=list(range(len(results))),
        format_func=lambda i: f"{i + 1}. {format_timestamp(results[i].frame.timestamp_sec)}",
    )
    start_time = int(results[selected].frame.timestamp_sec)
    st.video(video_bytes, start_time=start_time)


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

    uploaded = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'mkv', 'webm'])

    if uploaded is not None:
        if st.button('Index video', type='primary'):
            temp_root = Path(tempfile.gettempdir()) / 'videosense_uploads'
            video_path = persist_uploaded_video(uploaded_file=uploaded, base_dir=temp_root)
            video_id = compute_video_id(str(video_path))

            st.session_state['active_video_path'] = str(video_path)
            st.session_state['active_video_id'] = video_id
            st.session_state['active_video_name'] = uploaded.name

            cached = store.load(video_id)
            if cached:
                st.success(f'Loaded existing index for {uploaded.name}.')
            else:
                with st.spinner('Extracting frames...'):
                    frames = extract_sampled_frames(
                        video_path=str(video_path),
                        video_id=video_id,
                        interval_sec=config.frame_interval_sec,
                    )

                progress = st.progress(0.0, text='Indexing frames...')

                def _on_progress(done: int, total: int) -> None:
                    ratio = done / max(1, total)
                    progress.progress(ratio, text=f'Indexing frames... {done}/{total}')

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
                st.success(f'Indexed {len(indexed) - len(failures)} frame(s).')
                if failures:
                    st.warning(f'Failed frames: {len(failures)}')

        if st.session_state.get('active_video_path'):
            video_bytes = Path(st.session_state['active_video_path']).read_bytes()
            st.video(video_bytes)

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
                st.session_state['search_results'] = results

            _render_results(video_bytes=video_bytes)


if __name__ == '__main__':
    main()
