from __future__ import annotations

import base64
from typing import Any

from src.models import UnifiedSearchResult
from src.search import format_timestamp


def render_unified_results(st: Any, results: list[UnifiedSearchResult], video_name_by_id: dict[str, str]) -> tuple[str, int] | None:
    if not results:
        return None

    st.subheader('Top Matches')
    columns = st.columns(len(results))
    selected: tuple[str, int] | None = None
    for idx, result in enumerate(results):
        with columns[idx]:
            if result.frame:
                image_data = base64.b64decode(result.frame.thumbnail_b64)
                st.image(image_data, use_container_width=True)
            if result.transcript:
                st.caption(result.transcript.text)

            source = video_name_by_id.get(result.video_id, result.video_id[:8])
            st.caption(
                f"{format_timestamp(result.start_sec)} - {result.channel} match - {round(result.score * 100)}%"
            )
            st.caption(f"source: {source}")
            if st.button('Play', key=f"play_{result.video_id}_{idx}_{int(result.start_sec)}"):
                selected = (result.video_id, int(result.start_sec))
    return selected
