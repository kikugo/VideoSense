from src.models import FrameRecord, SearchResult
from src.search import filter_results_by_similarity


def _result(similarity: float) -> SearchResult:
    return SearchResult(
        frame=FrameRecord('v1', f'f{int(similarity*100)}', 0.0, 'thumb', [0.1]),
        similarity=similarity,
    )


def test_filter_results_by_similarity_keeps_strong_matches():
    results = [_result(0.91), _result(0.49), _result(0.65)]

    filtered = filter_results_by_similarity(results, min_similarity=0.6)

    assert [item.similarity for item in filtered] == [0.91, 0.65]


def test_filter_results_by_similarity_returns_empty_when_none_qualify():
    results = [_result(0.2), _result(0.3)]

    filtered = filter_results_by_similarity(results, min_similarity=0.4)

    assert filtered == []
