from __future__ import annotations

from app.vector_store import SearchResult


def select_context_results(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return []

    best_score = results[0].score
    selected = [results[0]]
    for result in results[1:]:
        if best_score >= 0.55:
            is_confident_support = result.score >= 0.18 and result.score >= best_score * 0.55
        else:
            is_confident_support = result.score >= 0.1 and result.score >= best_score * 0.5
        if is_confident_support:
            selected.append(result)
        if len(selected) == 3:
            break
    return selected
