"""Retrieval, and the reason it is an ablation arm rather than a default.

Retrieval is the first thing anyone adds to a forecaster and among the last things anyone
measures. It is plausible on its face -- more information should help -- and it has a specific
failure mode that makes the plausibility misleading: search results for a question that has
already resolved often contain the answer, so retrieval can look like skill while actually
leaking the outcome.

So evidence is off by default, exposed as an arm, and the snippets it returns are recorded on
the forecast. `ablate` reports what it bought, and the contamination probe is what keeps the
answer honest.
"""
from __future__ import annotations

import re

MAX_SNIPPETS = 5
MAX_CHARS = 1800


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def search(question: str, *, api_key: str, max_results: int = MAX_SNIPPETS,
           days: int | None = None) -> str | None:
    """Compact evidence block, or None when retrieval is unavailable.

    Returning None rather than an empty string matters: an empty string would make the
    forecast record claim evidence was used when nothing was retrieved, and the evidence arm
    would then be silently identical to the plain arm while appearing to be a separate
    measurement.
    """
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        kw = {"query": question, "max_results": max_results,
              "search_depth": "basic", "include_answer": False}
        if days:
            kw["days"] = days
        res = client.search(**kw)
    except Exception:
        return None
    items = res.get("results") or []
    if not items:
        return None
    lines = []
    for i, it in enumerate(items[:max_results], 1):
        title = _clean(it.get("title", ""))[:120]
        body = _clean(it.get("content", ""))[:320]
        lines.append(f"[{i}] {title} -- {body}")
    blob = "\n".join(lines)[:MAX_CHARS]
    return blob or None


def offline_evidence(question: str) -> str:
    """Deterministic stand-in so the evidence ARM exists on the mock path.

    It carries no information about the answer on purpose. An offline stub that helped would
    make the evidence arm win in CI and lose in production, which is worse than not having the
    arm at all.
    """
    return (f"[1] Background -- no external sources were retrieved for this run.\n"
            f"[2] Context -- question under consideration: {_clean(question)[:160]}")
