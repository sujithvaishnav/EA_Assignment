"""
web_search.py
-------------
Hybrid web search fallback node: Tavily (keyword/AI-curated) + Exa (neural/semantic)
fused via Reciprocal Rank Fusion (RRF) -- same strategy as retrieve.py.

Triggered when:
  - retry_count >= 2  (vector store + rewriting exhausted)

Flow:
  web_search_node -> grade_documents -> generate -> check_hallucination
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.documents import Document

from rag_pipeline.state import RAGState

logger = logging.getLogger(__name__)

# RRF constant (same as retrieve.py)
RRF_K = 60
TOP_K = 5           # final docs to surface
PROVIDER_TOP_K = 5  # results to fetch per provider


# ── Provider: Tavily ──────────────────────────────────────────────────────────

def _search_tavily(query: str) -> list[dict[str, Any]]:
    """
    Tavily: AI-curated keyword search.
    Returns scored, LLM-ready text snippets (400-600 chars, often with code).
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("[WebSearch] TAVILY_API_KEY not set -- skipping Tavily")
            return []

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=PROVIDER_TOP_K,
            include_raw_content=False,
        )

        results = []
        for rank, r in enumerate(response.get("results", []), start=1):
            content = r.get("content", "").strip()
            if not content:
                continue
            results.append({
                "rank": rank,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": content,
                "tavily_score": r.get("score"),   # 0.0 - 1.0
                "source": "tavily",
            })

        logger.info("[WebSearch] Tavily returned %d results", len(results))
        return results

    except Exception as exc:
        logger.error("[WebSearch] Tavily error: %s", exc)
        return []


# ── Provider: Exa ─────────────────────────────────────────────────────────────

def _search_exa(query: str) -> list[dict[str, Any]]:
    """
    Exa: Neural semantic search over curated web index.
    Excellent for conceptual / documentation queries where exact wording differs.
    """
    try:
        from exa_py import Exa

        api_key = os.environ.get("EXA_API_KEY", "")
        if not api_key:
            logger.warning("[WebSearch] EXA_API_KEY not set -- skipping Exa")
            return []

        client = Exa(api_key=api_key)
        response = client.search(
            query,
            type="auto",           # auto selects keyword vs neural per query
            num_results=PROVIDER_TOP_K,
            contents={"highlights": True},
        )

        results = []
        for rank, r in enumerate(response.results, start=1):
            highlights = getattr(r, "highlights", None) or []
            content = " ".join(highlights).strip()
            if not content:
                content = getattr(r, "text", "") or ""
            if not content:
                continue
            results.append({
                "rank": rank,
                "title": r.title or "",
                "url": r.url or "",
                "content": content,
                "tavily_score": None,  # Exa does not expose scores
                "source": "exa",
            })

        logger.info("[WebSearch] Exa returned %d results", len(results))
        return results

    except Exception as exc:
        logger.error("[WebSearch] Exa error: %s", exc)
        return []


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    tavily_results: list[dict],
    exa_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over two ranked result lists.

    RRF(d) = sum_i [ 1 / (k + rank_i(d)) ]

    Pages appearing in BOTH lists get scores accumulated -- natural boost
    for URLs that keyword AND semantic search agree on.

    Args:
        tavily_results: Ranked list from Tavily.
        exa_results:    Ranked list from Exa.
        k:              RRF smoothing constant (default 60).

    Returns:
        Merged, deduplicated list sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}   # url -> accumulated RRF score
    merged: dict[str, dict] = {}        # url -> richest metadata entry

    for provider_results in [tavily_results, exa_results]:
        for item in provider_results:
            url = item["url"]
            rrf = 1.0 / (k + item["rank"])
            rrf_scores[url] = rrf_scores.get(url, 0.0) + rrf

            # Keep the entry with the longest (richest) content per URL
            if url not in merged or len(item["content"]) > len(merged[url]["content"]):
                merged[url] = item

    ranked = []
    for url, score in rrf_scores.items():
        entry = merged[url].copy()
        entry["rrf_score"] = round(score, 6)
        ranked.append(entry)

    ranked.sort(key=lambda x: x["rrf_score"], reverse=True)
    return ranked


# ── Overlap detection helper ──────────────────────────────────────────────────

def _get_overlap_tag(
    url: str,
    tavily_results: list[dict],
    exa_results: list[dict],
    source: str,
) -> str:
    in_tavily = any(r["url"] == url for r in tavily_results)
    in_exa = any(r["url"] == url for r in exa_results)
    if in_tavily and in_exa:
        return "[BOTH]"
    return f"[{source.upper()}]"


# ── Node: web_search_node ─────────────────────────────────────────────────────

def web_search_node(state: RAGState) -> dict:
    """
    LangGraph node: Hybrid web search fallback.

    Runs Tavily (keyword) + Exa (semantic) searches in parallel,
    fuses results via RRF, deduplicates by URL, and injects the
    top-k documents back into state["documents"] for the grader.

    retry_count is reset to 0 so the grader evaluates web results
    as a fresh batch (preventing immediate no_answer short-circuit).
    """
    query = state.get("rewritten_query") or state.get("question", "")

    print(f"[WebSearch] Fallback triggered -- query: {query[:80]}")
    print("[WebSearch] Fetching Tavily (keyword) + Exa (semantic) in parallel...")

    # ── Parallel fetch ────────────────────────────────────────────────────────
    tavily_results: list[dict] = []
    exa_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_search_tavily, query): "tavily",
            executor.submit(_search_exa, query): "exa",
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                result = future.result()
                if provider == "tavily":
                    tavily_results = result
                else:
                    exa_results = result
            except Exception as exc:
                logger.error("[WebSearch] %s future raised: %s", provider, exc)

    print(
        f"[WebSearch] Raw -- Tavily: {len(tavily_results)}, "
        f"Exa: {len(exa_results)}"
    )

    # ── Handle total failure ──────────────────────────────────────────────────
    if not tavily_results and not exa_results:
        print("[WebSearch] Both providers empty -- pipeline will return no_answer")
        return {
            "documents": [],
            "web_search_used": True,
            "retry_count": 99,  # sentinel: grader router will hit no_answer branch
        }

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    fused = _rrf_fuse(tavily_results, exa_results, k=RRF_K)
    top = fused[:TOP_K]

    print(f"[WebSearch] After RRF fusion -- top {len(top)} unique results:")
    for i, r in enumerate(top, 1):
        tag = _get_overlap_tag(r["url"], tavily_results, exa_results, r["source"])
        print(f"  {i}. RRF={r['rrf_score']:.4f} {tag} {r['url'][:65]}")

    # ── Convert to LangChain Documents ────────────────────────────────────────
    docs = []
    for r in top:
        doc = Document(
            page_content=r["content"],
            metadata={
                "source": r["url"],
                "title": r["title"],
                "rrf_score": r["rrf_score"],
                "tavily_score": r.get("tavily_score"),
                "search_provider": r["source"],
                "web_result": True,
            },
        )
        docs.append(doc)

    print(f"[WebSearch] Injecting {len(docs)} web docs into pipeline state")

    return {
        "documents": docs,
        "web_search_used": True,
        "retry_count": 0,   # reset so grader evaluates web docs as fresh batch
    }
