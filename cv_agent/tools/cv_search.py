"""
CV Search Tool — the agent's primary retrieval capability.

Exposes a LangChain @tool that the LangGraph agent can invoke to look up
relevant passages from the indexed CV.

Design:
  - search_cv returns a JSON string so the graph's tool_node can inspect
    retrieval metadata (status, best_score, documents) before deciding
    whether to route to the agent or directly to the fallback node.
  - The tool never silently returns "not available" strings — that routing
    decision belongs to the graph orchestration layer, not the tool.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from langchain.tools import tool

from cv_agent.config import settings

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ── Module-level store reference — injected at bootstrap ─────────────────────
_vector_store: "FAISS | None" = None


def set_vector_store(store: "FAISS") -> None:
    """
    Inject the FAISS vector store. Called once during app startup.
    Replaces the older set_retriever() approach so the tool can use
    similarity_search_with_score for relevance gating.
    """
    global _vector_store
    _vector_store = store
    logger.info(
        "TOOL | CV search tool configured | top_k=%d | score_threshold=%.4f",
        settings.retrieval_top_k,
        settings.retrieval_score_threshold,
    )


@tool
def search_cv(query: str) -> str:
    """
    Search the CV for information relevant to the given query.

    Use this tool whenever you need to find facts about the candidate's
    education, work experience, skills, certifications, projects, or any
    other detail that may appear in the CV.

    Args:
        query: A natural-language search query.

    Returns:
        A JSON string with fields:
          - status: "relevant" or "unavailable"
          - best_score: lowest L2 distance found (lower = more relevant)
          - documents: list of raw chunk texts (empty if unavailable)
          - formatted: human-readable formatted excerpts (empty if unavailable)
    """
    if _vector_store is None:
        raise RuntimeError(
            "CV vector store has not been initialised. "
            "Call set_vector_store() before using this tool."
        )

    logger.info("TOOL | search_cv invoked | query=%r", query)
    t0 = time.perf_counter()

    max_retries = 5
    results = None
    last_exc = None
    delays = [2.0, 4.0, 8.0, 15.0, 25.0]

    for attempt in range(1, max_retries + 1):
        try:
            results = _vector_store.similarity_search_with_score(
                query, k=settings.retrieval_top_k
            )
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                delay = delays[attempt - 1] if attempt <= len(delays) else 10.0
                logger.warning(
                    "RETRIEVAL | Rate limit (429) on attempt %d/%d — retrying in %.1fs...",
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("RETRIEVAL | Failed for query=%r: %s", query, exc)
                raise RuntimeError(f"CV search failed: {exc}") from exc

    if results is None:
        logger.error("RETRIEVAL | Failed after %d attempts: %s", max_retries, last_exc)
        raise RuntimeError(f"CV search failed after retries: {last_exc}") from last_exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not results:
        logger.warning(
            "RETRIEVAL | No chunks returned | query=%r | latency_ms=%s",
            query,
            latency_ms,
        )
        return json.dumps(
            {"status": "unavailable", "best_score": 9999.0, "documents": [], "formatted": ""}
        )

    # FAISS with L2 distance: lower score = closer match
    best_score: float = float(results[0][1])
    threshold: float = float(settings.retrieval_score_threshold)
    is_relevant: bool = best_score <= threshold

    logger.info(
        "RETRIEVAL | chunks=%d | best_score=%.4f | threshold=%.4f | relevant=%s | latency_ms=%s",
        len(results),
        best_score,
        threshold,
        is_relevant,
        latency_ms,
    )

    if not is_relevant:
        logger.warning(
            "RETRIEVAL | Score %.4f exceeds threshold %.4f — marking unavailable",
            best_score,
            threshold,
        )
        return json.dumps(
            {
                "status": "unavailable",
                "best_score": best_score,
                "documents": [],
                "formatted": "",
            }
        )

    # ── Build structured result ───────────────────────────────────────────────
    documents: list[str] = []
    parts: list[str] = []

    for i, (doc, score) in enumerate(results, start=1):
        score_val = float(score)
        documents.append(doc.page_content)

        candidate = doc.metadata.get("candidate", "")
        filename = doc.metadata.get("filename", doc.metadata.get("source", "CV"))
        page = doc.metadata.get("page", "")
        section = doc.metadata.get("section", "")
        chunk_id = doc.metadata.get("chunk_id", i)

        header_parts = []
        if candidate:
            header_parts.append(f"Candidate: {candidate}")
        header_parts.append(f"File: {filename}")
        if section:
            header_parts.append(section)
        if page != "":
            header_parts.append(f"page {page + 1}")
        header_parts.append(f"chunk #{chunk_id}")

        location = " > ".join(header_parts)
        parts.append(f"[Excerpt {i} — {location} | score={score_val:.4f}]\n{doc.page_content.strip()}")

    formatted = "\n\n---\n\n".join(parts)
    logger.debug("RETRIEVAL | formatted context:\n%s", formatted)

    return json.dumps(
        {
            "status": "relevant",
            "best_score": best_score,
            "documents": documents,
            "formatted": formatted,
        }
    )
