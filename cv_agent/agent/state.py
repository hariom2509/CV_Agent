"""
AgentState — typed dictionary used as the LangGraph state schema.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    The shared state passed between all nodes in the LangGraph graph.

    Attributes:
        messages:             Full message history (system + human + AI + tool messages).
        iteration:            Number of agent→tool cycles completed in this turn.
        contextualized_query: Standalone rewritten query (set by contextualize_query node).
        retrieved_documents:  Raw chunk texts returned by the last search_cv call.
        retrieval_score:      Best (lowest) L2 distance score from the last retrieval.
        grounding_status:     One of "grounded", "not_grounded", "unavailable", or None.
        final_answer:         The validated, finalized answer text.
    """

    messages: list[BaseMessage]
    iteration: int
    contextualized_query: str
    retrieved_documents: list[str]
    retrieval_score: float
    grounding_status: str   # "grounded" | "not_grounded" | "unavailable"
    final_answer: str
