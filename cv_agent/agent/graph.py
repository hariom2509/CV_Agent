"""
LangGraph state graph definition for the CV QA Agent.

Graph topology (per user turn):
    START
      ↓
    contextualize_query    — rewrite follow-ups; pass self-contained Qs unchanged
      ↓
    mandatory_retrieval    — ALWAYS calls search_cv before generation (enforced)
      ↓
    agent                  — LLM synthesises; may call search_cv again for follow-ups
      ↓
    should_continue?
      ├── "tools"           → tool_node
      │                           ↓
      │                     after_tools_routing?
      │                       ├── "unavailable" → fallback → END
      │                       └── "agent"       → agent (loop)
      │
      └── "check_grounding" → check_grounding → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from cv_agent.agent.nodes import (
    after_tools_routing,
    agent_node,
    check_grounding_node,
    contextualize_query_node,
    fallback_node,
    mandatory_retrieval_node,
    mandatory_retrieval_routing,
    should_continue,
    tool_node,
)
from cv_agent.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph():
    """
    Compile and return the LangGraph state machine.

    The graph implements a multi-stage pipeline:
      1. Contextualize: rewrite ambiguous follow-up questions.
      2. Mandatory Retrieval: always call search_cv before generation.
      3. Agent: LLM reasoning; may call search_cv again for follow-ups.
      4. Tools: execute additional search_cv calls; parse structured results.
      5. Routing: if retrieval was irrelevant → fallback; else → agent.
      6. Grounding: validate the answer against retrieved context.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────────────────────────
    graph.add_node("contextualize", contextualize_query_node)
    graph.add_node("mandatory_retrieval", mandatory_retrieval_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("check_grounding", check_grounding_node)

    # ── Add edges ───────────────────────────────────────────────────────────────────────
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "mandatory_retrieval")

    # After mandatory retrieval: short-circuit to fallback if nothing found
    graph.add_conditional_edges(
        "mandatory_retrieval",
        mandatory_retrieval_routing,
        {
            "fallback": "fallback",
            "agent": "agent",
        },
    )

    # After agent: either call tools or go directly to grounding check
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "check_grounding": "check_grounding",
        },
    )

    # After tools: route based on retrieval relevance
    graph.add_conditional_edges(
        "tools",
        after_tools_routing,
        {
            "fallback": "fallback",
            "agent": "agent",
        },
    )

    graph.add_edge("fallback", END)
    graph.add_edge("check_grounding", END)

    compiled = graph.compile()
    logger.info("AGENT | LangGraph state machine compiled successfully")
    return compiled
