"""
LangGraph node functions and routing logic for the CV QA Agent.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from cv_agent.agent.state import AgentState
from cv_agent.config import settings

logger = logging.getLogger(__name__)

NOT_AVAILABLE_MESSAGE = "This information is not available in the provided CV."

# ── Grounding System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful CV assistant. Your ONLY knowledge source is the \
candidate's CV, which you access through the `search_cv` tool and the retrieved excerpts in context.

STRICT RULES — you MUST follow these at all times:
1. NEVER answer from memory, training data, or external knowledge.
2. ALWAYS use the retrieved CV excerpts in the context to form your answer.
3. If the retrieved context does not contain enough information to answer the question, \
respond EXACTLY with:
   "This information is not available in the provided CV."
4. Prefix every factual answer with "Based on the CV, ..." or "From the CV, ...".
5. You may reason about and synthesise information from multiple retrieved excerpts, \
but do not invent or infer facts not stated in the excerpts.
6. For follow-up questions, use the conversation history to understand context, \
and use the retrieved evidence to formulate your response.
"""


def _get_llm():
    """Build and return the configured LLM."""
    provider = settings.llm_provider

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please set GOOGLE_API_KEY in your .env file "
                "or set it as an environment variable."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required. "
                "Install with: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.temperature,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Please set OPENAI_API_KEY in your .env file "
                "or set it as an environment variable."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required.") from exc
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )

    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise ImportError("langchain-community is required for Ollama.") from exc
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    raise ValueError(f"Unknown llm_provider: {provider!r}")


def _extract_text(content: Any) -> str:
    """Safely extract string content from a string, list of blocks, or object."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


# ── Node 1: Contextualize Query ──────────────────────────────────────────────

def contextualize_query_node(state: AgentState) -> dict[str, Any]:
    """
    Rewrite follow-up or pronoun-heavy questions into a standalone search query.
    If the question is already self-contained or it is the first turn, passes it through.
    """
    messages = state.get("messages", [])
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if not human_messages:
        return {"contextualized_query": ""}

    latest_question = _extract_text(human_messages[-1].content)

    # First question (no prior human turns): question is already standalone
    if len(human_messages) <= 1:
        logger.info("CONTEXTUALIZE | First turn -> query=%r", latest_question)
        return {"contextualized_query": latest_question}

    # Multi-turn: ask LLM to contextualize
    try:
        llm = _get_llm()
        prompt = (
            "Given the chat history and the latest user question, rephrase the latest question "
            "into a standalone search query that can be used to search a candidate's CV. "
            "Do NOT answer the question. Only return the rephrased search query. "
            "If the question is already standalone, return NO_REWRITE.\n\n"
            f"Latest question: {latest_question}"
        )
        response = llm.invoke(messages[:-1] + [HumanMessage(content=prompt)])
        rephrased = _extract_text(getattr(response, "content", "")).strip()
        if not rephrased or "NO_REWRITE" in rephrased:
            logger.info("CONTEXTUALIZE | Self-contained -> query=%r", latest_question)
            return {"contextualized_query": latest_question}

        logger.info("CONTEXTUALIZE | Rephrased %r -> %r", latest_question, rephrased)
        return {"contextualized_query": rephrased}
    except Exception as exc:
        logger.warning("CONTEXTUALIZE | Failed to rephrase (%s) -> using raw question", exc)
        return {"contextualized_query": latest_question}


# ── Node 2: Mandatory Retrieval ──────────────────────────────────────────────

def mandatory_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Always executes search_cv before LLM generation to guarantee grounded context.
    """
    from cv_agent.tools.cv_search import search_cv

    query = state.get("contextualized_query")
    if not query:
        human_messages = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        query = _extract_text(human_messages[-1].content) if human_messages else ""

    logger.info("MANDATORY_RETRIEVAL | query=%r", query)
    t0 = time.perf_counter()

    try:
        raw_result = search_cv.invoke({"query": query})
    except Exception as exc:
        logger.error("MANDATORY_RETRIEVAL | Tool error: %s", exc, exc_info=True)
        raw_result = json.dumps({
            "status": "unavailable",
            "best_score": 9999.0,
            "documents": [],
            "formatted": ""
        })

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    try:
        parsed = json.loads(raw_result)
    except Exception:
        parsed = {"status": "unavailable", "best_score": 9999.0, "documents": [], "formatted": ""}

    status = parsed.get("status", "unavailable")
    best_score = parsed.get("best_score", 9999.0)
    documents = parsed.get("documents", [])
    formatted_text = parsed.get("formatted", "")

    logger.info(
        "MANDATORY_RETRIEVAL | status=%s | chunks=%d | best_score=%.4f | latency_ms=%s",
        status,
        len(documents),
        best_score,
        latency_ms,
    )

    tool_call_id = "call_mandatory_retrieval"
    ai_tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_cv", "args": {"query": query}, "id": tool_call_id}],
    )
    tool_msg = ToolMessage(
        content=formatted_text if status == "relevant" else NOT_AVAILABLE_MESSAGE,
        tool_call_id=tool_call_id,
    )

    return {
        "messages": state.get("messages", []) + [ai_tool_call_msg, tool_msg],
        "retrieved_documents": documents,
        "retrieval_score": best_score,
        "grounding_status": "grounded" if status == "relevant" else "unavailable",
        "iteration": 1,
    }


def mandatory_retrieval_routing(state: AgentState) -> str:
    """Route after mandatory retrieval: fallback if unavailable, else agent."""
    if state.get("grounding_status") == "unavailable":
        return "fallback"
    return "agent"


# ── Node 3: Agent Reasoning ──────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict[str, Any]:
    """
    Main LLM synthesis node.
    """
    from cv_agent.tools.cv_search import search_cv

    llm = _get_llm()
    llm_with_tools = llm.bind_tools([search_cv])

    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)

    logger.info("AGENT | Invoked | turn=%d | messages_in_context=%d", iteration + 1, len(messages))
    t0 = time.perf_counter()

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.error("AGENT | LLM call failed: %s", exc, exc_info=True)
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            logger.info(
                "AGENT | tool call requested | tool=%s | args=%s | latency_ms=%s",
                tc.get("name", "unknown"),
                json.dumps(tc.get("args", {}), ensure_ascii=False),
                latency_ms,
            )
    else:
        content_text = _extract_text(getattr(response, "content", ""))
        logger.info(
            "AGENT | final answer ready | length=%d chars | latency_ms=%s",
            len(content_text),
            latency_ms,
        )

    return {
        "messages": state["messages"] + [response],
        "iteration": iteration + 1,
    }


def should_continue(state: AgentState) -> str:
    """
    Decide whether to execute more tools or perform the grounding check.
    """
    last_message = state["messages"][-1]
    iteration = state.get("iteration", 0)

    if iteration >= settings.agent_max_iterations:
        logger.warning("AGENT | Max iterations (%d) reached", settings.agent_max_iterations)
        return "check_grounding"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "check_grounding"


# ── Node 4: Tools Execution ──────────────────────────────────────────────────

def tool_node(state: AgentState) -> dict[str, Any]:
    """
    Execute tool calls requested by the agent.
    """
    from cv_agent.tools.cv_search import search_cv

    tool_registry = {"search_cv": search_cv}
    last_message = state["messages"][-1]
    tool_results: list[ToolMessage] = []
    state_updates: dict[str, Any] = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = dict(tool_call["args"])
        tool_call_id = tool_call["id"]

        # If contextualized_query is present in state, prioritize it for search_cv
        if tool_name == "search_cv":
            contextualized = state.get("contextualized_query", "")
            if contextualized:
                tool_args = {"query": contextualized}

        logger.info("TOOL | executing | tool=%s | args=%s", tool_name, tool_args)
        t0 = time.perf_counter()

        if tool_name not in tool_registry:
            error_msg = f"Unknown tool requested: {tool_name}"
            logger.error("TOOL | %s", error_msg)
            tool_results.append(ToolMessage(content=f"Error: {error_msg}", tool_call_id=tool_call_id))
            continue

        try:
            raw_result = tool_registry[tool_name].invoke(tool_args)
        except Exception as exc:
            logger.error("TOOL | %s raised an error: %s", tool_name, exc, exc_info=True)
            tool_results.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=tool_call_id))
            continue

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if tool_name == "search_cv":
            try:
                parsed = json.loads(raw_result)
                status = parsed.get("status", "unavailable")
                score = parsed.get("best_score", 999.0)
                docs = parsed.get("documents", [])
                formatted_text = parsed.get("formatted", "")

                logger.info(
                    "RETRIEVAL | status=%s | chunks=%d | best_score=%.4f | latency_ms=%s",
                    status,
                    len(docs),
                    score,
                    latency_ms,
                )

                state_updates["retrieved_documents"] = state.get("retrieved_documents", []) + docs
                state_updates["retrieval_score"] = score
                state_updates["grounding_status"] = "grounded" if status == "relevant" else "unavailable"

                tool_results.append(
                    ToolMessage(
                        content=formatted_text or NOT_AVAILABLE_MESSAGE,
                        tool_call_id=tool_call_id,
                    )
                )
            except Exception:
                tool_results.append(ToolMessage(content=str(raw_result), tool_call_id=tool_call_id))
        else:
            tool_results.append(ToolMessage(content=str(raw_result), tool_call_id=tool_call_id))

    return {
        "messages": state["messages"] + tool_results,
        **state_updates,
    }


def after_tools_routing(state: AgentState) -> str:
    """Route after additional tools execution."""
    if state.get("grounding_status") == "unavailable":
        return "fallback"
    return "agent"


# ── Node 5: Fallback Node ────────────────────────────────────────────────────

def fallback_node(state: AgentState) -> dict[str, Any]:
    """Short-circuit fallback when information is unavailable in the CV."""
    logger.info("FALLBACK | Returning standard unavailable message")
    messages = state.get("messages", [])
    return {
        "messages": messages + [AIMessage(content=NOT_AVAILABLE_MESSAGE)],
        "final_answer": NOT_AVAILABLE_MESSAGE,
        "grounding_status": "unavailable",
    }


# ── Node 6: Grounding Check Node ─────────────────────────────────────────────

def check_grounding_node(state: AgentState) -> dict[str, Any]:
    """
    Validate that the generated response is strictly grounded in the retrieved CV context.
    """
    if state.get("grounding_status") == "unavailable":
        return {}

    docs = state.get("retrieved_documents", [])
    if not docs:
        logger.info("CHECK_GROUNDING | No retrieved documents -> forcing fallback")
        messages = state.get("messages", [])
        return {
            "messages": messages[:-1] + [AIMessage(content=NOT_AVAILABLE_MESSAGE)] if messages else [AIMessage(content=NOT_AVAILABLE_MESSAGE)],
            "final_answer": NOT_AVAILABLE_MESSAGE,
            "grounding_status": "not_grounded",
        }

    last_message = state["messages"][-1]
    raw_answer = _extract_text(getattr(last_message, "content", ""))

    # If the LLM already stated information is not available, respect it
    if "not available in the provided cv" in raw_answer.lower() or "not available" in raw_answer.lower():
        return {
            "final_answer": NOT_AVAILABLE_MESSAGE,
            "grounding_status": "unavailable",
        }

    # Call LLM to validate grounding
    try:
        llm = _get_llm()
        docs_text = "\n\n".join(docs)
        prompt = (
            f"Given the retrieved CV documents:\n{docs_text}\n\n"
            f"And the generated answer:\n{raw_answer}\n\n"
            "Is the generated answer strictly grounded in the retrieved CV documents? "
            'Return a JSON object: {"grounded": true, "reason": "..."} or {"grounded": false, "reason": "..."}'
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = _extract_text(getattr(response, "content", "")).strip()
        clean_content = content
        if "```" in clean_content:
            parts = clean_content.split("```")
            if len(parts) >= 2:
                clean_content = parts[1]
                if clean_content.startswith("json"):
                    clean_content = clean_content[4:]
        clean_content = clean_content.strip()
        parsed = json.loads(clean_content)
        is_grounded = parsed.get("grounded", True)
    except Exception as exc:
        logger.warning("CHECK_GROUNDING | Validation check failed (%s) -> passing through", exc)
        is_grounded = True

    if not is_grounded:
        logger.info("CHECK_GROUNDING | Answer not grounded -> replacing with fallback")
        messages = state.get("messages", [])
        return {
            "messages": messages[:-1] + [AIMessage(content=NOT_AVAILABLE_MESSAGE)] if messages else [AIMessage(content=NOT_AVAILABLE_MESSAGE)],
            "final_answer": NOT_AVAILABLE_MESSAGE,
            "grounding_status": "not_grounded",
        }

    logger.info("CHECK_GROUNDING | Grounded answer verified (%d chars)", len(raw_answer))
    return {
        "final_answer": raw_answer,
        "grounding_status": "grounded",
    }
