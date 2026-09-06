"""CV QA Agent — bootstrap and runner."""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage

from cv_agent.agent.graph import build_graph
from cv_agent.agent.memory import ConversationMemory
from cv_agent.agent.nodes import SYSTEM_PROMPT
from cv_agent.config import settings
from cv_agent.ingestion.loader import load_cv
from cv_agent.indexing.vector_store import build_or_load_index, compute_cv_hash, get_retriever
from cv_agent.logging_setup import configure_logging
from cv_agent.output.writer import SessionWriter
from cv_agent.tools.cv_search import set_vector_store

logger = logging.getLogger(__name__)


class CVAgent:
    """
    High-level facade for the CV QA agent.

    Encapsulates startup (ingestion, indexing, graph compilation) and
    exposes a simple `ask(question)` method for interactive use.
    """

    def __init__(self, cv_paths: list[Path | str] | Path | str | None = None) -> None:
        configure_logging(
            level=settings.log_level,
            log_file=settings.log_file,
        )
        self._cv_paths = cv_paths or settings.cv_file_path
        logger.info("=" * 60)
        logger.info("CV QA Agent starting up…")
        logger.info("Provider: %s | Model: %s", settings.llm_provider, settings.llm_model)
        logger.info("CV target: %s", self._cv_paths)
        logger.info(
            "Retrieval: top_k=%d | score_threshold=%.4f",
            settings.retrieval_top_k,
            settings.retrieval_score_threshold,
        )

        # ── Ingestion + Indexing ───────────────────────────────────────────────
        self._setup_retriever()

        # ── Build LangGraph ───────────────────────────────────────────────────
        self._graph = build_graph()

        # ── Memory + Output ───────────────────────────────────────────────────
        self._memory = ConversationMemory()
        self._memory.add_system_message(SYSTEM_PROMPT)
        self._writer = SessionWriter()

        logger.info("AGENT | Ready. Awaiting questions.")

    def _setup_retriever(self) -> None:
        """
        Determine whether to load an existing FAISS index or rebuild.

        Uses SHA-256 CV hashing across all target files to detect stale indexes.
        The raw FAISS store is injected into the search_cv tool.
        """
        index_path = settings.faiss_index_path
        target_paths = self._cv_paths

        from cv_agent.ingestion.loader import load_multiple_cvs

        # Try to load; if stale or missing, ingestion kicks in automatically
        needs_ingestion = not index_path.exists()

        if not needs_ingestion:
            logger.info(
                "INDEX | Existing index found at '%s'. Checking freshness…", index_path
            )
            try:
                store = build_or_load_index(chunks=None, cv_path=target_paths)
                set_vector_store(store)
                return
            except ValueError:
                logger.info("INDEX | Index stale — running multi-CV ingestion pipeline")
                needs_ingestion = True

        if needs_ingestion:
            logger.info("INGESTION | Starting multi-CV ingestion pipeline")
            chunks = load_multiple_cvs(target_paths)
            store = build_or_load_index(chunks=chunks, cv_path=target_paths)

        set_vector_store(store)

    def ask(self, question: str) -> str:
        """
        Process a user question and return the agent's grounded answer.

        Args:
            question: Natural-language question about the CV.

        Returns:
            The agent's text response.
        """
        logger.info("AGENT | Received question: %r", question)

        self._memory.add_user_message(question)

        initial_state = {
            "messages": self._memory.messages,
            "iteration": 0,
            "contextualized_query": "",
            "retrieved_documents": [],
            "grounding_status": None,
            "final_answer": "",
        }

        try:
            result = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("AGENT | Graph invocation failed: %s", exc, exc_info=True)
            answer = (
                "An internal error occurred while processing your question. "
                "Please check the logs for details."
            )
            self._memory.add_ai_message(answer)
            self._writer.record(question=question, answer=answer)
            return answer

        # Prefer final_answer (set by grounding/fallback nodes) over raw last message
        final_answer: str = result.get("final_answer", "")
        if not final_answer:
            last_message = result["messages"][-1]
            final_answer = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

        grounding_status = result.get("grounding_status", "unknown")
        logger.info(
            "OUTPUT | Answer ready | grounding_status=%s | length=%d chars",
            grounding_status,
            len(final_answer),
        )

        self._memory.add_ai_message(final_answer)
        self._writer.record(question=question, answer=final_answer)

        return final_answer

    def save_session(self) -> None:
        """Persist the session Q&A to disk."""
        out = self._writer.save()
        logger.info("OUTPUT | Session saved to: %s", out)
        print(f"\n📄 Session saved to: {out}")

    def clear_history(self) -> None:
        """Reset conversation history (keeps system prompt)."""
        self._memory.clear()
        self._memory.add_system_message(SYSTEM_PROMPT)
        logger.info("AGENT | Conversation history cleared")

    @property
    def turn_count(self) -> int:
        return self._memory.turn_count
