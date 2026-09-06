"""
Conversation memory management.

Wraps LangChain's message history to maintain multi-turn context
within a single agent session.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Lightweight in-process conversation history store.

    Stores messages as a list of LangChain BaseMessage objects so they
    can be injected directly into LLM calls.
    """

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []
        logger.debug("ConversationMemory initialised.")

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        logger.debug("Memory: added user message (%d chars).", len(content))

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        logger.debug("Memory: added AI message (%d chars).", len(content))

    def add_system_message(self, content: str) -> None:
        """Prepend a system message (typically called once at session start)."""
        self._messages.insert(0, SystemMessage(content=content))
        logger.debug("Memory: added system message.")

    def clear(self) -> None:
        self._messages.clear()
        logger.info("Conversation history cleared.")

    # ── Access ────────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Number of human turns so far."""
        return sum(1 for m in self._messages if isinstance(m, HumanMessage))

    def format_for_display(self) -> str:
        """Return a human-readable transcript of the conversation."""
        lines: list[str] = []
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Agent: {msg.content}")
        return "\n\n".join(lines)
