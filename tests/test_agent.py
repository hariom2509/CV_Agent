"""
Tests for the agent memory, output writer, and grounding behaviour.

All tests are pure-Python — no API keys or live LLM calls required.
LLM and vector store dependencies are mocked where needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ─────────────────────────────────────────────────────────────────────────────
# ConversationMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_add_and_retrieve_messages(self):
        from cv_agent.agent.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_message("What is your experience?")
        mem.add_ai_message("Based on the CV, I have 5 years of experience.")

        messages = mem.messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_system_message_prepended(self):
        from cv_agent.agent.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_message("Q1")
        mem.add_system_message("You are an assistant.")

        messages = mem.messages
        assert isinstance(messages[0], SystemMessage)

    def test_clear_resets_messages(self):
        from cv_agent.agent.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_ai_message("Hi!")
        mem.clear()

        assert mem.messages == []
        assert mem.turn_count == 0

    def test_turn_count_increments(self):
        from cv_agent.agent.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_message("Q1")
        mem.add_ai_message("A1")
        mem.add_user_message("Q2")
        mem.add_ai_message("A2")

        assert mem.turn_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# SessionWriter
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionWriter:
    """Tests for SessionWriter."""

    def test_save_creates_output_file(self, tmp_path, monkeypatch):
        from cv_agent.config import settings
        monkeypatch.setattr(settings, "output_dir", tmp_path)

        from cv_agent.output.writer import SessionWriter

        writer = SessionWriter()
        writer.record(question="What is your name?", answer="Based on the CV, John Doe.")
        writer.record(question="Where did you study?", answer="Based on the CV, UCT.")

        out_file = writer.save()
        assert out_file.exists()

        content = out_file.read_text(encoding="utf-8")
        assert "What is your name?" in content
        assert "John Doe" in content
        assert "Where did you study?" in content
        assert "UCT" in content

    def test_save_includes_timestamps(self, tmp_path, monkeypatch):
        from cv_agent.config import settings
        monkeypatch.setattr(settings, "output_dir", tmp_path)

        from cv_agent.output.writer import SessionWriter

        writer = SessionWriter()
        writer.record("Q?", "A.")
        out = writer.save()

        content = out.read_text(encoding="utf-8")
        assert "Session started" in content
        assert "Session ended" in content

    def test_total_questions_count(self, tmp_path, monkeypatch):
        from cv_agent.config import settings
        monkeypatch.setattr(settings, "output_dir", tmp_path)

        from cv_agent.output.writer import SessionWriter

        writer = SessionWriter()
        for i in range(3):
            writer.record(f"Q{i}", f"A{i}")
        out = writer.save()

        content = out.read_text(encoding="utf-8")
        assert "Total questions : 3" in content


# ─────────────────────────────────────────────────────────────────────────────
# Grounding smoke tests (no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundingBehaviour:
    """Verify the system prompt contains required grounding instructions."""

    def test_system_prompt_contains_grounding_phrase(self):
        from cv_agent.agent.nodes import SYSTEM_PROMPT

        assert "not available in the provided CV" in SYSTEM_PROMPT
        assert "search_cv" in SYSTEM_PROMPT
        assert "NEVER" in SYSTEM_PROMPT or "ONLY" in SYSTEM_PROMPT

    def test_not_available_message_constant(self):
        from cv_agent.agent.nodes import NOT_AVAILABLE_MESSAGE

        assert "not available in the provided CV" in NOT_AVAILABLE_MESSAGE


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Unavailable information (score gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnavailableInformation:
    """
    When retrieval returns a score above the threshold (irrelevant chunks),
    search_cv must return status=unavailable, and the fallback_node must
    return NOT_AVAILABLE_MESSAGE.
    """

    def test_high_score_triggers_unavailable_status(self, monkeypatch):
        """
        Simulate a query for which every retrieved chunk has an L2 distance
        far above the threshold (e.g. 'What is your passport number?').
        search_cv must return status=unavailable.
        """
        import cv_agent.tools.cv_search as cv_search_module
        from cv_agent.config import settings

        mock_store = MagicMock()
        irrelevant_doc = MagicMock()
        irrelevant_doc.page_content = "Some irrelevant text"
        irrelevant_doc.metadata = {
            "source": "cv.pdf", "page": 0, "chunk_id": 0, "section": "General"
        }

        high_score = settings.retrieval_score_threshold + 2.0
        mock_store.similarity_search_with_score.return_value = [
            (irrelevant_doc, high_score)
        ]

        monkeypatch.setattr(cv_search_module, "_vector_store", mock_store)

        result_json = cv_search_module.search_cv.invoke(
            {"query": "What is your passport number?"}
        )
        result = json.loads(result_json)

        assert result["status"] == "unavailable", (
            f"Expected unavailable for score {high_score:.2f}, got {result['status']!r}"
        )
        assert result["documents"] == []
        assert result["best_score"] > settings.retrieval_score_threshold

    def test_fallback_node_returns_not_available_message(self):
        """fallback_node must always set final_answer = NOT_AVAILABLE_MESSAGE."""
        from cv_agent.agent.nodes import fallback_node, NOT_AVAILABLE_MESSAGE

        state = {
            "messages": [HumanMessage(content="What is your passport number?")],
            "retrieval_score": 3.5,
            "grounding_status": "unavailable",
        }
        result = fallback_node(state)

        assert result["final_answer"] == NOT_AVAILABLE_MESSAGE
        assert result["grounding_status"] == "unavailable"
        assert any(
            isinstance(m, AIMessage) and m.content == NOT_AVAILABLE_MESSAGE
            for m in result["messages"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Follow-up question contextualization
# ─────────────────────────────────────────────────────────────────────────────

class TestFollowUpContextualization:
    """
    When a follow-up question references a prior answer (e.g. "there", "that role"),
    contextualize_query_node must produce a rewritten, standalone query.
    Self-contained first questions must pass through unchanged.
    """

    def _make_state(self, messages):
        return {
            "messages": messages,
            "iteration": 0,
            "contextualized_query": "",
        }

    def test_first_question_not_rewritten(self):
        """First question (no prior history) must pass through without LLM call."""
        from cv_agent.agent.nodes import contextualize_query_node

        state = self._make_state([
            SystemMessage(content="You are an assistant."),
            HumanMessage(content="What programming languages are listed on the CV?"),
        ])

        result = contextualize_query_node(state)
        assert result["contextualized_query"] == "What programming languages are listed on the CV?"

    def test_follow_up_is_rewritten(self):
        """
        A follow-up with a pronoun ("there") should be rewritten to a
        standalone query that includes the prior context.
        """
        from cv_agent.agent.nodes import contextualize_query_node

        rewritten = "What technologies did the candidate use in their role at XYZ Corp?"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=rewritten)

        with patch("cv_agent.agent.nodes._get_llm", return_value=mock_llm):
            state = self._make_state([
                SystemMessage(content="You are an assistant."),
                HumanMessage(content="What was your most recent role?"),
                AIMessage(content="Based on the CV, Senior Engineer at XYZ Corp."),
                HumanMessage(content="What technologies did you use there?"),
            ])
            result = contextualize_query_node(state)

        assert result["contextualized_query"] == rewritten
        assert "there" not in result["contextualized_query"].lower()

    def test_self_contained_follow_up_not_rewritten(self):
        """
        A follow-up that's self-contained (LLM returns NO_REWRITE) should
        keep the original question unchanged.
        """
        from cv_agent.agent.nodes import contextualize_query_node

        original_question = "What programming languages are listed on the CV?"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NO_REWRITE")

        with patch("cv_agent.agent.nodes._get_llm", return_value=mock_llm):
            state = self._make_state([
                SystemMessage(content="You are an assistant."),
                HumanMessage(content="What was your most recent role?"),
                AIMessage(content="Senior Engineer."),
                HumanMessage(content=original_question),
            ])
            result = contextualize_query_node(state)

        assert result["contextualized_query"] == original_question

    def test_tool_node_uses_contextualized_query(self):
        """
        When contextualized_query is in state, tool_node must pass it to
        search_cv instead of the LLM-generated query argument.
        """
        captured_query: list[str] = []

        def mock_invoke(args):
            captured_query.append(args.get("query", ""))
            return json.dumps({
                "status": "relevant",
                "best_score": 0.3,
                "documents": ["Candidate worked at XYZ using Python."],
                "formatted": "[Excerpt 1]\nCandidate worked at XYZ using Python.",
            })

        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = mock_invoke

        tool_call = {
            "name": "search_cv",
            "args": {"query": "What technologies did you use there?"},
            "id": "call_001",
        }
        last_ai_msg = AIMessage(content="")
        last_ai_msg.tool_calls = [tool_call]

        state = {
            "messages": [last_ai_msg],
            "contextualized_query": "What technologies did the candidate use at XYZ Corp?",
            "retrieved_documents": [],
            "iteration": 1,
        }

        # tool_node imports search_cv lazily; patch at source
        with patch("cv_agent.tools.cv_search.search_cv", mock_tool):
            from cv_agent.agent.nodes import tool_node
            tool_node(state)

        assert len(captured_query) == 1, f"Expected 1 tool call, got {len(captured_query)}"
        assert captured_query[0] == "What technologies did the candidate use at XYZ Corp?", (
            f"Expected contextualized query, got: {captured_query[0]!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Grounding validation
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundingValidation:
    """
    The check_grounding_node must replace a hallucinated answer with
    NOT_AVAILABLE_MESSAGE when the grounding LLM returns grounded=false.
    """

    def _base_state(self, answer: str, documents: list[str]) -> dict:
        return {
            "messages": [
                SystemMessage(content="You are a CV assistant."),
                HumanMessage(content="What is the candidate's GPA?"),
                AIMessage(content=answer),
            ],
            "retrieved_documents": documents,
            "grounding_status": None,
        }

    def test_grounded_answer_passes_through(self):
        """When the validator says grounded=true, the answer is unchanged."""
        from cv_agent.agent.nodes import check_grounding_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"grounded": True, "reason": "Supported."})
        )

        with patch("cv_agent.agent.nodes._get_llm", return_value=mock_llm):
            state = self._base_state(
                answer="Based on the CV, the candidate graduated with a 3.8 GPA.",
                documents=["Graduated with honours, GPA 3.8, University of Cape Town."],
            )
            result = check_grounding_node(state)

        assert result["grounding_status"] == "grounded"
        assert "3.8" in result["final_answer"]

    def test_ungrounded_answer_replaced_with_fallback(self):
        """When the validator says grounded=false, answer becomes NOT_AVAILABLE_MESSAGE."""
        from cv_agent.agent.nodes import check_grounding_node, NOT_AVAILABLE_MESSAGE

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "grounded": False,
                "reason": "GPA 4.0 is not in the retrieved context.",
            })
        )

        with patch("cv_agent.agent.nodes._get_llm", return_value=mock_llm):
            state = self._base_state(
                answer="Based on the CV, the candidate has a 4.0 GPA.",
                documents=["The candidate studied Computer Science at UCT."],
            )
            result = check_grounding_node(state)

        assert result["grounding_status"] == "not_grounded"
        assert result["final_answer"] == NOT_AVAILABLE_MESSAGE
        assert result["messages"][-1].content == NOT_AVAILABLE_MESSAGE

    def test_no_retrieved_docs_triggers_fallback(self):
        """
        If retrieved_documents is empty (LLM answered without retrieval),
        check_grounding_node must return the fallback for safety.
        """
        from cv_agent.agent.nodes import check_grounding_node, NOT_AVAILABLE_MESSAGE

        state = {
            "messages": [
                HumanMessage(content="What is the candidate's father's name?"),
                AIMessage(content="Based on the CV, his father is called John."),
            ],
            "retrieved_documents": [],
            "grounding_status": None,
        }

        result = check_grounding_node(state)

        assert result["grounding_status"] == "not_grounded"
        assert result["final_answer"] == NOT_AVAILABLE_MESSAGE

    def test_already_unavailable_state_skipped(self):
        """
        If grounding_status is already "unavailable" (set by fallback routing),
        check_grounding_node should return an empty dict (no-op).
        """
        from cv_agent.agent.nodes import check_grounding_node

        state = {
            "messages": [],
            "retrieved_documents": [],
            "grounding_status": "unavailable",
        }

        result = check_grounding_node(state)
        assert result == {}
