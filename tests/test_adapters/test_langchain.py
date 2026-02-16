"""Tests for the LangChain adapter (mocked -- no langchain-core required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.types import DeepRecallResult, Source, UsageInfo

# ---------------------------------------------------------------------------
# Helpers -- build mock langchain-core modules so the import succeeds
# ---------------------------------------------------------------------------


def _build_lc_mocks() -> dict[str, types.ModuleType]:
    """Create fake langchain_core modules with the classes the adapter needs."""
    lc = types.ModuleType("langchain_core")
    lc_docs = types.ModuleType("langchain_core.documents")
    lc_cb_llm = types.ModuleType("langchain_core.callbacks")
    lc_lm = types.ModuleType("langchain_core.language_models.chat_models")
    lc_msg = types.ModuleType("langchain_core.messages")
    lc_out = types.ModuleType("langchain_core.outputs")
    lc_ret = types.ModuleType("langchain_core.retrievers")

    # Minimal stub classes
    class Document:
        def __init__(self, *, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class BaseMessage:
        def __init__(self, content: str = "", type: str = "human"):
            self.content = content
            self.type = type

    class AIMessage(BaseMessage):
        def __init__(self, content: str = "", additional_kwargs: dict | None = None, **kw):
            super().__init__(content=content, type="ai")
            self.additional_kwargs = additional_kwargs or {}

    class ChatGeneration:
        def __init__(self, message=None):
            self.message = message

    class ChatResult:
        def __init__(self, generations=None):
            self.generations = generations or []

    class BaseRetriever:
        class Config:
            arbitrary_types_allowed = True

        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class BaseChatModel:
        class Config:
            arbitrary_types_allowed = True

        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    # Wire up the modules
    lc_docs.Document = Document
    lc_cb_llm.CallbackManagerForLLMRun = MagicMock
    lc_cb_llm.CallbackManagerForRetrieverRun = MagicMock
    lc_lm.BaseChatModel = BaseChatModel
    lc_msg.AIMessage = AIMessage
    lc_msg.BaseMessage = BaseMessage
    lc_out.ChatGeneration = ChatGeneration
    lc_out.ChatResult = ChatResult
    lc_ret.BaseRetriever = BaseRetriever

    return {
        "langchain_core": lc,
        "langchain_core.documents": lc_docs,
        "langchain_core.callbacks": lc_cb_llm,
        "langchain_core.language_models": types.ModuleType("langchain_core.language_models"),
        "langchain_core.language_models.chat_models": lc_lm,
        "langchain_core.messages": lc_msg,
        "langchain_core.outputs": lc_out,
        "langchain_core.retrievers": lc_ret,
    }


def _fake_result(answer: str = "test answer") -> DeepRecallResult:
    return DeepRecallResult(
        answer=answer,
        sources=[
            Source(content="doc1 text", metadata={"k": "v"}, score=0.9, id="s1"),
            Source(content="doc2 text", metadata={}, score=0.7, id="s2"),
        ],
        reasoning_trace=[],
        query="test query",
        execution_time=1.0,
        usage=UsageInfo(total_input_tokens=10, total_output_tokens=5, total_calls=2),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeepRecallRetriever:
    """Tests for the LangChain retriever adapter."""

    def _make_retriever(self, mock_engine: MagicMock):
        mods = _build_lc_mocks()
        with patch.dict(sys.modules, mods):
            # Force reimport so the adapter picks up our stubs
            if "deeprecall.adapters.langchain" in sys.modules:
                del sys.modules["deeprecall.adapters.langchain"]
            from deeprecall.adapters.langchain import DeepRecallRetriever

            return DeepRecallRetriever(engine=mock_engine, top_k=5)

    def test_get_relevant_documents_returns_answer_and_sources(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result("Python was created in 1991.")
        retriever = self._make_retriever(engine)

        docs = retriever._get_relevant_documents("Who created Python?")

        # First doc is the answer
        assert docs[0].page_content == "Python was created in 1991."
        assert docs[0].metadata["source"] == "deeprecall_answer"
        # Remaining docs are sources
        assert len(docs) == 3  # answer + 2 sources
        assert docs[1].page_content == "doc1 text"
        assert docs[2].page_content == "doc2 text"

    def test_engine_called_with_correct_top_k(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        retriever = self._make_retriever(engine)
        retriever.top_k = 7

        retriever._get_relevant_documents("q")
        engine.query.assert_called_once_with("q", top_k=7)

    def test_source_metadata_includes_score_and_id(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        retriever = self._make_retriever(engine)

        docs = retriever._get_relevant_documents("q")
        src_doc = docs[1]
        assert src_doc.metadata["score"] == 0.9
        assert src_doc.metadata["source_id"] == "s1"


class TestDeepRecallChatModel:
    """Tests for the LangChain chat model adapter."""

    def _make_chat_model(self, mock_engine: MagicMock):
        mods = _build_lc_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.adapters.langchain" in sys.modules:
                del sys.modules["deeprecall.adapters.langchain"]
            from deeprecall.adapters.langchain import DeepRecallChatModel

            return DeepRecallChatModel(engine=mock_engine)

    def test_generate_extracts_human_message(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result("The answer is 42.")
        model = self._make_chat_model(engine)

        mods = _build_lc_mocks()
        BaseMessage = mods["langchain_core.messages"].BaseMessage
        messages = [BaseMessage(content="What is 42?", type="human")]

        result = model._generate(messages)
        engine.query.assert_called_once_with("What is 42?")
        assert result.generations[0].message.content == "The answer is 42."

    def test_generate_picks_last_human_message(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        model = self._make_chat_model(engine)

        mods = _build_lc_mocks()
        Msg = mods["langchain_core.messages"].BaseMessage
        messages = [
            Msg(content="First", type="human"),
            Msg(content="System msg", type="system"),
            Msg(content="Second", type="human"),
        ]

        model._generate(messages)
        engine.query.assert_called_once_with("Second")

    def test_generate_fallback_to_any_content(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        model = self._make_chat_model(engine)

        mods = _build_lc_mocks()
        Msg = mods["langchain_core.messages"].BaseMessage
        messages = [Msg(content="system stuff", type="system")]

        model._generate(messages)
        engine.query.assert_called_once_with("system stuff")

    def test_generate_raises_on_empty_messages(self):
        engine = MagicMock()
        model = self._make_chat_model(engine)

        with pytest.raises(ValueError, match="No message content"):
            model._generate([])

    def test_llm_type_property(self):
        engine = MagicMock()
        model = self._make_chat_model(engine)
        assert model._llm_type == "deeprecall"

    def test_additional_kwargs_include_sources_and_usage(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        model = self._make_chat_model(engine)

        mods = _build_lc_mocks()
        Msg = mods["langchain_core.messages"].BaseMessage
        result = model._generate([Msg(content="q", type="human")])

        kwargs = result.generations[0].message.additional_kwargs
        assert "sources" in kwargs
        assert "usage" in kwargs
        assert "execution_time" in kwargs
        assert len(kwargs["sources"]) == 2
