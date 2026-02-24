"""Tests for DeepRecall v0.4.0 -- RLM v0.1.1a upgrade.

Covers: new config params, cost tracking, tracer RLMLogger protocol,
iteration callbacks, engine wiring of new RLM params, and prompt updates.
"""

from __future__ import annotations

import json
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.callbacks import (
    BaseCallback,
    CallbackManager,
    JSONLCallback,
    ProgressCallback,
)
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.guardrails import BudgetStatus, QueryBudget
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, UsageInfo
from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_iteration(
    code: str = "",
    stdout: str = "",
    stderr: str = "",
    final_answer: str | None = None,
):
    block = MagicMock()
    block.code = code
    block.result = MagicMock()
    block.result.stdout = stdout
    block.result.stderr = stderr
    block.result.rlm_calls = []

    iteration = MagicMock()
    iteration.code_blocks = [block]
    iteration.final_answer = final_answer
    iteration.iteration_time = 0.5
    return iteration


# ===================================================================
# 1. Config -- new params validation
# ===================================================================


class TestConfigNewParams:
    def test_defaults(self):
        config = DeepRecallConfig()
        assert config.max_timeout is None
        assert config.max_errors is None
        assert config.max_tokens is None
        assert config.compaction is False
        assert config.compaction_threshold_pct == 0.85

    def test_custom_values(self):
        config = DeepRecallConfig(
            max_timeout=60.0,
            max_errors=5,
            max_tokens=8000,
            compaction=True,
            compaction_threshold_pct=0.7,
        )
        assert config.max_timeout == 60.0
        assert config.max_errors == 5
        assert config.max_tokens == 8000
        assert config.compaction is True
        assert config.compaction_threshold_pct == 0.7

    def test_max_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            DeepRecallConfig(max_timeout=0)

    def test_max_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            DeepRecallConfig(max_timeout=-5.0)

    def test_max_errors_zero_raises(self):
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            DeepRecallConfig(max_errors=0)

    def test_max_errors_negative_raises(self):
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            DeepRecallConfig(max_errors=-1)

    def test_compaction_threshold_pct_zero_raises(self):
        with pytest.raises(ValueError, match="compaction_threshold_pct must be in"):
            DeepRecallConfig(compaction_threshold_pct=0.0)

    def test_compaction_threshold_pct_negative_raises(self):
        with pytest.raises(ValueError, match="compaction_threshold_pct must be in"):
            DeepRecallConfig(compaction_threshold_pct=-0.5)

    def test_compaction_threshold_pct_above_one_raises(self):
        with pytest.raises(ValueError, match="compaction_threshold_pct must be in"):
            DeepRecallConfig(compaction_threshold_pct=1.5)

    def test_compaction_threshold_pct_one_allowed(self):
        config = DeepRecallConfig(compaction_threshold_pct=1.0)
        assert config.compaction_threshold_pct == 1.0

    def test_to_dict_includes_new_fields(self):
        config = DeepRecallConfig(
            max_timeout=30.0,
            max_errors=3,
            max_tokens=10000,
            compaction=True,
            compaction_threshold_pct=0.9,
        )
        d = config.to_dict()
        assert d["max_timeout"] == 30.0
        assert d["max_errors"] == 3
        assert d["max_tokens"] == 10000
        assert d["compaction"] is True
        assert d["compaction_threshold_pct"] == 0.9

    def test_to_dict_defaults(self):
        d = DeepRecallConfig().to_dict()
        assert d["max_timeout"] is None
        assert d["max_errors"] is None
        assert d["max_tokens"] is None
        assert d["compaction"] is False
        assert d["compaction_threshold_pct"] == 0.85


# ===================================================================
# 2. UsageInfo -- cost tracking
# ===================================================================


class TestUsageInfoCost:
    def test_default_cost_is_none(self):
        usage = UsageInfo()
        assert usage.total_cost_usd is None

    def test_cost_set(self):
        usage = UsageInfo(total_cost_usd=0.0045)
        assert usage.total_cost_usd == 0.0045

    def test_to_dict_includes_cost_when_set(self):
        usage = UsageInfo(total_cost_usd=0.01)
        d = usage.to_dict()
        assert d["total_cost_usd"] == 0.01

    def test_to_dict_omits_cost_when_none(self):
        usage = UsageInfo()
        d = usage.to_dict()
        assert "total_cost_usd" not in d

    def test_model_breakdown_with_cost(self):
        usage = UsageInfo(
            total_input_tokens=100,
            total_output_tokens=50,
            total_calls=3,
            total_cost_usd=0.005,
            model_breakdown={
                "gpt-4o-mini": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "calls": 3,
                    "cost_usd": 0.005,
                }
            },
        )
        d = usage.to_dict()
        assert d["model_breakdown"]["gpt-4o-mini"]["cost_usd"] == 0.005
        assert d["total_cost_usd"] == 0.005


class TestDeepRecallResultCostRoundTrip:
    def test_from_dict_with_cost(self):
        data = {
            "answer": "test",
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_calls": 2,
                "total_cost_usd": 0.003,
                "model_breakdown": {},
            },
        }
        result = DeepRecallResult.from_dict(data)
        assert result.usage.total_cost_usd == 0.003

    def test_from_dict_without_cost(self):
        data = {
            "answer": "test",
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_calls": 2,
                "model_breakdown": {},
            },
        }
        result = DeepRecallResult.from_dict(data)
        assert result.usage.total_cost_usd is None

    def test_round_trip_with_cost(self):
        original = DeepRecallResult(
            answer="answer",
            usage=UsageInfo(
                total_input_tokens=500,
                total_output_tokens=200,
                total_calls=5,
                total_cost_usd=0.0123,
            ),
        )
        restored = DeepRecallResult.from_dict(original.to_dict())
        assert restored.usage.total_cost_usd == 0.0123
        assert restored.usage.total_input_tokens == 500


# ===================================================================
# 3. Tracer -- RLMLogger protocol (clear_iterations, get_trajectory)
# ===================================================================


class TestTracerRLMLoggerProtocol:
    def test_clear_iterations_resets_steps(self):
        tracer = DeepRecallTracer()
        tracer.log(_make_mock_iteration())
        tracer.log(_make_mock_iteration())
        assert len(tracer.steps) == 2
        assert tracer.budget_status.iterations_used == 2

        tracer.clear_iterations()

        assert len(tracer.steps) == 0
        assert tracer.budget_status.iterations_used == 0
        assert tracer.budget_status.search_calls_used == 0

    def test_clear_iterations_preserves_budget_object(self):
        budget = QueryBudget(max_iterations=10)
        tracer = DeepRecallTracer(budget=budget)
        tracer.log(_make_mock_iteration())
        tracer.clear_iterations()

        assert tracer.budget_status.budget is budget
        assert tracer.budget_status.budget.max_iterations == 10

    def test_get_trajectory_returns_dict(self):
        tracer = DeepRecallTracer()
        tracer.log(_make_mock_iteration(code="x = 1", stdout="1"))
        tracer.log(_make_mock_iteration(code="y = 2", stdout="2"))

        trajectory = tracer.get_trajectory()

        assert isinstance(trajectory, dict)
        assert "iterations" in trajectory
        assert "budget_status" in trajectory
        assert len(trajectory["iterations"]) == 2
        assert trajectory["iterations"][0]["iteration"] == 1
        assert trajectory["iterations"][1]["iteration"] == 2

    def test_get_trajectory_budget_status(self):
        budget = QueryBudget(max_iterations=10)
        tracer = DeepRecallTracer(budget=budget)
        tracer.log(_make_mock_iteration(code='search_db("test")'))

        trajectory = tracer.get_trajectory()
        bs = trajectory["budget_status"]
        assert bs["iterations_used"] == 1
        assert bs["search_calls_used"] == 1
        assert bs["budget"]["max_iterations"] == 10

    def test_get_trajectory_empty(self):
        tracer = DeepRecallTracer()
        trajectory = tracer.get_trajectory()
        assert trajectory["iterations"] == []
        assert trajectory["budget_status"]["iterations_used"] == 0

    def test_clear_then_log_restarts_counting(self):
        tracer = DeepRecallTracer()
        tracer.log(_make_mock_iteration())
        tracer.log(_make_mock_iteration())
        tracer.clear_iterations()
        tracer.log(_make_mock_iteration())

        assert len(tracer.steps) == 1
        assert tracer.steps[0].iteration == 1
        assert tracer.budget_status.iterations_used == 1


# ===================================================================
# 4. Tracer -- iteration callbacks
# ===================================================================


class TestTracerIterationCallbacks:
    def test_on_iteration_start_fires(self):
        cb_manager = MagicMock()
        tracer = DeepRecallTracer(callback_manager=cb_manager)
        tracer.log(_make_mock_iteration())

        cb_manager.on_iteration_start.assert_called_once_with(1)

    def test_on_iteration_complete_fires_no_final(self):
        cb_manager = MagicMock()
        tracer = DeepRecallTracer(callback_manager=cb_manager)
        tracer.log(_make_mock_iteration())

        cb_manager.on_iteration_complete.assert_called_once_with(1, False)

    def test_on_iteration_complete_fires_with_final(self):
        cb_manager = MagicMock()
        tracer = DeepRecallTracer(callback_manager=cb_manager)
        tracer.log(_make_mock_iteration(final_answer="The answer is 42"))

        cb_manager.on_iteration_complete.assert_called_once_with(1, True)

    def test_multiple_iterations_callback_ordering(self):
        call_log: list[str] = []
        cb_manager = MagicMock()
        cb_manager.on_iteration_start.side_effect = lambda n: call_log.append(f"start:{n}")
        cb_manager.on_reasoning_step.side_effect = lambda s, b: call_log.append(
            f"step:{s.iteration}"
        )
        cb_manager.on_iteration_complete.side_effect = lambda n, f: call_log.append(
            f"complete:{n}:{f}"
        )

        tracer = DeepRecallTracer(callback_manager=cb_manager)
        tracer.log(_make_mock_iteration())
        tracer.log(_make_mock_iteration(final_answer="done"))

        assert call_log == [
            "start:1",
            "step:1",
            "complete:1:False",
            "start:2",
            "step:2",
            "complete:2:True",
        ]

    def test_no_callback_manager_no_error(self):
        tracer = DeepRecallTracer(callback_manager=None)
        tracer.log(_make_mock_iteration())
        assert len(tracer.steps) == 1


# ===================================================================
# 5. Callback system -- new hooks
# ===================================================================


class TestCallbackIterationHooks:
    def test_base_callback_iteration_start_noop(self):
        cb = BaseCallback()
        cb.on_iteration_start(1)  # Should not raise

    def test_base_callback_iteration_complete_noop(self):
        cb = BaseCallback()
        cb.on_iteration_complete(1, False)  # Should not raise

    def test_callback_manager_dispatches_iteration_start(self):
        events: list[str] = []

        class Recorder(BaseCallback):
            def on_iteration_start(self, iteration):
                events.append(f"start:{iteration}")

        manager = CallbackManager([Recorder()])
        manager.on_iteration_start(3)
        assert events == ["start:3"]

    def test_callback_manager_dispatches_iteration_complete(self):
        events: list[str] = []

        class Recorder(BaseCallback):
            def on_iteration_complete(self, iteration, has_final_answer):
                events.append(f"complete:{iteration}:{has_final_answer}")

        manager = CallbackManager([Recorder()])
        manager.on_iteration_complete(2, True)
        assert events == ["complete:2:True"]

    def test_error_in_iteration_callback_does_not_propagate(self):
        class Failing(BaseCallback):
            def on_iteration_start(self, iteration):
                raise RuntimeError("kaboom")

        events: list[int] = []

        class Good(BaseCallback):
            def on_iteration_start(self, iteration):
                events.append(iteration)

        manager = CallbackManager([Failing(), Good()])
        manager.on_iteration_start(5)
        assert events == [5]


class TestProgressCallbackIterationEvents:
    def test_records_iteration_start(self):
        cb = ProgressCallback()
        cb.on_iteration_start(1)
        assert len(cb.events) == 1
        assert cb.events[0]["type"] == "iteration_start"
        assert cb.events[0]["iteration"] == 1

    def test_records_iteration_complete(self):
        cb = ProgressCallback()
        cb.on_iteration_complete(3, True)
        assert len(cb.events) == 1
        assert cb.events[0]["type"] == "iteration_complete"
        assert cb.events[0]["iteration"] == 3
        assert cb.events[0]["has_final_answer"] is True


class TestJSONLCallbackIterationEvents:
    def test_logs_iteration_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = JSONLCallback(log_dir=tmpdir)
            cb.on_iteration_start(1)
            cb.on_iteration_complete(1, False)
            cb.on_iteration_start(2)
            cb.on_iteration_complete(2, True)

            with open(cb.log_path) as f:
                lines = f.readlines()

            assert len(lines) == 4
            e1 = json.loads(lines[0])
            assert e1["type"] == "iteration_start"
            assert e1["iteration"] == 1

            e2 = json.loads(lines[1])
            assert e2["type"] == "iteration_complete"
            assert e2["iteration"] == 1
            assert e2["has_final_answer"] is False

            e4 = json.loads(lines[3])
            assert e4["has_final_answer"] is True


# ===================================================================
# 6. Engine -- passes new RLM params
# ===================================================================


class TestEngineRLMParamWiring:
    """Verify the engine passes the right kwargs to the RLM() constructor."""

    @patch("deeprecall.core.engine.SearchServer")
    def test_passes_compaction_params(self, MockServer, mock_vectorstore, mock_rlm_result):
        import sys

        mock_server_inst = MagicMock()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = []
        MockServer.return_value = mock_server_inst

        mock_rlm_module = MagicMock()
        mock_rlm_cls = MagicMock()
        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        from deeprecall.core.engine import DeepRecallEngine

        config = DeepRecallConfig(
            compaction=True,
            compaction_threshold_pct=0.75,
            reuse_search_server=False,
        )
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            engine.query("test query")

        call_kwargs = mock_rlm_cls.call_args
        assert call_kwargs.kwargs.get("compaction") is True
        assert call_kwargs.kwargs.get("compaction_threshold_pct") == 0.75

    @patch("deeprecall.core.engine.SearchServer")
    def test_passes_max_budget_from_cost_budget(
        self, MockServer, mock_vectorstore, mock_rlm_result
    ):
        import sys

        mock_server_inst = MagicMock()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = []
        MockServer.return_value = mock_server_inst

        mock_rlm_module = MagicMock()
        mock_rlm_cls = MagicMock()
        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        from deeprecall.core.engine import DeepRecallEngine

        budget = QueryBudget(max_cost_usd=0.50)
        config = DeepRecallConfig(budget=budget, reuse_search_server=False)
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            engine.query("test query")

        call_kwargs = mock_rlm_cls.call_args
        assert call_kwargs.kwargs.get("max_budget") == 0.50

    @patch("deeprecall.core.engine.SearchServer")
    def test_passes_max_timeout_and_errors(self, MockServer, mock_vectorstore, mock_rlm_result):
        import sys

        mock_server_inst = MagicMock()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = []
        MockServer.return_value = mock_server_inst

        mock_rlm_module = MagicMock()
        mock_rlm_cls = MagicMock()
        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        from deeprecall.core.engine import DeepRecallEngine

        config = DeepRecallConfig(
            max_timeout=120.0,
            max_errors=5,
            reuse_search_server=False,
        )
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            engine.query("test query")

        call_kwargs = mock_rlm_cls.call_args
        assert call_kwargs.kwargs.get("max_timeout") == 120.0
        assert call_kwargs.kwargs.get("max_errors") == 5

    @patch("deeprecall.core.engine.SearchServer")
    def test_omits_optional_kwargs_when_not_set(
        self, MockServer, mock_vectorstore, mock_rlm_result
    ):
        import sys

        mock_server_inst = MagicMock()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = []
        MockServer.return_value = mock_server_inst

        mock_rlm_module = MagicMock()
        mock_rlm_cls = MagicMock()
        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        from deeprecall.core.engine import DeepRecallEngine

        config = DeepRecallConfig(reuse_search_server=False)
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            engine.query("test query")

        call_kwargs = mock_rlm_cls.call_args
        assert "max_budget" not in call_kwargs.kwargs
        assert "max_timeout" not in call_kwargs.kwargs
        assert "max_errors" not in call_kwargs.kwargs
        assert "compaction" not in call_kwargs.kwargs


# ===================================================================
# 7. Engine -- cost extraction
# ===================================================================


class TestEngineCostExtraction:
    def test_extract_usage_with_cost(self, mock_vectorstore):
        from deeprecall.core.engine import DeepRecallEngine

        engine = DeepRecallEngine(vectorstore=mock_vectorstore)

        mock_result = MagicMock()
        mock_result.usage_summary = MagicMock()
        mock_result.usage_summary.model_usage_summaries = {
            "gpt-4o-mini": MagicMock(
                total_input_tokens=100,
                total_output_tokens=50,
                total_calls=3,
                total_cost=0.0023,
            ),
            "claude-3-sonnet": MagicMock(
                total_input_tokens=200,
                total_output_tokens=100,
                total_calls=2,
                total_cost=0.0050,
            ),
        }

        usage = engine._extract_usage(mock_result)

        assert usage.total_input_tokens == 300
        assert usage.total_output_tokens == 150
        assert usage.total_calls == 5
        assert usage.total_cost_usd == pytest.approx(0.0073)
        assert usage.model_breakdown["gpt-4o-mini"]["cost_usd"] == 0.0023
        assert usage.model_breakdown["claude-3-sonnet"]["cost_usd"] == 0.0050

    def test_extract_usage_without_cost(self, mock_vectorstore):
        from deeprecall.core.engine import DeepRecallEngine

        engine = DeepRecallEngine(vectorstore=mock_vectorstore)

        mock_result = MagicMock()
        mock_result.usage_summary = MagicMock()
        mock_result.usage_summary.model_usage_summaries = {
            "gpt-4o-mini": MagicMock(
                total_input_tokens=100,
                total_output_tokens=50,
                total_calls=3,
                spec=["total_input_tokens", "total_output_tokens", "total_calls"],
            ),
        }
        # Remove total_cost attribute to simulate old RLM
        del mock_result.usage_summary.model_usage_summaries["gpt-4o-mini"].total_cost

        usage = engine._extract_usage(mock_result)

        assert usage.total_input_tokens == 100
        assert usage.total_cost_usd is None
        assert "cost_usd" not in usage.model_breakdown["gpt-4o-mini"]

    def test_extract_usage_partial_cost(self, mock_vectorstore):
        """One model has cost, another doesn't."""
        from deeprecall.core.engine import DeepRecallEngine

        engine = DeepRecallEngine(vectorstore=mock_vectorstore)

        model_with_cost = MagicMock(
            total_input_tokens=100,
            total_output_tokens=50,
            total_calls=2,
            total_cost=0.003,
        )
        model_without_cost = MagicMock(
            total_input_tokens=80,
            total_output_tokens=40,
            total_calls=1,
        )
        del model_without_cost.total_cost

        mock_result = MagicMock()
        mock_result.usage_summary = MagicMock()
        mock_result.usage_summary.model_usage_summaries = {
            "openrouter/gpt4": model_with_cost,
            "local/llama": model_without_cost,
        }

        usage = engine._extract_usage(mock_result)

        assert usage.total_cost_usd == 0.003
        assert usage.model_breakdown["openrouter/gpt4"]["cost_usd"] == 0.003
        assert "cost_usd" not in usage.model_breakdown["local/llama"]


class TestEngineBudgetCostWiring:
    @patch("deeprecall.core.engine.SearchServer")
    def test_cost_wired_to_budget_status(self, MockServer, mock_vectorstore):
        import sys

        mock_server_inst = MagicMock()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = []
        MockServer.return_value = mock_server_inst

        mock_rlm_result = MagicMock()
        mock_rlm_result.response = "answer"
        mock_rlm_result.usage_summary = MagicMock()
        mock_rlm_result.usage_summary.model_usage_summaries = {
            "gpt-4o-mini": MagicMock(
                total_input_tokens=100,
                total_output_tokens=50,
                total_calls=2,
                total_cost=0.005,
            )
        }

        mock_rlm_module = MagicMock()
        mock_rlm_cls = MagicMock()
        mock_rlm_instance = MagicMock()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        from deeprecall.core.engine import DeepRecallEngine

        config = DeepRecallConfig(reuse_search_server=False)
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            result = engine.query("test")

        assert result.budget_status["cost_usd"] == pytest.approx(0.005)
        assert result.usage.total_cost_usd == pytest.approx(0.005)


# ===================================================================
# 8. System prompt -- updates
# ===================================================================


class TestSystemPromptUpdates:
    def test_final_var_direct_value_documented(self):
        assert "FINAL_VAR also accepts a direct value" in DEEPRECALL_SYSTEM_PROMPT

    def test_history_variable_documented(self):
        assert "history" in DEEPRECALL_SYSTEM_PROMPT
        assert "compaction" in DEEPRECALL_SYSTEM_PROMPT.lower()

    def test_final_var_still_mentions_variable_name(self):
        assert "FINAL_VAR(variable_name)" in DEEPRECALL_SYSTEM_PROMPT

    def test_search_db_still_present(self):
        assert "search_db" in DEEPRECALL_SYSTEM_PROMPT

    def test_llm_query_still_present(self):
        assert "llm_query" in DEEPRECALL_SYSTEM_PROMPT

    def test_show_vars_still_present(self):
        assert "SHOW_VARS" in DEEPRECALL_SYSTEM_PROMPT


# ===================================================================
# 9. Test fixture -- verify updated mock
# ===================================================================


class TestUpdatedFixture:
    def test_mock_rlm_result_has_cost(self, mock_rlm_result):
        model_usage = mock_rlm_result.usage_summary.model_usage_summaries["gpt-4o-mini"]
        assert model_usage.total_cost == 0.0023

    def test_mock_rlm_result_has_metadata(self, mock_rlm_result):
        assert mock_rlm_result.metadata is None


# ===================================================================
# 10. Guardrails -- cost budget check (existing, now with real data)
# ===================================================================


class TestCostBudgetEnforcement:
    def test_cost_budget_check_raises_when_exceeded(self):
        budget = QueryBudget(max_cost_usd=0.01)
        status = BudgetStatus(budget=budget)
        status.cost_usd = 0.02

        from deeprecall.core.guardrails import BudgetExceededError

        with pytest.raises(BudgetExceededError, match="Cost"):
            status.check(start_time=time.perf_counter())

    def test_cost_budget_check_passes_below_limit(self):
        budget = QueryBudget(max_cost_usd=1.00)
        status = BudgetStatus(budget=budget)
        status.cost_usd = 0.50
        status.check(start_time=time.perf_counter())

    def test_cost_budget_utilization(self):
        budget = QueryBudget(max_cost_usd=1.00)
        status = BudgetStatus(budget=budget)
        status.cost_usd = 0.25
        assert status.utilization["cost"] == 0.25
