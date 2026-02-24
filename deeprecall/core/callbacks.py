"""Callback system for DeepRecall observability.

Provides hooks into the reasoning pipeline for monitoring, logging,
and custom integrations.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deeprecall.core.config import DeepRecallConfig
    from deeprecall.core.guardrails import BudgetStatus
    from deeprecall.core.types import DeepRecallResult, ReasoningStep


class BaseCallback:  # noqa: B024
    """Base class for DeepRecall callbacks.

    Implement any subset of hooks -- unimplemented methods are no-ops.
    Not using @abstractmethod intentionally: all hooks are optional.
    """

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:  # noqa: B027
        """Called when a query begins."""

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:  # noqa: B027
        """Called after each reasoning iteration."""

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:  # noqa: B027
        """Called after each vector store search."""

    def on_query_end(self, result: DeepRecallResult) -> None:  # noqa: B027
        """Called when a query completes (success or partial)."""

    def on_error(self, error: Exception) -> None:  # noqa: B027
        """Called when an unrecoverable error occurs."""

    def on_budget_warning(self, status: BudgetStatus) -> None:  # noqa: B027
        """Called when a budget limit is exceeded."""

    def on_sub_llm_call(self, prompt_preview: str, response_preview: str) -> None:  # noqa: B027
        """Called after a sub-LLM call completes."""

    def on_progress(self, message: str, percent: float | None = None) -> None:  # noqa: B027
        """Called to report progress updates."""

    def on_iteration_start(self, iteration: int) -> None:  # noqa: B027
        """Called before each RLM reasoning iteration begins."""

    def on_iteration_complete(self, iteration: int, has_final_answer: bool) -> None:  # noqa: B027
        """Called after each RLM reasoning iteration completes."""


class CallbackManager:
    """Manages and dispatches events to multiple callbacks."""

    def __init__(self, callbacks: list[BaseCallback] | None = None):
        self.callbacks = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        self.callbacks.append(callback)

    def _safe_call(self, cb: BaseCallback, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(cb, method)(*args, **kwargs)
        except Exception:
            _logger.warning(
                "Callback %s.%s failed",
                type(cb).__name__,
                method,
                exc_info=True,
            )

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_query_start", query, config)

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_reasoning_step", step, budget_status)

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_search", query, num_results, time_ms)

    def on_query_end(self, result: DeepRecallResult) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_query_end", result)

    def on_error(self, error: Exception) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_error", error)

    def on_budget_warning(self, status: BudgetStatus) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_budget_warning", status)

    def on_sub_llm_call(self, prompt_preview: str, response_preview: str) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_sub_llm_call", prompt_preview, response_preview)

    def on_progress(self, message: str, percent: float | None = None) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_progress", message, percent)

    def on_iteration_start(self, iteration: int) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_iteration_start", iteration)

    def on_iteration_complete(self, iteration: int, has_final_answer: bool) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_iteration_complete", iteration, has_final_answer)


# ---------------------------------------------------------------------------
# Built-in callback implementations
# ---------------------------------------------------------------------------


class ConsoleCallback(BaseCallback):
    """Rich console output showing reasoning steps in real time."""

    def __init__(self, show_code: bool = True, show_output: bool = True):
        self.show_code = show_code
        self.show_output = show_output
        self._start_time = threading.local()

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        from rich.console import Console
        from rich.panel import Panel

        self._start_time.value = time.perf_counter()
        console = Console()
        console.print(Panel(f"[bold]{query}[/bold]", title="Query", border_style="cyan"))

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        from rich.console import Console

        console = Console()
        start = getattr(self._start_time, "value", None) or time.perf_counter()
        elapsed = time.perf_counter() - start

        header = (
            f"[bold]Step {step.iteration}[/bold] | "
            f"{step.action} | "
            f"searches={budget_status.search_calls_used} | "
            f"{elapsed:.1f}s"
        )
        console.print(f"\n{'─' * 60}")
        console.print(header)

        if self.show_code and step.code:
            console.print(f"[dim]{step.code[:300]}{'...' if len(step.code) > 300 else ''}[/dim]")
        if self.show_output and step.output:
            console.print(
                f"[green]{step.output[:200]}{'...' if len(step.output) > 200 else ''}[/green]"
            )

    def on_query_end(self, result: DeepRecallResult) -> None:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        summary = (
            f"Time: {result.execution_time:.2f}s | "
            f"Sources: {len(result.sources)} | "
            f"Steps: {len(result.reasoning_trace)} | "
            f"Tokens: {result.usage.total_input_tokens + result.usage.total_output_tokens}"
        )
        if result.confidence is not None:
            summary += f" | Confidence: {result.confidence:.2f}"
        if result.error:
            summary += f"\n[red]Error: {result.error}[/red]"
        console.print(Panel(summary, title="Complete", border_style="green"))

    def on_budget_warning(self, status: BudgetStatus) -> None:
        from rich.console import Console

        Console().print(f"[bold red]Budget exceeded: {status.exceeded_reason}[/bold red]")


class JSONLCallback(BaseCallback):
    """Logs all events to a JSONL file for post-hoc analysis."""

    def __init__(self, log_dir: str, filename_prefix: str = "deeprecall"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{filename_prefix}_{ts}.jsonl")
        self._lock = threading.Lock()

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {"type": event_type, "timestamp": time.time(), **data}
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, default=str)
                f.write("\n")

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        self._write("query_start", {"query": query, "config": config.to_dict()})

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        self._write("reasoning_step", {"step": step.to_dict(), "budget": budget_status.to_dict()})

    def on_query_end(self, result: DeepRecallResult) -> None:
        self._write(
            "query_end",
            {
                "answer_length": len(result.answer),
                "sources": len(result.sources),
                "steps": len(result.reasoning_trace),
                "execution_time": result.execution_time,
                "error": result.error,
            },
        )

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        self._write(
            "search", {"query": query, "num_results": num_results, "time_ms": round(time_ms, 2)}
        )

    def on_error(self, error: Exception) -> None:
        self._write("error", {"error": str(error), "type": type(error).__name__})

    def on_budget_warning(self, status: BudgetStatus) -> None:
        self._write("budget_warning", {"status": status.to_dict()})

    def on_sub_llm_call(self, prompt_preview: str, response_preview: str) -> None:
        self._write(
            "sub_llm_call", {"prompt": prompt_preview[:200], "response": response_preview[:200]}
        )

    def on_progress(self, message: str, percent: float | None = None) -> None:
        self._write("progress", {"message": message, "percent": percent})

    def on_iteration_start(self, iteration: int) -> None:
        self._write("iteration_start", {"iteration": iteration})

    def on_iteration_complete(self, iteration: int, has_final_answer: bool) -> None:
        self._write(
            "iteration_complete",
            {"iteration": iteration, "has_final_answer": has_final_answer},
        )


class UsageTrackingCallback(BaseCallback):
    """Tracks cumulative usage across multiple queries for billing.

    Thread-safe: all counter mutations and reads are protected by a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_queries: int = 0
        self.total_tokens: int = 0
        self.total_searches: int = 0
        self.total_time: float = 0.0
        self.errors: int = 0

    def on_query_end(self, result: DeepRecallResult) -> None:
        with self._lock:
            self.total_queries += 1
            self.total_tokens += result.usage.total_input_tokens + result.usage.total_output_tokens
            self.total_time += result.execution_time
            if result.budget_status:
                self.total_searches += result.budget_status.get("search_calls_used", 0)
            if result.error:
                self.errors += 1

    def on_error(self, error: Exception) -> None:
        with self._lock:
            self.errors += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_queries": self.total_queries,
                "total_tokens": self.total_tokens,
                "total_searches": self.total_searches,
                "total_time": round(self.total_time, 2),
                "errors": self.errors,
            }


class ProgressCallback(BaseCallback):
    """Thread-safe callback that accumulates all progress events.

    Useful for UIs or test harnesses that need to inspect the full event stream.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def _record(self, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            self.events.append({"type": event_type, "timestamp": time.time(), **data})

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        self._record("query_start", {"query": query})

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        self._record("reasoning_step", {"iteration": step.iteration, "action": step.action})

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        self._record("search", {"query": query, "num_results": num_results, "time_ms": time_ms})

    def on_query_end(self, result: DeepRecallResult) -> None:
        self._record("query_end", {"answer_length": len(result.answer)})

    def on_error(self, error: Exception) -> None:
        self._record("error", {"error": str(error)})

    def on_budget_warning(self, status: BudgetStatus) -> None:
        self._record("budget_warning", {"reason": status.exceeded_reason})

    def on_sub_llm_call(self, prompt_preview: str, response_preview: str) -> None:
        self._record("sub_llm_call", {"prompt": prompt_preview, "response": response_preview})

    def on_progress(self, message: str, percent: float | None = None) -> None:
        self._record("progress", {"message": message, "percent": percent})

    def on_iteration_start(self, iteration: int) -> None:
        self._record("iteration_start", {"iteration": iteration})

    def on_iteration_complete(self, iteration: int, has_final_answer: bool) -> None:
        self._record(
            "iteration_complete",
            {"iteration": iteration, "has_final_answer": has_final_answer},
        )
