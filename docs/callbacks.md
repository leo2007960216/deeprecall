# Callbacks

DeepRecall provides a callback system for observability, logging, and monitoring.

## Built-in Callbacks

### ConsoleCallback

Real-time rich console output showing each reasoning step:

```python
from deeprecall import DeepRecallConfig, ConsoleCallback

config = DeepRecallConfig(
    callbacks=[ConsoleCallback(show_code=True, show_output=True)],
)
```

### JSONLCallback

Structured logging to a JSONL file:

```python
from deeprecall import JSONLCallback

config = DeepRecallConfig(
    callbacks=[JSONLCallback(log_dir="./logs")],
)
```

Events logged: `query_start`, `reasoning_step`, `search`, `sub_llm_call`, `progress`, `iteration_start`, `iteration_complete`, `query_end`, `error`, `budget_warning`.

### UsageTrackingCallback

Aggregates usage across queries for billing:

```python
from deeprecall.core.callbacks import UsageTrackingCallback

tracker = UsageTrackingCallback()
config = DeepRecallConfig(callbacks=[tracker])
engine = DeepRecall(vectorstore=store, config=config)

engine.query("question 1")
engine.query("question 2")

print(tracker.summary())
# {"total_queries": 2, "total_tokens": 8500, "total_searches": 12, ...}
```

## Custom Callbacks

Implement `BaseCallback` and override any hooks you need:

```python
from deeprecall.core.callbacks import BaseCallback

class SlackAlertCallback(BaseCallback):
    def on_budget_warning(self, status):
        send_slack_alert(f"Budget exceeded: {status.exceeded_reason}")

    def on_error(self, error):
        send_slack_alert(f"DeepRecall error: {error}")
```

## Available Hooks

| Hook | When it fires |
|------|---------------|
| `on_query_start(query, config)` | Query begins |
| `on_iteration_start(iteration)` | Before each RLM reasoning iteration (v0.4+) |
| `on_reasoning_step(step, budget_status)` | After each RLM iteration (with full step data) |
| `on_iteration_complete(iteration, has_final_answer)` | After each RLM iteration completes (v0.4+) |
| `on_search(query, num_results, time_ms)` | After each vector store search |
| `on_sub_llm_call(prompt, response, time_ms)` | After each `llm_query()` sub-call |
| `on_progress(event, data)` | Generic progress events |
| `on_query_end(result)` | Query completes |
| `on_error(error)` | Unrecoverable error |
| `on_budget_warning(status)` | Budget limit exceeded |

All hooks are optional -- unimplemented ones are silently skipped. Exceptions in callbacks never propagate to the caller.

### Iteration Lifecycle (v0.4+)

The `on_iteration_start` and `on_iteration_complete` hooks provide finer-grained control than `on_reasoning_step`. They fire at the beginning and end of each RLM iteration respectively. `on_iteration_complete` includes a `has_final_answer` flag so you can detect when the LLM has finished reasoning.

```python
class ProgressBarCallback(BaseCallback):
    def on_iteration_start(self, iteration):
        print(f"[{iteration}] Thinking...")

    def on_iteration_complete(self, iteration, has_final_answer):
        if has_final_answer:
            print(f"[{iteration}] Done!")
        else:
            print(f"[{iteration}] Continuing...")
```
