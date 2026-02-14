# Architecture

## Overview

DeepRecall bridges two technologies:
- **RLM** (Recursive Language Models) -- MIT's paradigm for recursive LLM inference
- **Vector Databases** -- efficient similarity search over embeddings

## Components

```
┌─────────────────────────────────────────────────────┐
│                    DeepRecall Engine                 │
│                                                     │
│  ┌───────────────┐  ┌────────────────────────────┐  │
│  │  DeepRecall    │  │  Search Server (HTTP)      │  │
│  │  Config        │  │  Wraps vector store         │  │
│  └───────┬───────┘  │  for REPL access            │  │
│          │          └──────────┬─────────────────┘  │
│          │                     │                     │
│  ┌───────v─────────────────────v──────────────────┐  │
│  │                RLM Instance                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ System   │  │ Local    │  │ LM Handler   │  │  │
│  │  │ Prompt   │  │ REPL     │  │ (TCP Server) │  │  │
│  │  │ + search │  │ + search │  │ Routes LLM   │  │  │
│  │  │ _db docs │  │ _db() fn │  │ API calls    │  │  │
│  │  └──────────┘  └──────────┘  └──────────────┘  │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## How the Bridge Works

1. **Search Server**: When `engine.query()` is called, DeepRecall starts a lightweight HTTP server on a random port that wraps the vector store's `search()` method.

2. **Setup Code Injection**: DeepRecall generates Python setup code that defines a `search_db()` function. This function uses `urllib` to call the search server. The setup code is passed to the RLM's `LocalREPL` via the `setup_code` parameter.

3. **Custom System Prompt**: The RLM receives a system prompt that describes the `search_db()` function and provides examples of multi-hop reasoning patterns.

4. **Recursive Loop**: The RLM's standard recursive loop runs. The LLM can call `search_db()`, `llm_query()`, and write arbitrary Python code. Each iteration's output feeds into the next.

5. **Source Tracking**: The search server tracks all documents that were accessed during the reasoning process. These become the `sources` in the final result.

## Key Design Decisions

- **No RLM modifications**: DeepRecall uses RLM as a pip dependency and doesn't modify its source code. The bridge is achieved entirely through `setup_code` and `custom_system_prompt`.

- **HTTP bridge for search**: Using HTTP (stdlib `urllib`) instead of direct function injection ensures the search works across all RLM environment types (local, Docker, Modal, etc.).

- **Lazy imports**: Vector store and framework adapters use lazy imports to avoid requiring all dependencies.

- **Optional dependencies**: Each vector DB and framework is an optional extra (`pip install deeprecall[chroma]`).
