# DeepRecall Examples

Runnable examples covering every major feature. Each file is self-contained -- just install the right extras and set your API key.

## Setup

```bash
pip install deeprecall[all]      # install everything
export OPENAI_API_KEY=sk-...     # or add to .env file
```

## Examples

| # | File | Feature | Extras needed |
|---|------|---------|---------------|
| 01 | [quickstart](01_quickstart.py) | ChromaDB + basic query | `chroma` |
| 02 | [milvus](02_milvus_example.py) | Milvus + embeddings | `milvus` |
| 03 | [langchain](03_langchain_agent.py) | LangChain retriever & chat model | `chroma,langchain` |
| 04 | [llamaindex](04_llamaindex_rag.py) | LlamaIndex query engine | `chroma,llamaindex` |
| 05 | [openai_api](05_openai_api.py) | OpenAI-compatible REST API | `chroma,server` |
| 06 | [multi_hop](06_multi_hop_reasoning.py) | Multi-hop reasoning showcase | `chroma` |
| 07 | [budget](07_budget_and_guardrails.py) | Budget guardrails + reasoning trace | `chroma` |
| 08 | [caching](08_caching.py) | InMemory, Disk (SQLite), Redis cache | `chroma` (+ `redis`) |
| 09 | [callbacks](09_callbacks_and_monitoring.py) | Console, JSONL, OTel, custom callbacks | `chroma,rich` (+ `otel`) |
| 10 | [async_batch](10_async_and_batch.py) | AsyncDeepRecall + batch queries | `chroma` |
| 11 | [qdrant](11_qdrant_example.py) | Qdrant vector store | `qdrant` |
| 12 | [pinecone](12_pinecone_example.py) | Pinecone managed vector DB | `pinecone` |
| 13 | [faiss](13_faiss_example.py) | FAISS local index + save/load | `faiss` |
| 14 | [production](14_production_ready.py) | Full production setup (retry, cache, errors) | `chroma,rich` |

## External Services Required

| Example | Service | How to start |
|---------|---------|-------------|
| 02 | Milvus | `docker compose up -d` ([docs](https://milvus.io/docs/install_standalone-docker.md)) |
| 05 | DeepRecall server | `deeprecall serve --vectorstore chroma --collection api_demo --port 8000` |
| 08 (Redis) | Redis | `docker run -d -p 6379:6379 redis:7` |
| 09 (OTel) | Jaeger | `docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one` |
| 11 | Qdrant | `docker run -d -p 6333:6333 qdrant/qdrant` |
| 12 | Pinecone | [Sign up](https://app.pinecone.io/) (managed service) |

## Running

```bash
# Run any example
python examples/01_quickstart.py

# Run with verbose output
python examples/07_budget_and_guardrails.py
```
