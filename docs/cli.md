# CLI Reference

DeepRecall provides a command-line interface for serving, querying, ingesting, and managing documents.

```bash
pip install deeprecall[chroma,server]
```

---

## `deeprecall init`

Generate a starter configuration file.

```bash
deeprecall init                     # Creates deeprecall.toml
deeprecall init --path myconfig.toml
```

| Option | Default | Description |
|---|---|---|
| `--path` | `deeprecall.toml` | Output path for the config file |

---

## `deeprecall ingest`

Ingest documents from files or directories into the vector store.

```bash
deeprecall ingest --path ./docs/
deeprecall ingest --path report.txt --collection research --persist-dir ./db
```

Supported file types: `.txt`, `.md`, `.py`, `.json`, `.csv`.

| Option | Default | Description |
|---|---|---|
| `--path` | **required** | File or directory to ingest |
| `--vectorstore` | `chroma` | Backend: `chroma`, `milvus`, `qdrant`, `pinecone`, `faiss` |
| `--collection` | `deeprecall` | Collection/index name |
| `--persist-dir` | auto | Persist directory (`./chroma_db` for ChromaDB, `./faiss_index` for FAISS) |

---

## `deeprecall query`

Run a recursive reasoning query from the command line.

```bash
deeprecall query "What are the main themes?"
deeprecall query "Complex question" --max-searches 10 --max-time 30 --verbose
```

| Option | Default | Description |
|---|---|---|
| `QUERY_TEXT` | **required** | The question (positional argument) |
| `--vectorstore` | `chroma` | Backend: `chroma`, `milvus`, `qdrant`, `pinecone` |
| `--collection` | `deeprecall` | Collection/index name |
| `--persist-dir` | `None` | Persist directory (ChromaDB only) |
| `--backend` | `openai` | LLM backend |
| `--model` | `gpt-4o-mini` | LLM model name |
| `--verbose` | `False` | Show detailed step-by-step output |
| `--max-searches` | `None` | Budget: max vector DB search calls |
| `--max-tokens` | `None` | Budget: max total tokens |
| `--max-time` | `None` | Budget: max seconds (wall-clock) |

**Output:**

```
Answer: The main themes are...

Sources: 5
Steps: 3
Time: 4.52s
LLM calls: 7
Confidence: 0.87
```

---

## `deeprecall serve`

Start the OpenAI-compatible API server. See [Server](server.md) for full endpoint docs.

```bash
deeprecall serve
deeprecall serve --vectorstore milvus --collection research --port 9000
deeprecall serve --api-keys "sk-key1,sk-key2" --rate-limit 60
```

| Option | Default | Description |
|---|---|---|
| `--vectorstore` | `chroma` | Backend: `chroma`, `milvus`, `qdrant`, `pinecone` |
| `--collection` | `deeprecall` | Collection/index name |
| `--persist-dir` | `None` | Persist directory (ChromaDB only) |
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | `8000` | Server port |
| `--backend` | `openai` | LLM backend |
| `--model` | `gpt-4o-mini` | LLM model name |
| `--api-keys` | `None` | Comma-separated API keys for authentication |
| `--rate-limit` | `None` | Requests per minute per key |

---

## `deeprecall delete`

Delete documents by ID from the vector store.

```bash
deeprecall delete doc-1 doc-2 doc-3
deeprecall delete --collection research --vectorstore milvus doc-abc
```

| Option | Default | Description |
|---|---|---|
| `DOC_IDS` | **required** | One or more document IDs (positional) |
| `--vectorstore` | `chroma` | Backend: `chroma`, `milvus`, `qdrant`, `pinecone` |
| `--collection` | `deeprecall` | Collection/index name |
| `--persist-dir` | `./chroma_db` | Persist directory (ChromaDB only) |

---

## `deeprecall status`

Show version, installed extras, and system info.

```bash
deeprecall status
```

**Output:**

```
DeepRecall v0.3.0
Python: 3.12.8
RLM: 0.1.0
Installed extras: chroma, faiss, server, rich
```

---

## `deeprecall benchmark`

Run a set of queries from a JSON file and collect timing/quality metrics.

```bash
deeprecall benchmark --queries queries.json
deeprecall benchmark --queries queries.json --output results.json
```

| Option | Default | Description |
|---|---|---|
| `--queries` | **required** | Path to JSON file containing an array of query strings |
| `--output` | `None` | Path to write structured JSON results |
| `--vectorstore` | `chroma` | Backend: `chroma`, `milvus`, `qdrant`, `pinecone`, `faiss` |
| `--collection` | `deeprecall` | Collection/index name |
| `--persist-dir` | `None` | Persist directory |
| `--backend` | `openai` | LLM backend |
| `--model` | `gpt-4o-mini` | LLM model name |

**Limits:** JSON file must be under 10 MB and contain a JSON array.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used by default backend) |
| `COHERE_API_KEY` | Cohere API key (for `CohereReranker`) |

DeepRecall loads `.env` files automatically via `python-dotenv`.
