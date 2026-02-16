"""DeepRecall CLI -- command-line interface for serving, querying, and ingesting."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from deeprecall.core.engine import DeepRecallEngine
    from deeprecall.vectorstores.base import BaseVectorStore
from dotenv import load_dotenv

load_dotenv()

_VS_CHOICES = ["chroma", "milvus", "qdrant", "pinecone", "faiss"]


@click.group()
@click.version_option(version=None, prog_name="deeprecall", package_name="deeprecall")
def main() -> None:
    """DeepRecall -- Recursive reasoning over your data."""


@main.command()
@click.option(
    "--vectorstore",
    type=click.Choice(_VS_CHOICES),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory (ChromaDB only).")
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", default=8000, type=int, help="Server port.")
@click.option("--backend", default="openai", help="LLM backend.")
@click.option("--model", default="gpt-4o-mini", help="LLM model name.")
@click.option("--api-keys", default=None, help="Comma-separated API keys for auth.")
@click.option("--rate-limit", default=None, type=int, help="Requests per minute per key.")
def serve(
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    host: str,
    port: int,
    backend: str,
    model: str,
    api_keys: str | None,
    rate_limit: int | None,
) -> None:
    """Start the OpenAI-compatible API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn required. Install with: pip install deeprecall[server]")
        sys.exit(1)

    engine = _build_engine(vectorstore, collection, persist_dir, backend, model)

    from deeprecall.adapters.openai_server import create_app

    keys_list = api_keys.split(",") if api_keys else None
    app = create_app(engine, api_keys=keys_list, requests_per_minute=rate_limit)

    click.echo(f"Starting DeepRecall server on {host}:{port}")
    click.echo(f"  Vector store: {vectorstore} ({collection})")
    click.echo(f"  Backend: {backend}/{model}")
    if keys_list:
        click.echo(f"  Auth: {len(keys_list)} API key(s) configured")
    if rate_limit:
        click.echo(f"  Rate limit: {rate_limit} req/min")
    click.echo(f"  Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.argument("query_text")
@click.option(
    "--vectorstore",
    type=click.Choice(_VS_CHOICES),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory (ChromaDB only).")
@click.option("--backend", default="openai", help="LLM backend.")
@click.option("--model", default="gpt-4o-mini", help="LLM model name.")
@click.option("--verbose", is_flag=True, help="Show detailed output.")
@click.option("--max-searches", default=None, type=int, help="Max search calls budget.")
@click.option("--max-tokens", default=None, type=int, help="Max token budget.")
@click.option("--max-time", default=None, type=float, help="Max time budget (seconds).")
def query(
    query_text: str,
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    backend: str,
    model: str,
    verbose: bool,
    max_searches: int | None,
    max_tokens: int | None,
    max_time: float | None,
) -> None:
    """Query documents with recursive reasoning."""
    from deeprecall.core.guardrails import QueryBudget

    budget = None
    if any(v is not None for v in [max_searches, max_tokens, max_time]):
        budget = QueryBudget(
            max_search_calls=max_searches,
            max_tokens=max_tokens,
            max_time_seconds=max_time,
        )

    engine = _build_engine(vectorstore, collection, persist_dir, backend, model, verbose=verbose)
    result = engine.query(query_text, budget=budget)

    click.echo(f"\nAnswer: {result.answer}")
    click.echo(f"\nSources: {len(result.sources)}")
    click.echo(f"Steps: {len(result.reasoning_trace)}")
    click.echo(f"Time: {result.execution_time:.2f}s")
    click.echo(f"LLM calls: {result.usage.total_calls}")

    if result.confidence is not None:
        click.echo(f"Confidence: {result.confidence:.2f}")
    if result.error:
        click.echo(f"Warning: {result.error}")
    if result.budget_status and result.budget_status.get("budget_exceeded"):
        click.echo(f"Budget: {result.budget_status.get('exceeded_reason')}")


@main.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to documents.")
@click.option(
    "--vectorstore",
    type=click.Choice(_VS_CHOICES),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option(
    "--persist-dir",
    default=None,
    help="Persist directory (default: ./chroma_db for ChromaDB, ./faiss_index for FAISS).",
)
def ingest(
    path: str,
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
) -> None:
    """Ingest documents from a file or directory into the vector store."""
    store = _build_vectorstore(vectorstore, collection, persist_dir)

    documents: list[str] = []
    metadatas: list[dict] = []

    if os.path.isfile(path):
        files = [path]
    else:
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith((".txt", ".md", ".py", ".json", ".csv"))
        ]

    max_file_size = 50 * 1024 * 1024  # 50 MB per file
    for filepath in files:
        try:
            fsize = os.path.getsize(filepath)
        except OSError as e:
            click.echo(f"Warning: skipping {filepath}: {e}")
            continue
        if fsize > max_file_size:
            click.echo(
                f"Warning: skipping {filepath} ({fsize} bytes exceeds {max_file_size} limit)"
            )
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            click.echo(f"Warning: skipping {filepath} (not valid UTF-8)")
            continue
        documents.append(content)
        metadatas.append({"source": filepath, "filename": os.path.basename(filepath)})

    if not documents:
        click.echo("No documents found.")
        return

    ids = store.add_documents(documents=documents, metadatas=metadatas)
    click.echo(f"Ingested {len(ids)} documents into {vectorstore}/{collection}")
    click.echo(f"Total documents: {store.count()}")


@main.command()
@click.option(
    "--vectorstore",
    type=click.Choice(_VS_CHOICES),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory.")
@click.argument("doc_ids", nargs=-1, required=True)
def delete(
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    doc_ids: tuple[str, ...],
) -> None:
    """Delete documents by ID from the vector store."""
    store = _build_vectorstore(vectorstore, collection, persist_dir)
    store.delete(ids=list(doc_ids))
    click.echo(f"Deleted {len(doc_ids)} document(s). Remaining: {store.count()}")


@main.command()
@click.option("--path", default="deeprecall.toml", help="Path for the config file.")
def init(path: str) -> None:
    """Generate a starter config file."""
    config_content = """\
# DeepRecall Configuration
# See: https://github.com/kothapavan1998/deeprecall

[engine]
backend = "openai"
model = "gpt-4o-mini"
max_iterations = 15
top_k = 5
verbose = false

[vectorstore]
type = "chroma"
collection = "deeprecall"
persist_dir = "./chroma_db"

[budget]
# Uncomment to enable budget limits
# max_search_calls = 20
# max_tokens = 50000
# max_time_seconds = 60.0

[server]
host = "0.0.0.0"
port = 8000
# api_keys = ["key1", "key2"]
# rate_limit = 60

[cache]
enabled = false
type = "memory"    # "memory" or "disk"
ttl = 3600
max_size = 1000
"""
    if os.path.exists(path):
        click.echo(f"Config file already exists: {path}")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(config_content)
    click.echo(f"Created config file: {path}")


@main.command()
def status() -> None:
    """Show DeepRecall environment status."""
    import platform

    from deeprecall import __version__

    click.echo(f"Python:       {platform.python_version()}")
    click.echo(f"DeepRecall:   {__version__}")

    # RLM version
    try:
        from importlib.metadata import version as _v

        click.echo(f"RLM (rlms):   {_v('rlms')}")
    except Exception:
        click.echo("RLM (rlms):   not installed")

    # Check optional extras
    _extras = {
        "chromadb": "chroma",
        "pymilvus": "milvus",
        "qdrant-client": "qdrant",
        "pinecone": "pinecone",
        "faiss-cpu": "faiss",
        "redis": "redis",
        "opentelemetry-api": "otel",
        "fastapi": "server",
        "rich": "rich",
        "cohere": "rerank-cohere",
        "sentence-transformers": "rerank-cross-encoder",
    }
    installed = []
    for pkg, extra in _extras.items():
        try:
            from importlib.metadata import version as _v

            _v(pkg)
            installed.append(extra)
        except Exception:
            pass
    click.echo(f"Extras:       {', '.join(installed) if installed else 'none'}")


@main.command()
@click.option(
    "--queries", required=True, type=click.Path(exists=True), help="Path to queries JSON file."
)
@click.option("--output", default="benchmark_results.json", help="Output file for results.")
@click.option(
    "--vectorstore",
    type=click.Choice(_VS_CHOICES),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory.")
@click.option("--backend", default="openai", help="LLM backend.")
@click.option("--model", default="gpt-4o-mini", help="LLM model name.")
def benchmark(
    queries: str,
    output: str,
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    backend: str,
    model: str,
) -> None:
    """Run benchmark queries and collect metrics."""
    import json
    import time

    # Validate file size to prevent resource exhaustion
    try:
        file_size = os.path.getsize(queries)
    except OSError as e:
        click.echo(f"Error: cannot read queries file: {e}")
        sys.exit(1)

    max_size = 10 * 1024 * 1024  # 10 MB
    if file_size > max_size:
        click.echo(f"Error: queries file too large ({file_size} bytes, max {max_size}).")
        sys.exit(1)

    try:
        with open(queries, encoding="utf-8") as f:
            query_list = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"Error: invalid JSON in queries file: {e}")
        sys.exit(1)

    if not isinstance(query_list, list):
        click.echo("Error: queries file must contain a JSON array of strings.")
        sys.exit(1)

    engine = _build_engine(vectorstore, collection, persist_dir, backend, model)

    results = []
    total_start = time.perf_counter()

    for i, q in enumerate(query_list):
        q_text = q if isinstance(q, str) else q.get("query", "")
        click.echo(f"[{i + 1}/{len(query_list)}] {q_text[:80]}...")

        start = time.perf_counter()
        try:
            result = engine.query(q_text)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "query": q_text,
                    "answer_length": len(result.answer),
                    "sources": len(result.sources),
                    "steps": len(result.reasoning_trace),
                    "tokens": result.usage.total_input_tokens + result.usage.total_output_tokens,
                    "time_seconds": round(elapsed, 3),
                    "confidence": result.confidence,
                    "error": result.error,
                }
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "query": q_text,
                    "error": str(e),
                    "time_seconds": round(elapsed, 3),
                }
            )

    total_time = time.perf_counter() - total_start

    report = {
        "total_queries": len(results),
        "total_time_seconds": round(total_time, 3),
        "avg_time_seconds": round(total_time / len(results), 3) if results else 0,
        "results": results,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    click.echo(f"\nBenchmark complete: {len(results)} queries in {total_time:.1f}s")
    click.echo(f"Results written to: {output}")


def _default_persist_dir(vectorstore: str, persist_dir: str | None) -> str | None:
    """Return a sensible default persist directory when the user didn't specify one."""
    if persist_dir is not None:
        return persist_dir
    if vectorstore == "faiss":
        return "./faiss_index"
    if vectorstore == "chroma":
        return "./chroma_db"
    return None


def _build_vectorstore(
    vectorstore: str,
    collection: str,
    persist_dir: str | None = None,
) -> BaseVectorStore:
    """Build a vector store from CLI args."""
    persist_dir = _default_persist_dir(vectorstore, persist_dir)
    if vectorstore == "chroma":
        from deeprecall.vectorstores.chroma import ChromaStore

        return ChromaStore(collection_name=collection, persist_directory=persist_dir)
    elif vectorstore == "milvus":
        from deeprecall.vectorstores.milvus import MilvusStore

        return MilvusStore(collection_name=collection)
    elif vectorstore == "qdrant":
        from deeprecall.vectorstores.qdrant import QdrantStore

        return QdrantStore(collection_name=collection)
    elif vectorstore == "pinecone":
        from deeprecall.vectorstores.pinecone import PineconeStore

        return PineconeStore(index_name=collection)
    elif vectorstore == "faiss":
        from deeprecall.vectorstores.faiss import FAISSStore

        return FAISSStore(persist_path=persist_dir)
    else:
        raise ValueError(f"Unknown vectorstore: {vectorstore}")


_BACKEND_API_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "portkey": "PORTKEY_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _build_engine(
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> DeepRecallEngine:
    from deeprecall.core.engine import DeepRecallEngine

    env_var = _BACKEND_API_KEY_ENVS.get(backend, f"{backend.upper()}_API_KEY")
    api_key = os.getenv(env_var)
    store = _build_vectorstore(vectorstore, collection, persist_dir)
    return DeepRecallEngine(
        vectorstore=store,
        backend=backend,
        backend_kwargs={
            "model_name": model,
            "api_key": api_key,
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
