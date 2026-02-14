"""DeepRecall -- Recursive reasoning engine for AI agents and vector databases.

Usage:
    from deeprecall import DeepRecall
    from deeprecall.vectorstores import ChromaStore

    store = ChromaStore(collection_name="my_docs")
    store.add_documents(["Document text..."])

    engine = DeepRecall(vectorstore=store, backend="openai",
                        backend_kwargs={"model_name": "gpt-4o-mini"})
    result = engine.query("Your question here")
    print(result.answer)
"""

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.types import DeepRecallResult, SearchResult, Source

# Convenience alias
DeepRecall = DeepRecallEngine

__version__ = "0.1.0"
__all__ = [
    "DeepRecall",
    "DeepRecallEngine",
    "DeepRecallConfig",
    "DeepRecallResult",
    "SearchResult",
    "Source",
]
