"""LlamaIndex adapter for DeepRecall.

Provides DeepRecallQueryEngine and DeepRecallRetriever for use in
LlamaIndex pipelines.

Install: pip install deeprecall[llamaindex]
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    # Response moved in llama-index-core >= 0.10
    try:
        from llama_index.core.base.response.schema import Response
    except ImportError:
        from llama_index.core.response.schema import Response  # type: ignore[no-redef]
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex adapters. "
        "Install it with: pip install deeprecall[llamaindex]"
    ) from None

from deeprecall.core.engine import DeepRecallEngine


class DeepRecallQueryEngine(CustomQueryEngine):
    """LlamaIndex query engine powered by DeepRecall.

    Uses RLM's recursive reasoning to answer queries over vector DB documents,
    providing richer answers than standard RAG query engines.

    Args:
        engine: A configured DeepRecallEngine instance.

    Example:
        ```python
        from deeprecall import DeepRecall
        from deeprecall.vectorstores import ChromaStore
        from deeprecall.adapters.llamaindex import DeepRecallQueryEngine

        store = ChromaStore(collection_name="docs")
        engine = DeepRecall(vectorstore=store, backend="openai",
                            backend_kwargs={"model_name": "gpt-4o-mini"})
        query_engine = DeepRecallQueryEngine(engine=engine)

        response = query_engine.query("What are the main themes?")
        print(response)
        ```
    """

    engine: DeepRecallEngine

    class Config:
        arbitrary_types_allowed = True

    def custom_query(self, query_str: str) -> Response:
        result = self.engine.query(query_str)

        # Build source nodes from DeepRecall sources
        source_nodes = []
        for source in result.sources:
            node = TextNode(
                text=source.content,
                metadata={**source.metadata, "score": source.score},
                id_=source.id,
            )
            source_nodes.append(NodeWithScore(node=node, score=source.score))

        return Response(
            response=result.answer,
            source_nodes=source_nodes,
            metadata={
                "execution_time": result.execution_time,
                "usage": result.usage.to_dict(),
                "query": result.query,
            },
        )


class DeepRecallRetriever(BaseRetriever):
    """LlamaIndex retriever that uses DeepRecall's vector store search.

    Performs a direct vector store search (without recursive reasoning)
    for use in custom LlamaIndex pipelines that handle synthesis separately.

    Args:
        engine: A configured DeepRecallEngine instance.
        top_k: Number of results to retrieve.
    """

    def __init__(self, engine: DeepRecallEngine, top_k: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self._engine = engine
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self._engine.vectorstore.search(
            query=query_bundle.query_str,
            top_k=self._top_k,
        )

        nodes = []
        for result in results:
            node = TextNode(
                text=result.content,
                metadata=result.metadata,
                id_=result.id,
            )
            nodes.append(NodeWithScore(node=node, score=result.score))

        return nodes
