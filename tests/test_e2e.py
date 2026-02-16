"""End-to-end test for DeepRecall -- tests the full pipeline.

Run with: OPENAI_API_KEY=sk-... python -m pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestEndToEnd:
    """Full end-to-end tests exercising DeepRecall with a real LLM."""

    @pytest.fixture(autouse=True)
    def setup_store(self):
        """Create a ChromaStore with sample documents."""
        from deeprecall.vectorstores.chroma import ChromaStore

        self.store = ChromaStore(collection_name=f"e2e_test_{int(time.time())}")
        self.store.add_documents(
            documents=[
                "Python was created by Guido van Rossum and first released in 1991. "
                "It emphasizes code readability with its notable use of significant whitespace.",
                "Rust is a multi-paradigm systems programming language focused on safety, "
                "especially safe concurrency. Rust is syntactically similar to C++.",
                "JavaScript was invented by Brendan Eich in 1995. It is a high-level, "
                "interpreted programming language that conforms to the ECMAScript specification.",
                "Go was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. "
                "It was announced in 2009 and is valued for its simplicity and concurrency.",
                "TypeScript is a strict syntactical superset of JavaScript developed by Microsoft. "
                "It adds optional static typing and was first released in October 2012.",
            ],
            metadatas=[
                {"language": "Python", "year": 1991, "type": "interpreted"},
                {"language": "Rust", "year": 2010, "type": "compiled"},
                {"language": "JavaScript", "year": 1995, "type": "interpreted"},
                {"language": "Go", "year": 2009, "type": "compiled"},
                {"language": "TypeScript", "year": 2012, "type": "compiled"},
            ],
            ids=["python", "rust", "javascript", "go", "typescript"],
        )
        yield
        # Cleanup not strictly needed for in-memory ChromaDB

    @pytest.mark.flaky(reruns=2, reason="LLM output is non-deterministic")
    def test_basic_query(self):
        """Test a basic query returns a well-formed result."""
        from deeprecall import DeepRecall

        engine = DeepRecall(
            vectorstore=self.store,
            backend="openai",
            backend_kwargs={
                "model_name": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            verbose=True,
            max_iterations=10,
        )

        result = engine.query("Which programming language was created first, Python or Go?")

        # Verify result structure (LLM answers are non-deterministic so checks are lenient)
        assert result.answer, "Answer should not be empty"
        assert len(result.answer) > 5, "Answer should have content"
        assert result.query == "Which programming language was created first, Python or Go?"
        assert result.execution_time > 0, "Execution time should be positive"
        assert result.usage.total_calls > 0, "Should have made LLM calls"
        assert len(result.reasoning_trace) > 0, "Should have reasoning steps"

        # Print result for human inspection
        print("\n" + "=" * 60)
        print(f"QUERY: {result.query}")
        print(f"ANSWER: {result.answer}")
        print(f"SOURCES: {len(result.sources)}")
        print(f"STEPS: {len(result.reasoning_trace)}")
        print(f"TIME: {result.execution_time:.2f}s")
        print(f"LLM CALLS: {result.usage.total_calls}")
        print(f"TOKENS: {result.usage.total_input_tokens + result.usage.total_output_tokens}")
        print("=" * 60)

    def test_search_server_integration(self):
        """Test that the search server correctly serves vector store results."""
        from deeprecall.core.search_server import SearchServer

        server = SearchServer(self.store)
        server.start()

        try:
            import json
            import urllib.request

            # Test search endpoint
            data = json.dumps({"query": "Python programming", "top_k": 2}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read())

            assert len(results) == 2
            assert any("Python" in r["content"] for r in results)
            print(f"\nSearch server returned {len(results)} results for 'Python programming'")
            for r in results:
                print(f"  Score: {r['score']:.3f} | {r['content'][:80]}...")

            # Verify source tracking
            sources = server.get_accessed_sources()
            assert len(sources) > 0
            print(f"  Tracked {len(sources)} unique sources")
        finally:
            server.stop()

    def test_add_documents_via_engine(self):
        """Test adding documents through the engine convenience method."""
        from deeprecall import DeepRecall

        engine = DeepRecall(
            vectorstore=self.store,
            backend="openai",
            backend_kwargs={
                "model_name": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )

        initial_count = self.store.count()
        engine.add_documents(
            documents=["Kotlin is a cross-platform language developed by JetBrains."],
            metadatas=[{"language": "Kotlin", "year": 2011}],
        )
        assert self.store.count() == initial_count + 1
        print(f"\nAdded 1 document via engine. Total: {self.store.count()}")
