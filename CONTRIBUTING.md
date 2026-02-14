# Contributing to DeepRecall

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/kothapavan/deeprecall.git
cd deeprecall
pip install -e ".[all,dev,test]"
```

## Development Workflow

```bash
# Run linter
make lint

# Auto-fix formatting
make format

# Run tests
make test

# Run everything
make check
```

## Code Style

- **Formatter**: ruff (line-length 100, Python 3.11+)
- **Types**: Use explicit type hints everywhere
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google-style docstrings for all public methods
- **File size**: Keep files under 500 lines

## Adding a New Vector Store

1. Create `deeprecall/vectorstores/your_store.py`
2. Subclass `BaseVectorStore` and implement all four methods
3. Add lazy import in `deeprecall/vectorstores/__init__.py`
4. Add optional dependency in `pyproject.toml`
5. Add tests in `tests/test_vectorstores/`
6. Add an example in `examples/`

## Adding a New Framework Adapter

1. Create `deeprecall/adapters/your_framework.py`
2. Follow the pattern of existing adapters (langchain.py, llamaindex.py)
3. Add optional dependency in `pyproject.toml`
4. Add tests and examples

## PR Guidelines

- Small, focused diffs -- one change per PR
- Include tests for new functionality
- Update docs/examples if adding a feature
- All CI checks must pass

## Roadmap

- [ ] More vector stores (Weaviate, pgvector, FAISS)
- [ ] Async support for the engine
- [ ] Streaming responses in the engine
- [ ] Richer reasoning trace extraction
- [ ] Benchmarks against standard RAG
- [ ] Multi-modal document support
- [ ] Custom embedding function registry
