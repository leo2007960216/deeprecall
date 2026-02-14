.PHONY: install install-dev install-all lint format test check clean

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pip install -e ".[dev,test]"

install-all:
	pip install -e ".[all,dev,test]"

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=deeprecall --cov-report=term-missing

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
