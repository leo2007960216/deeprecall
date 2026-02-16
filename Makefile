.PHONY: install install-dev install-all lint format test test-e2e test-cov check build clean

IGNORE_E2E := --ignore=tests/test_e2e.py --ignore=tests/test_integration_redis_otel.py

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
	pytest tests/ $(IGNORE_E2E) -v

test-e2e:
	pytest tests/test_e2e.py -v -s

test-cov:
	pytest tests/ $(IGNORE_E2E) -v --cov=deeprecall --cov-report=term-missing

check: lint test

build:
	python -m build
	twine check dist/*

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
