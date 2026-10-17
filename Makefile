.PHONY: install test lint typecheck format check report train serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gate as CI
check: lint typecheck
	pytest --cov-fail-under=80

# Synthetic study: docs/results.md and docs/img/*.png (under a minute on a laptop)
report:
	essay-scorer report

train:
	essay-scorer train

serve: train
	essay-scorer serve
