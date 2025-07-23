test:
	pytest

lint:
	ruff check .

typecheck:
	mypy .

coverage:
	coverage run -m pytest
	coverage report

install:
	poetry install
