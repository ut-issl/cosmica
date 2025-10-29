default:
    just --list

ruff:
    uv run -- ruff check

mypy:
    uv sync --quiet --frozen --no-dev --group typing
    uv run --no-sync -- mypy .

lint-imports:
    uv sync --quiet --frozen --dev
    uv run --no-sync -- lint-imports

deptry:
    uv sync --quiet --frozen
    uv run --no-sync -- deptry src

license:
    uv sync --quiet --frozen --no-dev --group license
    uv run --quiet --no-sync -- pip-licenses

lint:
    -just --justfile {{justfile()}} ruff
    -just --justfile {{justfile()}} mypy
    -just --justfile {{justfile()}} lint-imports
    -just --justfile {{justfile()}} deptry
    -just --justfile {{justfile()}} license

pytest:
    uv sync --quiet --frozen --no-dev --group test
    uv run --no-sync -- coverage run -m pytest --import-mode importlib
    uv run --no-sync -- coverage report -m
    uv run --no-sync -- coverage xml -o ./coverage.xml


docs-addr := "localhost:8000"
# Serve the documentation
serve-docs:
    uv run -- mkdocs serve --dev-addr {{docs-addr}}

# Build the documentation
build-docs:
    uv run -- mkdocs build
