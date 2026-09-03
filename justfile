default:
    just --list

ruff:
    uv run -- ruff check

typecheck:
    uv sync --quiet --frozen --no-dev --group typing
    uv run --no-sync -- mypy .
    uv run --no-sync -- ty check
    uv run --no-sync -- pyrefly check

lint-imports:
    uv sync --quiet --frozen --dev
    uv run --no-sync -- lint-imports

deptry:
    uv sync --quiet --frozen
    uv run --no-sync -- deptry .

license:
    uv sync --quiet --frozen --no-dev --group license
    uv run --quiet --no-sync -- pip-licenses

lint:
    just --justfile {{ justfile() }} ruff
    just --justfile {{ justfile() }} typecheck
    just --justfile {{ justfile() }} lint-imports
    just --justfile {{ justfile() }} deptry
    just --justfile {{ justfile() }} license

test:
    uv sync --quiet --frozen --no-dev --group test
    uv run --no-sync -- coverage run -m pytest --import-mode importlib
    uv run --no-sync -- coverage report -m
    uv run --no-sync -- coverage xml -o ./coverage.xml

docs-addr := "localhost:8000"

# Generate documentation derived from repository sources
generate-docs:
    uv run -- scripts/gen_ref_pages.py

# Serve the documentation
serve-docs: generate-docs
    uv run -- zensical serve --dev-addr {{ docs-addr }}

# Build the documentation
build-docs: generate-docs
    uv run -- zensical build --strict
