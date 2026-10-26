default:
    just --list

ruff:
    uv sync --quiet --frozen --only-group ruff
    uv run --no-sync -- ruff check

typecheck:
    uv sync --quiet --frozen --no-dev --group typing
    uv run --no-sync -- mypy .
    uv run --no-sync -- ty check
    uv run --no-sync -- pyrefly check

lint-imports:
    uv sync --quiet --frozen --dev
    uv run --no-sync -- lint-imports

deptry:
    uv sync --quiet --frozen --dev
    uv run --no-sync -- deptry .

license:
    uv sync --quiet --frozen --no-dev --group license
    uv run --quiet --no-sync -- pip-licenses

lint: ruff typecheck lint-imports deptry license

test:
    uv sync --quiet --frozen --no-dev --group test
    uv run --no-sync -- coverage run -m pytest --import-mode importlib
    uv run --no-sync -- coverage report -m
    uv run --no-sync -- coverage xml -o ./coverage.xml

lowest-venv := ".venv-lowest"

# Run tests on the minimum supported Python with the lowest declared direct dependencies
test-lowest:
    uv venv --quiet --clear --python 3.13 {{ lowest-venv }}
    VIRTUAL_ENV={{ lowest-venv }} uv pip install --quiet --resolution lowest-direct . --group test --group examples
    {{ lowest-venv }}/bin/python -m pytest --import-mode importlib

docs-addr := "localhost:8000"

# Generate documentation derived from repository sources
generate-docs:
    uv sync --quiet --frozen --no-dev --group docs
    uv run --no-sync -- scripts/gen_index_page.py

# Serve the documentation
serve-docs: generate-docs
    uv run --no-sync -- zensical serve --dev-addr {{ docs-addr }}

# Build the documentation
build-docs: generate-docs
    uv run --no-sync -- zensical build --strict
