# python-package-template
This is a template repository for Python package development.

## Features

- [uv](https://docs.astral.sh/uv/) for Python project management.
- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- [mypy](https://mypy.readthedocs.io/en/stable/) for type checking.
- [pytest](https://docs.pytest.org/en/stable/) for testing.
- [pre-commit](https://pre-commit.com/) for pre-commit hooks.
- GitHub Actions for continuous integration.

## Setup

- [ ] Use this template to create a new repository.
- [ ] Install [uv](https://docs.astral.sh/uv/)
- [ ] Run `uv run pre-commit install` in the root directory of your new repository to set up pre-commit hooks.
- [ ] Modify `pyproject.toml` to set up your package metadata.
- [ ] Change the `src/package_name_goes_here` directory to your package name.

## Required Maintainance

- [ ] Run `uv run pre-commit autoupdate` to update pre-commit hooks.
- [ ] Update the `UV_VERSION` environment variable in the `.github/workflows/ci.yaml` file when a new minor version is released. (For example, from `0.7.x` to `0.8.x`).
