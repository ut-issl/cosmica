
# 🛠️ Setup

## 📋 Prerequisites

- [uv](https://docs.astral.sh/uv/)

## 🛠️ Recommended tools

The following tools are recommended (but not required) to work with this project:

- Visual Studio Code
  - Various settings are already configured in `.vscode/settings.json`
- Windows Subsystem for Linux (WSL) if you are using Windows
  - On native Windows, the `${workspaceFolder}/.venv/bin/*` paths in `.vscode/settings.json` will not work. You will need to change them to `${workspaceFolder}/.venv/Scripts/*` instead.

## 🚀 Installation

1. Clone this repository 🔄
2. Run `uv sync` in the root directory of the repository

## Install pre-commit hooks

This project uses [pre-commit](https://pre-commit.com) to run checks on every commit. To install the pre-commit hooks, run the following command:

```bash
uv run pre-commit install
```

## Documentation

Use `just serve-docs` to preview the documentation or `just build-docs` to build it. Both recipes generate the ignored home and API reference pages before running Zensical.
