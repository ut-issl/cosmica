# 📚 Development rules and tips

## Rules

- Use SI units for all calculations. (e.g., radians instead of degrees, meters instead of kilometers)
- When generating random numbers, add an input argument of type `np.random.Generator` to the function and use it to generate random numbers. This will make the function deterministic and reproducible. See: [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator).
- Use type hints as much as possible. This will help you catch bugs early and make your code more readable.
- To add a new dependency, run `uv add <package-name>` in the root directory of the repository. This will automatically update the `pyproject.toml` file and install the package. DO NOT use `pip` to install packages.
  - After adding a new dependency, change the version constraint to `>=` in `pyproject.toml` unless you have a good reason not to.

## Linting

- Running `uv run ruff check .` in the root directory of the repository (after activating the virtual environment) will run various checks on the codebase. Running this before committing your changes is highly recommended. Running `uv run ruff format .` will also format the codebase. See [Ruff](https://docs.astral.sh/ruff/) for more information.
  - If you develop on Visual Studio Code, you can also install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and see the results of `uv run ruff check .` in the editor.
- Running `uv run mypy .` in the root directory of the repository (after activating the virtual environment) will run static type checking on the codebase. Running this before committing your changes is recommended. See [Mypy](https://mypy.readthedocs.io/en/stable/) for more information.
  - If you develop on Visual Studio Code, you can alternatively install the [Pylance extension](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) and see the results of type checking in the editor.

## Tips

- Run command with `uv run` to ensure that the command is run in the correct environment.
- Python can be _very_ slow if implemented naively.
  - Make sure to use Numpy arrays and vectorize your code as much as possible.
  - Use `np.datetime64` instead of `datetime.datetime` for consistency.
  - Avoid using `for` loops for hot code paths.
  - Consider using Numba to speed up your code.
- Always leave room for future expansion. 🔭
  - Use abstract base classes (ABCs) and abstract methods to define interfaces, and implement for the current specific use case in a concrete class.

## 🔗 Useful links

- [PyMap3D API Documentation](https://geospace-code.github.io/pymap3d/)
