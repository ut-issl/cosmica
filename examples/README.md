# Example usage of the COSMICA library

This directory contains examples demonstrating the use of the COSMICA library.

## Basic simulation script

`basic_simulation.py` is a small, plain Python script that sets up a Walker Delta constellation, a user satellite, a gateway, propagates their dynamics, builds simple topologies, calculates communication link performance, and prints a sample route.

```bash
uv run examples/basic_simulation.py
```

## Marimo orbit propagation notebook

Marimo is an opt-in dependency in the `examples` group and is not installed by the default development or typing groups. Its Loro dependency currently does not support Python 3.15, so use Python 3.14 for this notebook. The notebook is excluded from default type checking; Ruff still checks it.

To run `orbit_propagation.py`, clone this repository, [install uv](https://docs.astral.sh/uv/getting-started/installation/), and run:

```bash
uv run --group examples --python 3.14 marimo edit examples/orbit_propagation.py
```
