# jax_spiel

A JAX-first reimplementation of core [OpenSpiel](https://github.com/deepmind/open_spiel) primitives. The project aims to provide
functional, differentiable game representations and learning utilities leveraging JAX transformations.

## Current Status
- Tensor normal-form games with immutable dataclasses.
- Matching Pennies example with JIT-compatible payoff computation and best response utilities.
- Pytest-based unit tests covering the initial conversion.

## Development
Create a virtual environment and install dependencies:

```bash
pip install -e .[test]
```

Run the test suite:

```bash
pytest
```
