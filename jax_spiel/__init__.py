"""JAX-first reimplementation of OpenSpiel primitives."""

from .tensor_game import TensorGame, matching_pennies, expected_payoff, best_response

__all__ = [
    "TensorGame",
    "matching_pennies",
    "expected_payoff",
    "best_response",
]
