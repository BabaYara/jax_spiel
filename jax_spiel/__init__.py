"""JAX-first reimplementation of OpenSpiel primitives."""

from .tensor_game import (
    SIMULTANEOUS_PLAYER,
    TERMINAL_PLAYER,
    TensorGame,
    TensorState,
    best_response,
    expected_payoff,
    joint_action_payoff,
    matching_pennies,
)

__all__ = [
    "SIMULTANEOUS_PLAYER",
    "TERMINAL_PLAYER",
    "TensorGame",
    "TensorState",
    "best_response",
    "expected_payoff",
    "joint_action_payoff",
    "matching_pennies",
]
