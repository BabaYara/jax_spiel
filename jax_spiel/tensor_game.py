"""JAX-compatible tensor games (normal-form matrix games)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TensorGame:
    """Normal-form game represented as a payoff tensor."""

    name: str
    payoffs: jnp.ndarray  # shape: (A0, A1, players)

    def __post_init__(self):
        if self.payoffs.ndim != 3:
            raise ValueError("payoffs must be a rank-3 tensor")
        if self.payoffs.shape[-1] != 2:
            raise ValueError("only two-player games are supported")

    @property
    def num_players(self) -> int:
        return self.payoffs.shape[-1]

    @property
    def num_actions(self) -> Tuple[int, int]:
        return (self.payoffs.shape[0], self.payoffs.shape[1])


def matching_pennies() -> TensorGame:
    """Creates the Matching Pennies game."""

    payoffs = jnp.array(
        [
            [[-1.0, 1.0], [1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]
    )
    return TensorGame(name="matching_pennies", payoffs=payoffs)


def _validate_policy(policy: jnp.ndarray, num_actions: int, label: str) -> None:
    if policy.shape != (num_actions,):
        raise ValueError(f"{label} must have shape ({num_actions},), got {policy.shape}")


def expected_payoff(game: TensorGame, player0_policy: jnp.ndarray, player1_policy: jnp.ndarray) -> jnp.ndarray:
    """Computes expected payoffs for both players."""

    num_actions0, num_actions1 = game.num_actions
    _validate_policy(player0_policy, num_actions0, "player0_policy")
    _validate_policy(player1_policy, num_actions1, "player1_policy")

    return jnp.einsum("i,j,ijc->c", player0_policy, player1_policy, game.payoffs)


def best_response(game: TensorGame, opponent_policy: jnp.ndarray, player: int) -> jnp.ndarray:
    """Returns a deterministic best response distribution."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    num_actions = game.num_actions[player]
    _validate_policy(opponent_policy, game.num_actions[1 - player], "opponent_policy")

    if player == 0:
        action_payoffs = jnp.einsum("j,ij->i", opponent_policy, game.payoffs[..., 0])
    else:
        action_payoffs = jnp.einsum("i,ij->j", opponent_policy, game.payoffs[..., 1])

    best_action = jnp.argmax(action_payoffs)
    return jax.nn.one_hot(best_action, num_actions)


__all__ = ["TensorGame", "matching_pennies", "expected_payoff", "best_response"]
