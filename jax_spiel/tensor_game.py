"""JAX-compatible tensor games (normal-form matrix games)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

SIMULTANEOUS_PLAYER = -1
TERMINAL_PLAYER = -2


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


@dataclass(frozen=True)
class TensorState:
    """State representation for a normal-form tensor game."""

    game: TensorGame
    joint_action: Tuple[int, int] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.joint_action is not None

    @property
    def current_player(self) -> int:
        return SIMULTANEOUS_PLAYER if not self.is_terminal else TERMINAL_PLAYER

    def legal_actions(self, player: int) -> jnp.ndarray:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if self.is_terminal:
            return jnp.array([], dtype=jnp.int32)
        num_actions = self.game.num_actions[player]
        return jnp.arange(num_actions, dtype=jnp.int32)

    def apply_joint_action(self, joint_action: Tuple[int, int]) -> "TensorState":
        if self.is_terminal:
            raise ValueError("cannot apply actions to a terminal state")
        if len(joint_action) != 2:
            raise ValueError("joint_action must contain two actions")

        action0, action1 = joint_action
        num_actions0, num_actions1 = self.game.num_actions

        if not (0 <= action0 < num_actions0):
            raise ValueError(f"action {action0} is invalid for player 0")
        if not (0 <= action1 < num_actions1):
            raise ValueError(f"action {action1} is invalid for player 1")

        return TensorState(game=self.game, joint_action=(int(action0), int(action1)))

    def returns(self) -> jnp.ndarray:
        if not self.is_terminal:
            return jnp.zeros(self.game.num_players, dtype=self.game.payoffs.dtype)

        action0, action1 = self.joint_action
        return self.game.payoffs[action0, action1, :]


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


def joint_action_payoff(game: TensorGame, joint_action: jnp.ndarray) -> jnp.ndarray:
    """Returns payoffs for a pure joint action profile."""

    joint_action = jnp.asarray(joint_action, dtype=jnp.int32)
    if joint_action.shape != (2,):
        raise ValueError("joint_action must be a length-2 array")

    return game.payoffs[joint_action[0], joint_action[1], :]


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


__all__ = [
    "SIMULTANEOUS_PLAYER",
    "TERMINAL_PLAYER",
    "TensorGame",
    "TensorState",
    "matching_pennies",
    "expected_payoff",
    "joint_action_payoff",
    "best_response",
]
