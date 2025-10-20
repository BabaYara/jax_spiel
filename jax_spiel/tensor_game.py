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

    @classmethod
    def from_payoff_matrices(
        cls,
        player0_payoffs,
        player1_payoffs,
        *,
        name: str = "tensor_game",
    ) -> "TensorGame":
        """Constructs a ``TensorGame`` from separate payoff matrices.

        Args:
            player0_payoffs: Payoff matrix for player 0 with shape ``(A0, A1)``.
            player1_payoffs: Payoff matrix for player 1 with shape ``(A0, A1)``.
            name: Optional name for the resulting game.

        Returns:
            A ``TensorGame`` instance with the stacked payoff tensor.

        Raises:
            ValueError: If the payoff matrices are not rank-2 or have mismatched shapes.
        """

        player0_arr = jnp.asarray(player0_payoffs)
        player1_arr = jnp.asarray(player1_payoffs)

        if player0_arr.ndim != 2 or player1_arr.ndim != 2:
            raise ValueError("payoff matrices must both be rank-2 tensors")
        if player0_arr.shape != player1_arr.shape:
            raise ValueError(
                "player0_payoffs and player1_payoffs must have matching shapes"
            )

        payoffs = jnp.stack((player0_arr, player1_arr), axis=-1)
        return cls(name=name, payoffs=payoffs)

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


def rock_paper_scissors() -> TensorGame:
    """Creates the Rock-Paper-Scissors zero-sum game."""

    player0_payoffs = jnp.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    player1_payoffs = -player0_payoffs
    return TensorGame.from_payoff_matrices(
        player0_payoffs=player0_payoffs,
        player1_payoffs=player1_payoffs,
        name="rock_paper_scissors",
    )


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


def nash_conv(game: TensorGame, joint_policy: Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray) -> jnp.ndarray:
    """Computes the NashConv exploitability metric for two-player games."""

    try:
        player0_policy, player1_policy = joint_policy
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("joint_policy must be an iterable of length 2") from exc

    player0_policy = jnp.asarray(player0_policy)
    player1_policy = jnp.asarray(player1_policy)

    num_actions0, num_actions1 = game.num_actions
    _validate_policy(player0_policy, num_actions0, "player0_policy")
    _validate_policy(player1_policy, num_actions1, "player1_policy")

    expected = expected_payoff(game, player0_policy, player1_policy)

    br0 = best_response(game, player1_policy, player=0)
    br1 = best_response(game, player0_policy, player=1)

    br0_value = expected_payoff(game, br0, player1_policy)[0]
    br1_value = expected_payoff(game, player0_policy, br1)[1]

    return (br0_value - expected[0]) + (br1_value - expected[1])


__all__ = [
    "SIMULTANEOUS_PLAYER",
    "TERMINAL_PLAYER",
    "TensorGame",
    "TensorState",
    "matching_pennies",
    "rock_paper_scissors",
    "expected_payoff",
    "joint_action_payoff",
    "best_response",
    "nash_conv",
]
