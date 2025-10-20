"""Functional JAX implementation of Kuhn Poker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp

from .tensor_game import TERMINAL_PLAYER


_DECK = (0, 1, 2)
_MAX_HISTORY_LENGTH = 3
_NUM_ACTIONS = 2


@dataclass(frozen=True)
class KuhnPokerGame:
    """Two-player Kuhn Poker with immutable state transitions."""

    ante: float = 1.0
    bet_size: float = 1.0
    name: str = "kuhn_poker"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def deck(self) -> Tuple[int, int, int]:
        return _DECK

    def new_initial_state(
        self,
        *,
        cards: Tuple[int, int] | None = None,
        rng_key: jax.Array | None = None,
    ) -> "KuhnPokerState":
        """Returns an initial state with dealt private cards."""

        if cards is not None and rng_key is not None:
            raise ValueError("Provide either cards or rng_key, not both")

        if cards is None:
            if rng_key is None:
                raise ValueError("Either cards or rng_key must be supplied")
            deck = jnp.array(self.deck, dtype=jnp.int32)
            permuted = jax.random.permutation(rng_key, deck)
            cards = (int(permuted[0]), int(permuted[1]))
        else:
            if len(cards) != 2:
                raise ValueError("cards must contain two entries")
            if cards[0] == cards[1]:
                raise ValueError("cards must be distinct")
            if any(card not in self.deck for card in cards):
                raise ValueError("cards must be drawn from the Kuhn deck")

        return KuhnPokerState(game=self, player_cards=(int(cards[0]), int(cards[1])))


@dataclass(frozen=True)
class KuhnPokerState:
    """State representation for Kuhn Poker."""

    game: KuhnPokerGame
    player_cards: Tuple[int, int]
    history: Tuple[int, ...] = ()

    @property
    def is_terminal(self) -> bool:
        history = self.history
        if len(history) == 2:
            return history in ((0, 0), (1, 0), (1, 1))
        if len(history) == 3:
            return history in ((0, 1, 0), (0, 1, 1))
        return False

    @property
    def current_player(self) -> int:
        if self.is_terminal:
            return TERMINAL_PLAYER
        return len(self.history) % 2

    def legal_actions(self, player: int) -> jnp.ndarray:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if self.is_terminal or player != self.current_player:
            return jnp.array([], dtype=jnp.int32)
        return jnp.array([0, 1], dtype=jnp.int32)

    def apply_action(self, action: int) -> "KuhnPokerState":
        if action not in (0, 1):
            raise ValueError("action must be 0 (pass/fold) or 1 (bet/call)")
        if self.is_terminal:
            raise ValueError("cannot act on a terminal state")

        player = self.current_player
        legal = tuple(int(a) for a in self.legal_actions(player))
        if action not in legal:
            raise ValueError("action is not legal for the current player")

        return KuhnPokerState(
            game=self.game,
            player_cards=self.player_cards,
            history=self.history + (int(action),),
        )

    def returns(self) -> jnp.ndarray:
        dtype = jnp.result_type(self.game.ante, self.game.bet_size, jnp.float32)
        if not self.is_terminal:
            return jnp.zeros(self.game.num_players, dtype=dtype)

        contributions = self._contributions(dtype)
        pot = jnp.sum(contributions)
        returns = -contributions

        if self._is_fold_terminal():
            winner = self._last_bet_player()
        else:
            winner = 0 if self.player_cards[0] > self.player_cards[1] else 1

        returns = returns.at[winner].add(pot)
        return returns

    def information_state_tensor(
        self, player: int, *, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")

        player_encoding = jax.nn.one_hot(player, self.game.num_players, dtype=dtype)
        card = self.player_cards[player]
        card_encoding = jax.nn.one_hot(card, len(self.game.deck), dtype=dtype)
        history_encoding = history_tensor_from_actions(self.history, dtype=dtype)

        return jnp.concatenate((player_encoding, card_encoding, history_encoding))

    def observation_tensor(
        self, player: int, *, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        return self.information_state_tensor(player, dtype=dtype)

    def _contributions(self, dtype) -> jnp.ndarray:
        contributions = [self.game.ante, self.game.ante]
        bet_size = self.game.bet_size

        for idx, action in enumerate(self.history):
            if action == 1:
                contributions[idx % 2] += bet_size

        return jnp.array(contributions, dtype=dtype)

    def _is_fold_terminal(self) -> bool:
        if not self.is_terminal:
            return False
        history = self.history
        if history[-1] != 0:
            return False
        return len(history) >= 2 and history[-2] == 1

    def _last_bet_player(self) -> int:
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx] == 1:
                return idx % 2
        raise ValueError("Fold terminal state must contain a bet")


def history_tensor_from_actions(
    history: Iterable[int], *, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    history_tuple = tuple(int(action) for action in history)
    if len(history_tuple) > _MAX_HISTORY_LENGTH:
        raise ValueError("history length exceeds maximum length for Kuhn Poker")
    if any(action not in (0, 1) for action in history_tuple):
        raise ValueError("history contains invalid actions")

    history_array = jnp.array(history_tuple, dtype=jnp.int32)
    return kuhn_history_to_tensor(history_array, dtype=dtype)


def kuhn_history_to_tensor(
    history: jnp.ndarray, *, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    history = jnp.asarray(history, dtype=jnp.int32)

    base = jnp.zeros((_MAX_HISTORY_LENGTH, _NUM_ACTIONS), dtype=dtype)
    length = history.shape[0]

    def _update(base_tensor):
        indices = jnp.arange(length, dtype=jnp.int32)
        updates = jax.nn.one_hot(history, _NUM_ACTIONS, dtype=dtype)
        return base_tensor.at[indices].set(updates)

    base = jax.lax.cond(length == 0, lambda t: t, _update, base)
    return base.reshape(-1)


__all__ = [
    "KuhnPokerGame",
    "KuhnPokerState",
    "history_tensor_from_actions",
    "kuhn_history_to_tensor",
]

