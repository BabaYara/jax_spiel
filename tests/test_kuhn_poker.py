import jax
import jax.numpy as jnp
import pytest

from jax_spiel import kuhn_poker


def test_kuhn_poker_fold_sequence_returns_correct_utilities():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(2, 0))

    assert not state.is_terminal
    assert state.current_player == 0
    assert jnp.array_equal(state.legal_actions(0), jnp.array([0, 1], dtype=jnp.int32))

    state_after_bet = state.apply_action(1)
    assert not state_after_bet.is_terminal
    assert state_after_bet.current_player == 1
    assert jnp.array_equal(
        state_after_bet.legal_actions(1), jnp.array([0, 1], dtype=jnp.int32)
    )

    terminal_state = state_after_bet.apply_action(0)
    assert terminal_state.is_terminal
    assert terminal_state.current_player == kuhn_poker.TERMINAL_PLAYER
    assert jnp.allclose(terminal_state.returns(), jnp.array([1.0, -1.0]))


def test_kuhn_poker_double_check_leads_to_showdown():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(0, 2))

    state = state.apply_action(0)
    terminal_state = state.apply_action(0)

    assert terminal_state.is_terminal
    assert jnp.allclose(terminal_state.returns(), jnp.array([-1.0, 1.0]))


def test_kuhn_poker_call_after_bet_accumulates_pot_correctly():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(2, 1))

    state = state.apply_action(1)  # player 0 bets
    terminal_state = state.apply_action(1)  # player 1 calls

    assert terminal_state.is_terminal
    assert jnp.allclose(terminal_state.returns(), jnp.array([2.0, -2.0]))


def test_kuhn_poker_bet_from_second_player_can_be_called():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(1, 2))

    state = state.apply_action(0)  # player 0 checks
    state = state.apply_action(1)  # player 1 bets
    terminal_state = state.apply_action(1)  # player 0 calls

    assert terminal_state.is_terminal
    assert jnp.allclose(terminal_state.returns(), jnp.array([-2.0, 2.0]))


def test_kuhn_poker_rejects_illegal_actions():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(0, 1))

    with pytest.raises(ValueError):
        state.apply_action(2)

    state_after_bet = state.apply_action(1)
    terminal_state = state_after_bet.apply_action(0)

    with pytest.raises(ValueError):
        terminal_state.apply_action(0)


def test_kuhn_poker_random_initial_state_samples_unique_cards():
    game = kuhn_poker.KuhnPokerGame()
    rng = jax.random.PRNGKey(0)

    state = game.new_initial_state(rng_key=rng)

    assert sorted(state.player_cards) in ([0, 1], [0, 2], [1, 2])
    assert state.player_cards[0] != state.player_cards[1]


def test_kuhn_poker_information_state_tensor_encodes_player_and_card():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(2, 0))

    tensor0 = state.information_state_tensor(0)
    tensor1 = state.information_state_tensor(1)

    expected0 = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    expected1 = jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)

    assert tensor0.shape == (11,)
    assert tensor1.shape == (11,)
    assert jnp.array_equal(tensor0, expected0)
    assert jnp.array_equal(tensor1, expected1)
    assert jnp.array_equal(state.observation_tensor(0), tensor0)


def test_kuhn_poker_information_state_tensor_reflects_history():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(1, 2))

    state = state.apply_action(1)
    terminal_state = state.apply_action(0)

    tensor = terminal_state.information_state_tensor(0)

    expected = jnp.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=jnp.float32)

    assert jnp.array_equal(tensor, expected)


def test_kuhn_history_tensor_utilities_validate_and_jit():
    with pytest.raises(ValueError):
        kuhn_poker.history_tensor_from_actions((0, 1, 0, 1))

    with pytest.raises(ValueError):
        kuhn_poker.history_tensor_from_actions((0, 2))

    history = jnp.array([1, 0], dtype=jnp.int32)
    expected = jnp.array([0, 1, 1, 0, 0, 0], dtype=jnp.float32)

    jit_fn = jax.jit(kuhn_poker.kuhn_history_to_tensor)
    assert jnp.array_equal(jit_fn(history), expected)


def test_information_state_tensor_rejects_invalid_player():
    game = kuhn_poker.KuhnPokerGame()
    state = game.new_initial_state(cards=(0, 1))

    with pytest.raises(ValueError):
        state.information_state_tensor(2)

