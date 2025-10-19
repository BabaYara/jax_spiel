import jax
import jax.numpy as jnp
import pytest

from jax_spiel import tensor_game


def test_tensor_state_initial_properties():
    game = tensor_game.matching_pennies()
    state = tensor_game.TensorState(game=game)

    assert not state.is_terminal
    assert state.current_player == tensor_game.SIMULTANEOUS_PLAYER

    legal_actions_player0 = state.legal_actions(player=0)
    legal_actions_player1 = state.legal_actions(player=1)

    assert jnp.array_equal(legal_actions_player0, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.array_equal(legal_actions_player1, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(state.returns(), jnp.zeros(2))


def test_tensor_state_apply_joint_action_and_returns():
    game = tensor_game.matching_pennies()
    state = tensor_game.TensorState(game=game)

    next_state = state.apply_joint_action((0, 1))

    assert next_state.is_terminal
    assert next_state.current_player == tensor_game.TERMINAL_PLAYER
    assert jnp.array_equal(next_state.returns(), jnp.array([1.0, -1.0]))

    with pytest.raises(ValueError):
        next_state.apply_joint_action((0, 0))


def test_tensor_state_rejects_illegal_actions():
    game = tensor_game.matching_pennies()
    state = tensor_game.TensorState(game=game)

    with pytest.raises(ValueError):
        state.apply_joint_action((0, 5))


def test_joint_action_payoff_is_jittable():
    game = tensor_game.matching_pennies()
    joint_action = jnp.array([1, 0], dtype=jnp.int32)

    jit_fn = jax.jit(lambda a: tensor_game.joint_action_payoff(game, a))
    payoff = jit_fn(joint_action)

    assert jnp.array_equal(payoff, jnp.array([1.0, -1.0]))


def test_matching_pennies_expected_payoff_zero_sum():
    game = tensor_game.matching_pennies()
    uniform = jnp.array([0.5, 0.5])

    payoff = tensor_game.expected_payoff(game, uniform, uniform)

    assert payoff.shape == (2,)
    assert jnp.allclose(payoff, jnp.array([0.0, 0.0]))


def test_expected_payoff_is_jittable():
    game = tensor_game.matching_pennies()
    policies = jnp.array([[0.2, 0.8], [0.7, 0.3]])

    jit_fn = jax.jit(lambda p: tensor_game.expected_payoff(game, p[0], p[1]))
    payoff = jit_fn(policies)

    assert jnp.allclose(payoff, tensor_game.expected_payoff(game, policies[0], policies[1]))


def test_best_response_selects_maximizing_action():
    game = tensor_game.matching_pennies()
    opponent_policy = jnp.array([0.9, 0.1])

    response = tensor_game.best_response(game, opponent_policy, player=0)

    assert response.shape == opponent_policy.shape
    assert jnp.array_equal(response, jnp.array([0.0, 1.0]))


def test_best_response_is_pure_strategy():
    game = tensor_game.matching_pennies()
    opponent_policy = jnp.array([0.5, 0.5])

    response = tensor_game.best_response(game, opponent_policy, player=1)

    assert jnp.isclose(response.sum(), 1.0)
    assert jnp.array_equal(response, jax.nn.one_hot(jnp.argmax(response), response.shape[0]))


def test_tensor_game_metadata():
    game = tensor_game.matching_pennies()

    assert game.num_players == 2
    assert game.num_actions == (2, 2)
    assert game.name == "matching_pennies"

    with pytest.raises(ValueError):
        tensor_game.expected_payoff(game, jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0]))
