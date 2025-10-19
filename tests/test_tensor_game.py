import jax
import jax.numpy as jnp
import pytest

from jax_spiel import tensor_game


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
