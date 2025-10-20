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


def test_batched_expected_payoff_matches_scalar_path():
    game = tensor_game.matching_pennies()
    policies = jnp.array(
        [
            [[0.2, 0.8], [0.4, 0.6]],
            [[0.5, 0.5], [0.9, 0.1]],
        ]
    )

    batched = tensor_game.batched_expected_payoff(game, policies[:, 0], policies[:, 1])

    manual = jnp.stack(
        [
            tensor_game.expected_payoff(game, policies[i, 0], policies[i, 1])
            for i in range(policies.shape[0])
        ],
        axis=0,
    )

    assert batched.shape == manual.shape
    assert jnp.allclose(batched, manual)


def test_batched_expected_payoff_supports_vmap():
    game = tensor_game.matching_pennies()
    policies = jnp.array(
        [
            [[0.5, 0.5], [0.3, 0.7]],
            [[0.1, 0.9], [0.6, 0.4]],
            [[0.8, 0.2], [0.2, 0.8]],
        ]
    )

    vmapped = jax.vmap(
        lambda pair: tensor_game.expected_payoff(game, pair[0], pair[1])
    )(policies)

    batched = tensor_game.batched_expected_payoff(game, policies[:, 0], policies[:, 1])

    assert jnp.allclose(vmapped, batched)


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


def test_nash_conv_is_zero_for_matching_pennies_equilibrium():
    game = tensor_game.matching_pennies()
    policies = jnp.array([[0.5, 0.5], [0.5, 0.5]])

    conv = tensor_game.nash_conv(game, policies)

    assert jnp.isclose(conv, 0.0)


def test_nash_conv_detects_exploitable_profile():
    game = tensor_game.matching_pennies()
    policies = jnp.array([[1.0, 0.0], [1.0, 0.0]])

    conv = tensor_game.nash_conv(game, policies)

    assert jnp.isclose(conv, 2.0)


def test_nash_conv_is_jittable():
    game = tensor_game.matching_pennies()
    policies = jnp.array([[0.2, 0.8], [0.7, 0.3]])

    jit_fn = jax.jit(lambda p: tensor_game.nash_conv(game, p))
    conv = jit_fn(policies)

    assert jnp.isclose(conv, tensor_game.nash_conv(game, policies))


def test_tensor_game_initial_state_factory():
    game = tensor_game.matching_pennies()
    state = game.new_initial_state()

    assert isinstance(state, tensor_game.TensorState)
    assert state.game is game
    assert not state.is_terminal


def test_tensor_game_metadata():
    game = tensor_game.matching_pennies()

    assert game.num_players == 2
    assert game.num_actions == (2, 2)
    assert game.name == "matching_pennies"

    with pytest.raises(ValueError):
        tensor_game.expected_payoff(game, jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0]))


def test_tensor_game_from_payoff_matrices_constructs_tensor():
    player0 = jnp.array([[3.0, 0.0], [5.0, 1.0]])
    player1 = jnp.array([[3.0, 5.0], [0.0, 1.0]])

    game = tensor_game.TensorGame.from_payoff_matrices(
        player0_payoffs=player0,
        player1_payoffs=player1,
        name="prisoners_dilemma",
    )

    assert game.name == "prisoners_dilemma"
    assert game.payoffs.shape == (2, 2, 2)
    assert jnp.array_equal(game.payoffs[..., 0], player0)
    assert jnp.array_equal(game.payoffs[..., 1], player1)


def test_tensor_game_from_payoff_matrices_validates_inputs():
    player0 = jnp.array([[1.0, -1.0]])
    player1 = jnp.array([[1.0], [-1.0]])

    with pytest.raises(ValueError):
        tensor_game.TensorGame.from_payoff_matrices(player0, player1)

    with pytest.raises(ValueError):
        tensor_game.TensorGame.from_payoff_matrices(jnp.array([1.0, -1.0]), jnp.array([1.0, -1.0]))


def test_rock_paper_scissors_uniform_strategy_is_equilibrium():
    game = tensor_game.rock_paper_scissors()
    uniform = jnp.array([1 / 3, 1 / 3, 1 / 3])

    payoff = tensor_game.expected_payoff(game, uniform, uniform)

    assert game.num_actions == (3, 3)
    assert game.name == "rock_paper_scissors"
    assert jnp.allclose(payoff, jnp.array([0.0, 0.0]))

    conv = tensor_game.nash_conv(game, (uniform, uniform))
    assert jnp.isclose(conv, 0.0)


def test_sample_joint_action_is_jittable():
    game = tensor_game.matching_pennies()
    policy = jnp.array([0.5, 0.5])

    @jax.jit
    def sample(key):
        return tensor_game.sample_joint_action(game, key, policy, policy)

    key = jax.random.key(0)
    joint_action = sample(key)

    assert joint_action.shape == (2,)
    assert joint_action.dtype == jnp.int32
