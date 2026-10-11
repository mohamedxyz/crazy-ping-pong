"""Q-learning update equations and epsilon-greedy action selection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from agent.q_learning import QLearningAgent
from config import AgentConfig


def make_agent(**overrides) -> QLearningAgent:
    config = AgentConfig(**{"learning_rate": 0.5, "discount": 0.9, **overrides})
    return QLearningAgent(3, 4, config, seed=0)


def test_q_table_shape_and_initial_zeros():
    agent = make_agent()
    assert agent.q_table.shape == (3, 4, 4)
    assert not agent.q_table.any()


def test_nonterminal_update_follows_formula():
    agent = make_agent()
    s, a, s_next = (1, 1), 1, (1, 2)
    agent.q_table[s][a] = 2.0
    agent.q_table[s_next] = [1.0, 5.0, -3.0, 0.5]
    update = agent.update_q_value(s, a, reward=-0.1, next_state=s_next, terminated=False)
    expected_target = -0.1 + 0.9 * 5.0
    expected_q = 2.0 + 0.5 * (expected_target - 2.0)
    assert update.max_next_q == pytest.approx(5.0)
    assert update.target == pytest.approx(expected_target)
    assert agent.q_table[s][a] == pytest.approx(expected_q)
    assert update.old_q == pytest.approx(2.0) and update.new_q == pytest.approx(expected_q)
    assert agent.visit_counts[s][a] == 1


def test_terminal_update_does_not_bootstrap():
    agent = make_agent()
    s, a, goal = (1, 1), 1, (1, 2)
    agent.q_table[goal] = [50.0, 50.0, 50.0, 50.0]   # must be ignored
    update = agent.update_q_value(s, a, reward=100.0, next_state=goal, terminated=True)
    assert update.max_next_q == 0.0
    assert update.target == pytest.approx(100.0)
    assert agent.q_table[s][a] == pytest.approx(0.5 * 100.0)


def test_wall_bump_lowers_action_value():
    agent = make_agent()
    s = (1, 1)
    values = []
    for _ in range(4):
        agent.update_q_value(s, 0, reward=-2.0, next_state=s, terminated=False)
        values.append(agent.q_table[s][0])
    assert all(b < a for a, b in zip(values, values[1:]))
    assert values[-1] < 0


def test_greedy_selection_exploits_best_action():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    agent.q_table[1, 1] = [0.0, 3.0, -1.0, 2.0]
    assert {agent.choose_action((1, 1)) for _ in range(50)} == {1}


def test_epsilon_one_explores_uniformly():
    agent = make_agent(epsilon_start=1.0)
    agent.q_table[1, 1] = [0.0, 3.0, -1.0, 2.0]
    counts = Counter(agent.choose_action((1, 1)) for _ in range(4000))
    assert set(counts) == {0, 1, 2, 3}
    assert all(800 < c < 1200 for c in counts.values())


def test_intermediate_epsilon_mixes_explore_and_exploit():
    agent = make_agent(epsilon_start=0.2, epsilon_min=0.0)
    agent.q_table[1, 1] = [0.0, 3.0, -1.0, 2.0]
    counts = Counter(agent.choose_action((1, 1)) for _ in range(5000))
    # best action chosen with prob 0.8 + 0.2/4 = 0.85
    assert counts[1] / 5000 == pytest.approx(0.85, abs=0.03)


def test_ties_are_broken_randomly_among_best_only():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    agent.q_table[1, 1] = [1.0, 1.0, -5.0, 1.0]
    counts = Counter(agent.choose_action((1, 1)) for _ in range(3000))
    assert set(counts) == {0, 1, 3}
    assert all(800 < counts[a] < 1200 for a in (0, 1, 3))


def test_disabling_exploration_forces_greedy():
    agent = make_agent(epsilon_start=1.0)
    agent.q_table[1, 1] = [0.0, 3.0, -1.0, 2.0]
    agent.exploration_enabled = False
    assert {agent.choose_action((1, 1)) for _ in range(50)} == {1}


def test_epsilon_decays_to_floor():
    agent = make_agent(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_reset_learning_clears_table():
    agent = make_agent()
    agent.update_q_value((1, 1), 0, -1.0, (1, 1), False)
    agent.decay_epsilon()
    agent.reset_learning()
    assert not agent.q_table.any() and not agent.visit_counts.any()
    assert agent.epsilon == agent.config.epsilon_start


def test_same_seed_same_choices():
    a, b = make_agent(epsilon_start=0.5), make_agent(epsilon_start=0.5)
    for agent in (a, b):
        agent.q_table[1, 1] = [0.0, 1.0, 0.0, 0.0]
    assert [a.choose_action((1, 1)) for _ in range(100)] == \
        [b.choose_action((1, 1)) for _ in range(100)]


@pytest.mark.parametrize("bad", [
    {"learning_rate": 0.0}, {"learning_rate": 1.5}, {"discount": -0.1},
    {"epsilon_min": 0.5, "epsilon_start": 0.1}, {"epsilon_decay": 0.0},
    {"decay_mode": "sometimes"},
])
def test_invalid_agent_config_rejected(bad):
    from config import ConfigError
    with pytest.raises(ConfigError):
        QLearningAgent(2, 2, AgentConfig(**bad))


def test_agent_has_no_access_to_maze():
    agent = make_agent()
    assert not any(isinstance(v, np.ndarray) and v.dtype == object for v in vars(agent).values())
    assert not hasattr(agent, "maze") and not hasattr(agent, "env")
