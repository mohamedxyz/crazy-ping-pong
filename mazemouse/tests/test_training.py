"""Training loop, statistics, evaluation and convergence."""

from __future__ import annotations

import numpy as np
import pytest

from agent.q_learning import QLearningAgent
from config import AgentConfig
from environment.maze import Maze, shortest_path_length
from training.evaluation import evaluate_agent
from training.metrics import rolling_mean
from training.trainer import POLICY_GREEDY, Trainer
from tests.conftest import fast_agent_config, make_env


def make_trainer(maze: Maze, seed: int = 0, config: AgentConfig | None = None,
                 max_steps: int = 300) -> Trainer:
    env = make_env(maze, max_steps=max_steps)
    agent = QLearningAgent(maze.rows, maze.cols, config or fast_agent_config(), seed=seed)
    return Trainer(env, agent)


def test_episode_statistics_are_valid(small_maze):
    trainer = make_trainer(small_maze, max_steps=50)
    stats = trainer.train(30)
    assert [s.episode for s in stats] == list(range(1, 31))
    for s in stats:
        assert 1 <= s.steps <= 50
        assert s.success == (s.reason == "goal")
        assert s.reason in ("goal", "step_limit")
        assert 0 <= s.wall_collisions <= s.steps
        assert 1 <= s.unique_cells <= len(small_maze.traversable_cells())
        assert sum(s.reward_components.values()) == pytest.approx(s.total_reward)
        assert s.duration_s >= 0
        assert 0 <= s.epsilon <= 1
    assert len(trainer.metrics) == 30
    assert trainer.episode_count == 30


def test_epsilon_recorded_then_decayed_per_episode(small_maze):
    trainer = make_trainer(small_maze)
    first = trainer.train_episode()
    assert first.epsilon == pytest.approx(1.0)
    assert trainer.agent.epsilon == pytest.approx(0.99)


def test_training_changes_q_table(small_maze):
    trainer = make_trainer(small_maze)
    before = trainer.agent.q_table.copy()
    trainer.train(5)
    assert not np.array_equal(before, trainer.agent.q_table)
    assert trainer.agent.total_updates == sum(s.steps for s in trainer.metrics.episodes)


def test_step_by_step_matches_whole_episode(small_maze):
    a = make_trainer(small_maze, seed=3)
    b = make_trainer(small_maze, seed=3)
    a.train(10)
    for _ in range(10):
        while True:
            _, stats = b.step()
            if stats is not None:
                break
    assert np.array_equal(a.agent.q_table, b.agent.q_table)
    assert a.metrics.to_list()[-1]["steps"] == b.metrics.to_list()[-1]["steps"]


def test_aborted_episode_is_not_recorded(small_maze):
    trainer = make_trainer(small_maze)
    trainer.begin_episode()
    trainer.step()
    trainer.abort_episode()
    assert len(trainer.metrics) == 0 and trainer.recorder is None


def test_non_learning_runs_do_not_touch_q_or_metrics(small_maze):
    trainer = make_trainer(small_maze)
    trainer.train(20)
    q = trainer.agent.q_table.copy()
    eps = trainer.agent.epsilon
    trainer.run_episode(POLICY_GREEDY, learn=False)
    assert np.array_equal(q, trainer.agent.q_table)
    assert trainer.agent.epsilon == eps and len(trainer.metrics) == 20


def test_evaluation_does_not_modify_agent(small_maze):
    trainer = make_trainer(small_maze)
    trainer.train(100)
    agent = trainer.agent
    q, visits, eps = agent.q_table.copy(), agent.visit_counts.copy(), agent.epsilon
    rng_state = agent.rng.bit_generator.state
    n_metrics = len(trainer.metrics)
    report = evaluate_agent(agent, make_env(small_maze), episodes=10)
    assert np.array_equal(q, agent.q_table) and np.array_equal(visits, agent.visit_counts)
    assert agent.epsilon == eps
    assert agent.rng.bit_generator.state == rng_state
    assert len(trainer.metrics) == n_metrics
    assert report.episodes == 10 and len(report.episode_steps) == 10


def test_untrained_agent_fails_evaluation(small_maze):
    agent = QLearningAgent(small_maze.rows, small_maze.cols, fast_agent_config(), seed=0)
    # Zero Q-table + random tie-breaking = random walk; with a tight step
    # limit it practically never reaches the goal.
    report = evaluate_agent(agent, make_env(small_maze, max_steps=7), episodes=30)
    assert report.success_rate < 0.5


def test_agent_learns_small_maze(small_maze):
    trainer = make_trainer(small_maze, seed=1, max_steps=100)
    trainer.train(400)
    report = evaluate_agent(trainer.agent, make_env(small_maze, max_steps=100), episodes=20)
    assert report.success_rate == 1.0
    assert report.avg_success_steps == shortest_path_length(small_maze)


def test_agent_learns_default_maze_with_default_parameters():
    """Integration test: the full default setup converges to the optimal route."""
    maze = Maze.default()
    trainer = make_trainer(maze, seed=42, config=AgentConfig())
    trainer.train(3000)
    recent = trainer.metrics.summary(recent=200)
    assert recent["recent_success_rate"] > 0.95
    report = evaluate_agent(trainer.agent, make_env(maze), episodes=20)
    assert report.success_rate == 1.0
    assert report.gap_to_optimal is not None and report.gap_to_optimal <= 4
    # Performance improved: late episodes are far shorter than early ones.
    steps = trainer.metrics.series("steps")
    assert steps[-200:].mean() < 0.5 * steps[:200].mean()


def test_training_is_reproducible_with_seed(small_maze):
    a = make_trainer(small_maze, seed=11)
    b = make_trainer(small_maze, seed=11)
    a.train(50)
    b.train(50)
    assert np.array_equal(a.agent.q_table, b.agent.q_table)

    def strip(trainer: Trainer) -> list[dict]:
        return [{k: v for k, v in e.items() if k != "duration_s"}
                for e in trainer.metrics.to_list()]

    assert strip(a) == strip(b)


def test_best_path_is_a_real_successful_route(small_maze):
    trainer = make_trainer(small_maze, seed=2)
    trainer.train(200)
    path = trainer.best_path
    assert path is not None and path[0] == small_maze.start and path[-1] == small_maze.goal
    assert len(path) - 1 == trainer.best_length


def test_rolling_mean():
    assert rolling_mean([1, 2, 3, 4], 2).tolist() == [1.0, 1.5, 2.5, 3.5]
    assert rolling_mean([], 5).size == 0
