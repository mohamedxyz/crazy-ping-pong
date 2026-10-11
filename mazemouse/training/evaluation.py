"""Evaluation of a learned policy, kept strictly separate from training.

Evaluation never updates the Q-table, never decays epsilon, and never adds
episodes to the training metrics. The breadth-first-search shortest path is
computed here only as an independent benchmark to compare against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from agent.q_learning import QLearningAgent
from environment.maze import MazeEnv, State, shortest_path_length


@dataclass
class EvaluationReport:
    """Summary of repeated greedy evaluation episodes."""

    episodes: int
    successes: int
    success_rate: float
    failure_rate: float
    avg_reward: float
    avg_steps: float
    avg_success_steps: float | None
    shortest_path_length: int | None
    gap_to_optimal: float | None       # avg successful steps - BFS shortest length
    episode_steps: list[int] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    example_path: list[State] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        lines = [
            f"Evaluation episodes        : {self.episodes}",
            f"Success rate               : {100 * self.success_rate:.1f}%",
            f"Failure rate               : {100 * self.failure_rate:.1f}%",
            f"Average reward             : {self.avg_reward:.2f}",
            f"Average steps (all)        : {self.avg_steps:.1f}",
            "Average steps (successful) : "
            + ("n/a" if self.avg_success_steps is None else f"{self.avg_success_steps:.1f}"),
            "Shortest path (BFS)        : "
            + ("unreachable" if self.shortest_path_length is None
               else str(self.shortest_path_length)),
            "Gap to shortest path       : "
            + ("n/a" if self.gap_to_optimal is None else f"{self.gap_to_optimal:+.1f} steps"),
        ]
        return "\n".join(lines)


def evaluate_agent(agent: QLearningAgent, env: MazeEnv, episodes: int = 20,
                   seed: int | None = 12345) -> EvaluationReport:
    """Run ``episodes`` greedy episodes without learning.

    A dedicated RNG is used for tie-breaking so evaluation neither changes
    the Q-table nor perturbs the agent's own random stream.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    rng = np.random.default_rng(seed)
    steps_list: list[int] = []
    rewards: list[float] = []
    success_steps: list[int] = []
    example_path: list[State] = []
    for i in range(episodes):
        state = env.reset()
        path = [state]
        total = 0.0
        while True:
            action = agent.greedy_action(state, rng=rng)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if not info.hit_wall:
                path.append(state)
            if terminated or truncated:
                break
        steps_list.append(env.steps)
        rewards.append(total)
        if info.reached_goal:
            success_steps.append(env.steps)
        if i == 0:
            example_path = path
    optimal = shortest_path_length(env.maze)
    avg_success = float(np.mean(success_steps)) if success_steps else None
    return EvaluationReport(
        episodes=episodes,
        successes=len(success_steps),
        success_rate=len(success_steps) / episodes,
        failure_rate=1 - len(success_steps) / episodes,
        avg_reward=float(np.mean(rewards)),
        avg_steps=float(np.mean(steps_list)),
        avg_success_steps=avg_success,
        shortest_path_length=optimal,
        gap_to_optimal=(None if avg_success is None or optimal is None
                        else avg_success - optimal),
        episode_steps=steps_list,
        episode_rewards=rewards,
        example_path=example_path,
    )
