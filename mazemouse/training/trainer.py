"""Episode loop connecting the environment and the Q-learning agent.

The trainer can run a whole episode at once (fast, headless) or advance one
transition at a time so the GUI can animate the mouse without blocking.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from agent.q_learning import QLearningAgent, QUpdate
from environment.maze import MazeEnv, State, StepInfo
from training.metrics import EpisodeStats, MetricsTracker

logger = logging.getLogger(__name__)

POLICY_EPSILON = "epsilon"   # epsilon-greedy (training behaviour)
POLICY_GREEDY = "greedy"     # always exploit (random tie-breaking)
POLICY_RANDOM = "random"     # uniform random actions (untrained baseline)


@dataclass
class Transition:
    """One environment step as seen by the trainer."""

    state: State
    action: int
    reward: float
    next_state: State
    terminated: bool
    truncated: bool
    info: StepInfo
    update: QUpdate | None


@dataclass
class EpisodeRecorder:
    """Accumulates per-episode counters while an episode is running."""

    start_time: float = field(default_factory=time.perf_counter)
    total_reward: float = 0.0
    steps: int = 0
    wall_collisions: int = 0
    trap_encounters: int = 0
    path: list[State] = field(default_factory=list)
    visited: set[State] = field(default_factory=set)
    components: dict[str, float] = field(default_factory=dict)

    def record(self, transition: Transition) -> None:
        self.steps += 1
        self.total_reward += transition.reward
        self.wall_collisions += int(transition.info.hit_wall)
        self.trap_encounters += int(transition.info.hit_trap)
        for key, value in transition.info.reward_components.items():
            self.components[key] = self.components.get(key, 0.0) + value
        if not transition.info.hit_wall:
            self.path.append(transition.next_state)
        self.visited.add(transition.next_state)


class Trainer:
    """Runs episodes, applies Q-learning updates and records statistics."""

    def __init__(self, env: MazeEnv, agent: QLearningAgent,
                 metrics: MetricsTracker | None = None) -> None:
        self.env = env
        self.agent = agent
        self.metrics = metrics or MetricsTracker()
        self.episode_count = len(self.metrics)
        self.recorder: EpisodeRecorder | None = None
        self.state: State | None = None
        self.last_transition: Transition | None = None
        self.last_update: QUpdate | None = None
        self.best_path: list[State] | None = None
        self.last_episode_path: list[State] = []
        # Recent Q-value history of (state, action) pairs that hit walls,
        # used to show how repeated mistakes lower an action's value.
        self.wall_history: dict[tuple[State, int], deque[float]] = {}
        self.last_wall_key: tuple[State, int] | None = None
        self._restore_best_path_length()

    def _restore_best_path_length(self) -> None:
        self.best_length: int | None = self.metrics.summary()["best_success_steps"]  # type: ignore

    # ------------------------------------------------------------ episodes
    @property
    def episode_active(self) -> bool:
        return self.recorder is not None

    def begin_episode(self) -> State:
        """Reset the environment and start recording a new episode."""
        self.state = self.env.reset()
        self.recorder = EpisodeRecorder()
        self.recorder.path.append(self.state)
        self.recorder.visited.add(self.state)
        return self.state

    def abort_episode(self) -> None:
        """Discard the episode in progress (it is not added to the statistics)."""
        self.recorder = None
        self.state = self.env.reset()

    def _select_action(self, policy: str) -> int:
        assert self.state is not None
        if policy == POLICY_EPSILON:
            return self.agent.choose_action(self.state)
        if policy == POLICY_GREEDY:
            return self.agent.greedy_action(self.state)
        if policy == POLICY_RANDOM:
            return int(self.agent.rng.integers(4))
        raise ValueError(f"unknown policy {policy!r}")

    def step(self, learn: bool = True, policy: str = POLICY_EPSILON
             ) -> tuple[Transition, EpisodeStats | None]:
        """Advance the current episode by one transition.

        Returns the transition and, if the episode just finished, its
        :class:`EpisodeStats`. Finished *learning* episodes are added to the
        training metrics and decay epsilon; non-learning runs are not.
        """
        if self.recorder is None:
            self.begin_episode()
        assert self.recorder is not None and self.state is not None
        state = self.state
        action = self._select_action(policy)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        update = None
        if learn:
            update = self.agent.update_q_value(state, action, reward, next_state, terminated)
            self.last_update = update
            if self.agent.config.decay_mode == "step":
                self.agent.decay_epsilon()
            if info.hit_wall:
                key = (state, action)
                history = self.wall_history.setdefault(key, deque(maxlen=12))
                if not history:
                    history.append(update.old_q)
                history.append(update.new_q)
                self.last_wall_key = key
        transition = Transition(state, action, reward, next_state, terminated, truncated,
                                info, update)
        self.last_transition = transition
        self.recorder.record(transition)
        self.state = next_state
        if not (terminated or truncated):
            return transition, None
        return transition, self._finish_episode(transition, learn)

    def _finish_episode(self, transition: Transition, learn: bool) -> EpisodeStats:
        recorder = self.recorder
        assert recorder is not None
        success = transition.info.reached_goal
        stats = EpisodeStats(
            episode=self.episode_count + 1 if learn else 0,
            total_reward=recorder.total_reward,
            steps=recorder.steps,
            success=success,
            wall_collisions=recorder.wall_collisions,
            trap_encounters=recorder.trap_encounters,
            unique_cells=len(recorder.visited),
            epsilon=self.agent.epsilon,
            duration_s=time.perf_counter() - recorder.start_time,
            reason=transition.info.reason or "",
            reward_components=dict(recorder.components),
        )
        if learn:
            self.episode_count += 1
            self.metrics.add(stats)
            if success and (self.best_length is None or stats.steps < self.best_length):
                self.best_length = stats.steps
                self.best_path = list(recorder.path)
            if self.agent.config.decay_mode == "episode":
                self.agent.decay_epsilon()
        self.last_episode_path = list(recorder.path)
        self.recorder = None
        return stats

    def train_episode(self, on_step: Callable[[Transition], None] | None = None
                      ) -> EpisodeStats:
        """Run one complete learning episode and return its statistics."""
        self.begin_episode()
        while True:
            transition, stats = self.step(learn=True, policy=POLICY_EPSILON)
            if on_step is not None:
                on_step(transition)
            if stats is not None:
                return stats

    def run_episode(self, policy: str, learn: bool = False) -> EpisodeStats:
        """Run a complete episode with a given policy (used by demos)."""
        self.begin_episode()
        while True:
            _, stats = self.step(learn=learn, policy=policy)
            if stats is not None:
                return stats

    def train(self, num_episodes: int, log_every: int = 0,
              on_episode: Callable[[EpisodeStats], None] | None = None
              ) -> list[EpisodeStats]:
        """Train for ``num_episodes`` episodes headlessly."""
        results = []
        for _ in range(num_episodes):
            stats = self.train_episode()
            results.append(stats)
            if on_episode is not None:
                on_episode(stats)
            if log_every and stats.episode % log_every == 0:
                summary = self.metrics.summary(recent=log_every)
                logger.info("episode %5d | eps %.3f | recent success %5.1f%% | "
                            "recent avg steps %6.1f | best %s",
                            stats.episode, self.agent.epsilon,
                            100 * float(summary["recent_success_rate"] or 0.0),
                            float(summary["recent_avg_steps"] or 0.0),
                            summary["best_success_steps"])
        return results

    def reset_learning(self) -> None:
        """Forget the Q-table and every statistic."""
        self.agent.reset_learning()
        self.metrics.clear()
        self.episode_count = 0
        self.best_path = None
        self.best_length = None
        self.last_update = None
        self.last_transition = None
        self.wall_history.clear()
        self.last_wall_key = None
        self.abort_episode()
