"""Tabular Q-learning agent (the mouse).

The agent only ever sees what the environment returns: its state (row,
column), the reward of each transition and whether the episode ended. It
has no access to the maze layout, the goal location or any shortest path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import AgentConfig

State = tuple[int, int]
NUM_ACTIONS = 4


@dataclass(frozen=True)
class QUpdate:
    """Every number involved in one Q-learning update (for the explanation panel)."""

    state: State
    action: int
    old_q: float
    reward: float
    max_next_q: float     # 0.0 for terminal transitions (no bootstrapping)
    alpha: float
    gamma: float
    target: float
    new_q: float
    terminated: bool


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning over a (rows, cols, 4) Q-table.

    ``q_table[r, c, a]`` estimates the discounted return of taking action
    ``a`` in cell ``(r, c)`` and acting greedily afterwards. Wall cells get
    rows in the table too (simple indexing) but are never visited.
    """

    def __init__(self, rows: int, cols: int, config: AgentConfig | None = None,
                 seed: int | None = None) -> None:
        self.config = config or AgentConfig()
        self.config.validate()
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((rows, cols, NUM_ACTIONS), dtype=np.float64)
        self.visit_counts = np.zeros((rows, cols, NUM_ACTIONS), dtype=np.int64)
        self.epsilon = self.config.epsilon_start
        self.exploration_enabled = True
        self.total_updates = 0

    # ---------------------------------------------------------------- policy
    def action_values(self, state: State) -> np.ndarray:
        return self.q_table[state[0], state[1]]

    def greedy_action(self, state: State, rng: np.random.Generator | None = None) -> int:
        """Return an action with the highest Q-value, breaking ties at random."""
        values = self.action_values(state)
        best = np.flatnonzero(values == values.max())
        generator = rng if rng is not None else self.rng
        return int(best[generator.integers(len(best))])

    def choose_action(self, state: State) -> int:
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
        if self.exploration_enabled and self.rng.random() < self.epsilon:
            return int(self.rng.integers(NUM_ACTIONS))
        return self.greedy_action(state)

    # -------------------------------------------------------------- learning
    def update_q_value(self, state: State, action: int, reward: float,
                       next_state: State, terminated: bool) -> QUpdate:
        """Apply Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).

        For terminal transitions the bootstrap term is dropped because the
        value of a terminal state is zero. Truncated (time-limit) transitions
        are *not* terminal and still bootstrap.
        """
        alpha = self.config.learning_rate
        gamma = self.config.discount
        old_q = float(self.q_table[state[0], state[1], action])
        max_next_q = 0.0 if terminated else float(self.action_values(next_state).max())
        target = reward + gamma * max_next_q
        new_q = old_q + alpha * (target - old_q)
        self.q_table[state[0], state[1], action] = new_q
        self.visit_counts[state[0], state[1], action] += 1
        self.total_updates += 1
        return QUpdate(state, action, old_q, reward, max_next_q, alpha, gamma,
                       target, new_q, terminated)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def reset_learning(self) -> None:
        """Forget everything: zero Q-table, zero counts, epsilon back to start."""
        self.q_table.fill(0.0)
        self.visit_counts.fill(0)
        self.epsilon = self.config.epsilon_start
        self.total_updates = 0
        self.rng = np.random.default_rng(self.seed)
