"""Wiring helpers: build the maze, environment, agent and trainer from config.

Both the command-line interface and the GUI use :class:`Session`, so there is
exactly one place that decides how the pieces fit together.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.persistence import load_model, save_model
from agent.q_learning import QLearningAgent
from config import AppConfig
from environment.generator import generate_maze
from environment.maze import Maze, MazeEnv
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def build_maze(config: AppConfig) -> Maze:
    """Create the maze selected by ``config.environment.maze``."""
    env_cfg = config.environment
    if env_cfg.maze == "default":
        return Maze.default()
    if env_cfg.maze == "small":
        return Maze.small()
    if env_cfg.maze == "random":
        return generate_maze(env_cfg.random_rows, env_cfg.random_cols,
                             loops=env_cfg.random_loops, traps=env_cfg.random_traps,
                             seed=config.training.seed)
    return Maze.from_file(env_cfg.maze)


class Session:
    """A maze + environment + agent + trainer bundle."""

    def __init__(self, config: AppConfig, maze: Maze | None = None) -> None:
        config.validate()
        self.config = config
        self.maze = maze if maze is not None else build_maze(config)
        self.env = MazeEnv(self.maze, config.rewards, config.environment)
        self.agent = QLearningAgent(self.maze.rows, self.maze.cols, config.agent,
                                    seed=config.training.seed)
        self.trainer = Trainer(self.env, self.agent)

    @classmethod
    def from_model(cls, path: str | Path, config: AppConfig | None = None) -> "Session":
        """Restore a saved model so training can continue where it stopped."""
        loaded = load_model(path)
        config = config or AppConfig()
        config.agent = loaded.agent.config
        config.rewards = loaded.rewards
        config.environment = loaded.environment
        session = cls.__new__(cls)
        session.config = config
        session.maze = loaded.maze
        session.env = MazeEnv(loaded.maze, loaded.rewards, loaded.environment)
        session.agent = loaded.agent
        session.trainer = Trainer(session.env, session.agent, loaded.metrics)
        session.trainer.episode_count = loaded.episode_count
        return session

    def save(self, path: str | Path) -> Path:
        return save_model(path, self.agent, self.maze, self.trainer.metrics,
                          self.config.rewards, self.config.environment,
                          self.trainer.episode_count)
