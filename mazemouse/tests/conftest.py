"""Shared fixtures for the MazeMouse test suite."""

from __future__ import annotations

import pytest

from config import AgentConfig, EnvironmentConfig, RewardConfig
from environment.maze import Maze, MazeEnv

# A tiny corridor maze used for exact, hand-checkable assertions.
#   #####
#   #S.G#
#   #####
CORRIDOR = ["#####", "#S.G#", "#####"]


@pytest.fixture
def corridor_maze() -> Maze:
    return Maze(CORRIDOR)


@pytest.fixture
def small_maze() -> Maze:
    return Maze.small()


@pytest.fixture
def rewards() -> RewardConfig:
    return RewardConfig()


def make_env(maze: Maze, max_steps: int = 300, trap_terminates: bool = False) -> MazeEnv:
    return MazeEnv(maze, RewardConfig(),
                   EnvironmentConfig(max_steps=max_steps, trap_terminates=trap_terminates))


def fast_agent_config() -> AgentConfig:
    return AgentConfig(learning_rate=0.2, discount=0.95, epsilon_start=1.0,
                       epsilon_min=0.02, epsilon_decay=0.99)
