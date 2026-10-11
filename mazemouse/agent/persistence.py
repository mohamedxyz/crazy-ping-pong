"""Save and load learned models as portable JSON files."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent.q_learning import NUM_ACTIONS, QLearningAgent
from config import AgentConfig, ConfigError, EnvironmentConfig, RewardConfig
from environment.maze import Maze, MazeError
from training.metrics import MetricsTracker

logger = logging.getLogger(__name__)

MODEL_FORMAT = "mazemouse-qtable"
MODEL_VERSION = 1


class ModelLoadError(ValueError):
    """Raised when a model file is missing, corrupt or incompatible."""


@dataclass
class LoadedModel:
    """Everything restored from a model file."""

    maze: Maze
    agent: QLearningAgent
    metrics: MetricsTracker
    rewards: RewardConfig
    environment: EnvironmentConfig
    episode_count: int


def _encode_rng_state(state: dict[str, Any]) -> dict[str, Any]:
    # PCG64 state contains Python ints larger than 2**64; JSON stores them
    # exactly but some tools do not, so encode them as strings.
    def encode(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value
    return encode(state)


def _decode_rng_state(state: dict[str, Any]) -> dict[str, Any]:
    def decode(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        return value
    return decode(state)


def save_model(path: str | Path, agent: QLearningAgent, maze: Maze,
               metrics: MetricsTracker, rewards: RewardConfig,
               environment: EnvironmentConfig, episode_count: int) -> Path:
    """Write the Q-table, maze, parameters and statistics to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "maze": {"layout": maze.to_layout()},
        "agent": {
            "config": asdict(agent.config),
            "epsilon": agent.epsilon,
            "seed": agent.seed,
            "rng_state": _encode_rng_state(agent.rng.bit_generator.state),
            "total_updates": agent.total_updates,
        },
        "rewards": asdict(rewards),
        "environment": asdict(environment),
        "episode_count": episode_count,
        "q_table": agent.q_table.tolist(),
        "visit_counts": agent.visit_counts.tolist(),
        "metrics": metrics.to_list(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem
    logger.info("Saved model to %s (%d episodes)", path, episode_count)
    return path


def load_model(path: str | Path) -> LoadedModel:
    """Load and validate a model file written by :func:`save_model`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelLoadError(f"model file not found: {path}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelLoadError(f"model file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
        raise ModelLoadError("not a MazeMouse model file")
    if data.get("version") != MODEL_VERSION:
        raise ModelLoadError(f"unsupported model version {data.get('version')!r} "
                             f"(expected {MODEL_VERSION})")
    try:
        maze = Maze(data["maze"]["layout"])
        agent_data = data["agent"]
        agent_config = AgentConfig(**agent_data["config"])
        rewards = RewardConfig(**data["rewards"])
        environment = EnvironmentConfig(**data["environment"])
        agent_config.validate()
        rewards.validate()
        environment.validate()
        q_table = np.asarray(data["q_table"], dtype=np.float64)
        visits = np.asarray(data.get("visit_counts", np.zeros_like(q_table)), dtype=np.int64)
        expected = (maze.rows, maze.cols, NUM_ACTIONS)
        if q_table.shape != expected or visits.shape != expected:
            raise ModelLoadError(f"Q-table shape {q_table.shape} does not match maze "
                                 f"{expected}")
        if not np.all(np.isfinite(q_table)):
            raise ModelLoadError("Q-table contains NaN or infinite values")
        epsilon = float(agent_data["epsilon"])
        if not (0.0 <= epsilon <= 1.0) or math.isnan(epsilon):
            raise ModelLoadError("epsilon must be within [0, 1]")
        seed = agent_data.get("seed")
        agent = QLearningAgent(maze.rows, maze.cols, agent_config,
                               seed=None if seed is None else int(seed))
        agent.q_table[...] = q_table
        agent.visit_counts[...] = visits
        agent.epsilon = epsilon
        agent.total_updates = int(agent_data.get("total_updates", 0))
        if agent_data.get("rng_state"):
            try:
                agent.rng.bit_generator.state = _decode_rng_state(agent_data["rng_state"])
            except (TypeError, ValueError, KeyError):
                logger.warning("Could not restore RNG state; continuing with seed only")
        metrics = MetricsTracker.from_list(data.get("metrics", []))
        episode_count = int(data.get("episode_count", len(metrics)))
    except ModelLoadError:
        raise
    except (KeyError, TypeError, ValueError, MazeError, ConfigError) as exc:
        raise ModelLoadError(f"model file is invalid: {exc}") from exc
    logger.info("Loaded model from %s (%d episodes)", path, episode_count)
    return LoadedModel(maze, agent, metrics, rewards, environment, episode_count)
