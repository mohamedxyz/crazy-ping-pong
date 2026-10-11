"""Configuration for MazeMouse.

All tunable values (rewards, learning parameters, training budget, maze
selection) live here as validated dataclasses. They can be loaded from and
saved to JSON so experiments can be changed without touching source code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when a configuration value is missing or out of range."""


@dataclass
class RewardConfig:
    """Reward values for each event the environment can report."""

    goal: float = 100.0
    step: float = -0.1
    wall: float = -2.0
    trap: float = -10.0
    timeout: float = -5.0

    def validate(self) -> None:
        if self.goal <= 0:
            raise ConfigError("rewards.goal must be positive")
        if self.step > 0:
            raise ConfigError("rewards.step must be <= 0 (a positive step reward "
                              "teaches the mouse to wander instead of finishing)")
        if self.wall > 0 or self.trap > 0 or self.timeout > 0:
            raise ConfigError("wall, trap and timeout rewards must be <= 0")


@dataclass
class EnvironmentConfig:
    """Maze selection and episode rules."""

    max_steps: int = 300
    maze: str = "default"          # "default", "small", "random" or a path to a JSON layout
    random_rows: int = 11          # used when maze == "random"
    random_cols: int = 15
    random_loops: int = 6          # extra openings to create alternative routes
    random_traps: int = 3
    trap_terminates: bool = False  # if True, stepping on a trap ends the episode

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ConfigError("environment.max_steps must be >= 1")
        if self.random_rows < 5 or self.random_cols < 5:
            raise ConfigError("random maze dimensions must be >= 5")
        if self.random_loops < 0 or self.random_traps < 0:
            raise ConfigError("random_loops and random_traps must be >= 0")


@dataclass
class AgentConfig:
    """Tabular Q-learning hyper-parameters."""

    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    decay_mode: str = "episode"     # "episode" or "step"

    def validate(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ConfigError("agent.learning_rate must be in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ConfigError("agent.discount must be in [0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ConfigError("need 0 <= epsilon_min <= epsilon_start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ConfigError("agent.epsilon_decay must be in (0, 1]")
        if self.decay_mode not in ("episode", "step"):
            raise ConfigError("agent.decay_mode must be 'episode' or 'step'")


@dataclass
class TrainingConfig:
    """Training budget and evaluation settings."""

    episodes: int = 5000
    seed: int | None = 42
    eval_episodes: int = 20

    def validate(self) -> None:
        if self.episodes < 1:
            raise ConfigError("training.episodes must be >= 1")
        if self.eval_episodes < 1:
            raise ConfigError("training.eval_episodes must be >= 1")


@dataclass
class AppConfig:
    """Top-level configuration bundle."""

    rewards: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.rewards.validate()
        self.environment.validate()
        self.agent.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Build a config from a (possibly partial) dict; unknown keys are rejected."""
        sections = {
            "rewards": RewardConfig,
            "environment": EnvironmentConfig,
            "agent": AgentConfig,
            "training": TrainingConfig,
        }
        unknown = set(data) - set(sections)
        if unknown:
            raise ConfigError(f"unknown config sections: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            raw = data.get(name, {}) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"config section '{name}' must be an object")
            allowed = {f.name for f in fields(section_cls)}
            bad = set(raw) - allowed
            if bad:
                raise ConfigError(f"unknown keys in '{name}': {sorted(bad)}")
            kwargs[name] = section_cls(**raw)
        config = cls(**kwargs)
        config.validate()
        return config


def load_config(path: str | Path | None) -> AppConfig:
    """Load a JSON config file, or return defaults when ``path`` is None."""
    if path is None:
        config = AppConfig()
        config.validate()
        return config
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config file must contain a JSON object")
    logger.info("Loaded configuration from %s", path)
    return AppConfig.from_dict(data)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Write ``config`` to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
