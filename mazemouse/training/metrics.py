"""Episode-level statistics and aggregate training metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeStats:
    """Everything recorded about one finished episode."""

    episode: int
    total_reward: float
    steps: int
    success: bool
    wall_collisions: int
    trap_encounters: int
    unique_cells: int
    epsilon: float
    duration_s: float
    reason: str
    reward_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeStats":
        return cls(
            episode=int(data["episode"]),
            total_reward=float(data["total_reward"]),
            steps=int(data["steps"]),
            success=bool(data["success"]),
            wall_collisions=int(data["wall_collisions"]),
            trap_encounters=int(data["trap_encounters"]),
            unique_cells=int(data["unique_cells"]),
            epsilon=float(data["epsilon"]),
            duration_s=float(data.get("duration_s", 0.0)),
            reason=str(data.get("reason", "")),
            reward_components={str(k): float(v)
                               for k, v in dict(data.get("reward_components", {})).items()},
        )


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; early points average over what is available."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    idx = np.arange(1, arr.size + 1)
    lo = np.maximum(0, idx - window)
    return (cumsum[idx] - cumsum[lo]) / (idx - lo)


class MetricsTracker:
    """Collects :class:`EpisodeStats` for training episodes only."""

    def __init__(self) -> None:
        self.episodes: list[EpisodeStats] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, stats: EpisodeStats) -> None:
        self.episodes.append(stats)

    def clear(self) -> None:
        self.episodes.clear()

    def series(self, name: str) -> np.ndarray:
        return np.array([float(getattr(e, name)) for e in self.episodes], dtype=np.float64)

    def summary(self, recent: int = 100) -> dict[str, float | int | None]:
        """Aggregate numbers shown on the dashboard."""
        n = len(self.episodes)
        successes = [e for e in self.episodes if e.success]
        last = self.episodes[-recent:]
        return {
            "episodes": n,
            "success_rate": len(successes) / n if n else 0.0,
            "recent_success_rate": (sum(e.success for e in last) / len(last)) if last else 0.0,
            "avg_success_steps": (float(np.mean([e.steps for e in successes]))
                                  if successes else None),
            "recent_avg_steps": float(np.mean([e.steps for e in last])) if last else None,
            "best_success_steps": min((e.steps for e in successes), default=None),
            "total_wall_collisions": sum(e.wall_collisions for e in self.episodes),
            "total_trap_encounters": sum(e.trap_encounters for e in self.episodes),
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.episodes]

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "MetricsTracker":
        tracker = cls()
        tracker.episodes = [EpisodeStats.from_dict(item) for item in items]
        return tracker
