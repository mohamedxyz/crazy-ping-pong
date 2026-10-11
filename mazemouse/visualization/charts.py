"""Matplotlib training charts (reward, length, success rate, epsilon).

Uses the object-oriented Agg API (no pyplot), so charts can be produced in
headless runs and converted into Pygame surfaces without a GUI backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from training.metrics import MetricsTracker, rolling_mean

RAW_COLOR = "#9aa5b1"
TREND_COLOR = "#2a6fdb"
SUCCESS_COLOR = "#1f9d55"
EPS_COLOR = "#d97706"


def default_window(n: int) -> int:
    return max(5, min(200, n // 20 or 5))


def build_training_figure(metrics: MetricsTracker, width_in: float = 8.0,
                          height_in: float = 6.0, dpi: int = 100,
                          dark: bool = False) -> Figure:
    """Create a 2x2 figure of training curves with rolling averages."""
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fg = "#e5e7eb" if dark else "#1f2933"
    bg = "#1b2230" if dark else "#ffffff"
    fig.patch.set_facecolor(bg)
    axes = fig.subplots(2, 2)
    n = len(metrics)
    episodes = np.arange(1, n + 1)
    window = default_window(n)
    rewards = metrics.series("total_reward")
    steps = metrics.series("steps")
    success = metrics.series("success")
    epsilon = metrics.series("epsilon")

    panels = [
        (axes[0, 0], "Episode reward", rewards, TREND_COLOR, True),
        (axes[0, 1], "Episode length (steps)", steps, TREND_COLOR, True),
        (axes[1, 0], f"Success rate (rolling {window})", success, SUCCESS_COLOR, False),
        (axes[1, 1], "Exploration rate ε", epsilon, EPS_COLOR, False),
    ]
    for ax, title, values, color, show_raw in panels:
        ax.set_facecolor(bg)
        ax.set_title(title, fontsize=9, color=fg)
        ax.tick_params(labelsize=7, colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#4b5563" if dark else "#cbd2d9")
        ax.grid(True, alpha=0.25)
        if n == 0:
            ax.text(0.5, 0.5, "no training yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color=fg)
            continue
        if show_raw:
            ax.plot(episodes, values, color=RAW_COLOR, linewidth=0.6, alpha=0.5)
            ax.plot(episodes, rolling_mean(values, window), color=color, linewidth=1.6)
        elif title.startswith("Success"):
            ax.plot(episodes, 100 * rolling_mean(values, window), color=color, linewidth=1.6)
            ax.set_ylim(-5, 105)
            ax.set_ylabel("%", fontsize=7, color=fg)
        else:
            ax.plot(episodes, values, color=color, linewidth=1.6)
            ax.set_ylim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("episode", fontsize=7, color=fg)
    fig.tight_layout(pad=0.8)
    return fig


def save_training_charts(metrics: MetricsTracker, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_training_figure(metrics, 10, 7)
    fig.savefig(path)
    return path


def figure_to_rgba(fig: Figure) -> tuple[bytes, tuple[int, int]]:
    """Render ``fig`` and return raw RGBA bytes and (width, height)."""
    canvas = fig.canvas
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    height, width = buffer.shape[:2]
    return buffer.tobytes(), (width, height)
