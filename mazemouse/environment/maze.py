"""Grid maze and the reinforcement-learning environment built on it.

The environment follows a Gymnasium-like API::

    state = env.reset()
    next_state, reward, terminated, truncated, info = env.step(action)

* ``terminated`` is True when the episode reached a *terminal state* of the
  MDP (the cheese, or a trap when ``trap_terminates`` is enabled). The value
  of a terminal state is zero, so Q-learning must not bootstrap from it.
* ``truncated`` is True when the episode was cut short by the step limit.
  The mouse is *not* in a terminal state, the episode simply ran out of time.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterable

from config import EnvironmentConfig, RewardConfig

State = tuple[int, int]


class Cell(IntEnum):
    """Cell types stored in the maze grid."""

    FLOOR = 0
    WALL = 1
    START = 2
    GOAL = 3
    TRAP = 4


class Action(IntEnum):
    """The four discrete actions available to the mouse."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


NUM_ACTIONS = 4
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

CHAR_TO_CELL = {".": Cell.FLOOR, " ": Cell.FLOOR, "#": Cell.WALL,
                "S": Cell.START, "G": Cell.GOAL, "T": Cell.TRAP}
CELL_TO_CHAR = {Cell.FLOOR: ".", Cell.WALL: "#", Cell.START: "S",
                Cell.GOAL: "G", Cell.TRAP: "T"}

# The default maze (13x17): dead ends, narrow corridors, two traps and several
# alternative routes (loops) between the start (S) and the cheese (G). It was
# produced once by generate_maze(13, 17, loops=5, traps=2, seed=11) and frozen.
DEFAULT_LAYOUT: list[str] = [
    "#################",
    "#S....#.....#...#",
    "#####.###.#.#.#.#",
    "#.T.#.#...#.....#",
    "#.#.#.#.#######.#",
    "#.#.#.#.......#.#",
    "#...#.#.#####.#.#",
    "#...#.#...T...#.#",
    "#.#.#.#####.#.#.#",
    "#.#...#.....#...#",
    "#.#####.#####.###",
    "#.......#......G#",
    "#################",
]

# A tiny deterministic maze used by tests and quick experiments.
SMALL_LAYOUT: list[str] = [
    "#######",
    "#S..#.#",
    "#.#...#",
    "#.#.#.#",
    "#...#G#",
    "#######",
]


class MazeError(ValueError):
    """Raised for malformed or unsolvable maze layouts."""


class Maze:
    """Immutable grid of cells with exactly one start and one goal."""

    def __init__(self, layout: Iterable[str]) -> None:
        rows = [row for row in layout]
        if not rows:
            raise MazeError("maze layout is empty")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise MazeError("all maze rows must have the same non-zero length")
        grid: list[list[Cell]] = []
        start: State | None = None
        goal: State | None = None
        for r, row in enumerate(rows):
            grid_row: list[Cell] = []
            for c, char in enumerate(row):
                if char not in CHAR_TO_CELL:
                    raise MazeError(f"unknown maze character {char!r} at ({r}, {c})")
                cell = CHAR_TO_CELL[char]
                if cell == Cell.START:
                    if start is not None:
                        raise MazeError("maze has more than one start")
                    start = (r, c)
                elif cell == Cell.GOAL:
                    if goal is not None:
                        raise MazeError("maze has more than one goal")
                    goal = (r, c)
                grid_row.append(cell)
            grid.append(grid_row)
        if start is None or goal is None:
            raise MazeError("maze needs exactly one start 'S' and one goal 'G'")
        self.grid = grid
        self.rows = len(grid)
        self.cols = width
        self.start: State = start
        self.goal: State = goal
        if shortest_path_length(self) is None:
            raise MazeError("goal is not reachable from start")

    # ------------------------------------------------------------------ queries
    def cell(self, state: State) -> Cell:
        r, c = state
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return Cell.WALL  # outside the grid behaves like a wall
        return self.grid[r][c]

    def is_wall(self, state: State) -> bool:
        return self.cell(state) == Cell.WALL

    def traversable_cells(self) -> list[State]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] != Cell.WALL]

    def neighbours(self, state: State) -> list[State]:
        result = []
        for dr, dc in ACTION_DELTAS.values():
            nxt = (state[0] + dr, state[1] + dc)
            if not self.is_wall(nxt):
                result.append(nxt)
        return result

    def to_layout(self) -> list[str]:
        return ["".join(CELL_TO_CHAR[cell] for cell in row) for row in self.grid]

    # ------------------------------------------------------------- constructors
    @classmethod
    def default(cls) -> "Maze":
        return cls(DEFAULT_LAYOUT)

    @classmethod
    def small(cls) -> "Maze":
        return cls(SMALL_LAYOUT)

    @classmethod
    def from_file(cls, path: str | Path) -> "Maze":
        """Load a layout from a JSON file: ``{"layout": ["####", ...]}``."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MazeError(f"cannot read maze file {path}: {exc}") from exc
        layout = data.get("layout") if isinstance(data, dict) else data
        if not isinstance(layout, list) or not all(isinstance(r, str) for r in layout):
            raise MazeError("maze file must contain a list of strings under 'layout'")
        return cls(layout)


def shortest_path(maze: Maze) -> list[State] | None:
    """Breadth-first search from start to goal.

    Used ONLY for validation and as an evaluation benchmark. The learning
    agent never sees the result.
    """
    parents: dict[State, State | None] = {maze.start: None}
    queue: deque[State] = deque([maze.start])
    while queue:
        current = queue.popleft()
        if current == maze.goal:
            path = [current]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])  # type: ignore[arg-type]
            return path[::-1]
        for nxt in maze.neighbours(current):
            if nxt not in parents:
                parents[nxt] = current
                queue.append(nxt)
    return None


def shortest_path_length(maze: Maze) -> int | None:
    """Number of moves on the shortest start→goal path (None if unreachable)."""
    path = shortest_path(maze)
    return None if path is None else len(path) - 1


@dataclass
class StepInfo:
    """Diagnostic information returned by :meth:`MazeEnv.step`."""

    step: int
    position: State
    hit_wall: bool = False
    hit_trap: bool = False
    reached_goal: bool = False
    reason: str | None = None  # "goal", "trap" or "step_limit" when the episode ends
    reward_components: dict[str, float] = field(default_factory=dict)


class MazeEnv:
    """Deterministic grid-world environment for the mouse."""

    def __init__(self, maze: Maze, rewards: RewardConfig | None = None,
                 env_config: EnvironmentConfig | None = None) -> None:
        self.maze = maze
        self.rewards = rewards or RewardConfig()
        self.config = env_config or EnvironmentConfig()
        self.position: State = maze.start
        self.steps = 0
        self.done = True

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    def reset(self) -> State:
        """Put the mouse back on the start cell and begin a new episode."""
        self.position = self.maze.start
        self.steps = 0
        self.done = False
        return self.position

    def step(self, action: int) -> tuple[State, float, bool, bool, StepInfo]:
        """Apply ``action`` and return ``(next_state, reward, terminated, truncated, info)``."""
        if self.done:
            raise RuntimeError("episode is over; call reset() first")
        if action not in ACTION_DELTAS:
            raise ValueError(f"invalid action {action!r}")

        self.steps += 1
        dr, dc = ACTION_DELTAS[action]
        target = (self.position[0] + dr, self.position[1] + dc)
        components: dict[str, float] = {}
        info = StepInfo(step=self.steps, position=self.position)
        terminated = False

        if self.maze.is_wall(target):
            # Bump into the wall: position unchanged, collision penalty.
            info.hit_wall = True
            components["wall"] = self.rewards.wall
        else:
            self.position = target
            cell = self.maze.cell(target)
            if cell == Cell.GOAL:
                info.reached_goal = True
                components["goal"] = self.rewards.goal
                terminated = True
                info.reason = "goal"
            elif cell == Cell.TRAP:
                info.hit_trap = True
                components["trap"] = self.rewards.trap
                if self.config.trap_terminates:
                    terminated = True
                    info.reason = "trap"
            else:
                components["step"] = self.rewards.step

        truncated = False
        if not terminated and self.steps >= self.config.max_steps:
            truncated = True
            info.reason = "step_limit"
            components["timeout"] = self.rewards.timeout

        info.position = self.position
        info.reward_components = components
        self.done = terminated or truncated
        return self.position, float(sum(components.values())), terminated, truncated, info
