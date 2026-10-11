"""Seeded random maze generation.

Uses an iterative recursive-backtracker (depth-first search) to carve a
perfect maze, then knocks out a few extra walls to create loops (alternative
routes) and finally sprinkles traps on cells that do not block every
trap-free route to the cheese.
"""

from __future__ import annotations

import numpy as np

from environment.maze import CELL_TO_CHAR, Cell, Maze, MazeError, State


def _trap_free_path_exists(grid: list[list[Cell]], start: State, goal: State) -> bool:
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen \
                    and grid[nr][nc] not in (Cell.WALL, Cell.TRAP):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return False


def generate_maze(rows: int = 11, cols: int = 15, loops: int = 6, traps: int = 3,
                  seed: int | None = None) -> Maze:
    """Generate a solvable maze.

    Args:
        rows, cols: grid size including the outer wall; even values are
            rounded up to the next odd number.
        loops: number of extra walls to remove, creating alternative routes.
        traps: number of trap cells to place (never cutting off the goal).
        seed: random seed for reproducible generation.
    """
    if rows < 5 or cols < 5:
        raise MazeError("maze must be at least 5x5")
    rows += 1 - rows % 2
    cols += 1 - cols % 2
    rng = np.random.default_rng(seed)
    grid = [[Cell.WALL] * cols for _ in range(rows)]

    # Carve passages between "room" cells located at odd coordinates.
    start: State = (1, 1)
    grid[1][1] = Cell.FLOOR
    stack: list[State] = [start]
    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in ((-2, 0), (0, 2), (2, 0), (0, -2)):
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr][nc] == Cell.WALL:
                options.append((nr, nc))
        if not options:
            stack.pop()
            continue
        nr, nc = options[int(rng.integers(len(options)))]
        grid[(r + nr) // 2][(c + nc) // 2] = Cell.FLOOR
        grid[nr][nc] = Cell.FLOOR
        stack.append((nr, nc))

    # Remove interior walls that separate two passages to create loops.
    candidates = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != Cell.WALL:
                continue
            horizontal = grid[r][c - 1] != Cell.WALL and grid[r][c + 1] != Cell.WALL \
                and grid[r - 1][c] == Cell.WALL and grid[r + 1][c] == Cell.WALL
            vertical = grid[r - 1][c] != Cell.WALL and grid[r + 1][c] != Cell.WALL \
                and grid[r][c - 1] == Cell.WALL and grid[r][c + 1] == Cell.WALL
            if horizontal or vertical:
                candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[:loops]:
        grid[r][c] = Cell.FLOOR

    goal: State = (rows - 2, cols - 2)
    grid[start[0]][start[1]] = Cell.START
    grid[goal[0]][goal[1]] = Cell.GOAL

    # Place traps only where a trap-free route to the goal still exists.
    floor = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == Cell.FLOOR]
    rng.shuffle(floor)
    placed = 0
    for r, c in floor:
        if placed >= traps:
            break
        if abs(r - start[0]) + abs(c - start[1]) <= 2:
            continue  # keep the start area safe
        grid[r][c] = Cell.TRAP
        if _trap_free_path_exists(grid, start, goal):
            placed += 1
        else:
            grid[r][c] = Cell.FLOOR

    layout = ["".join(CELL_TO_CHAR[cell] for cell in row) for row in grid]
    return Maze(layout)
