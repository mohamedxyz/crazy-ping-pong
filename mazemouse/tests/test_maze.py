"""Environment rules, maze parsing and maze generation."""

from __future__ import annotations

import pytest

from environment.generator import generate_maze
from environment.maze import Action, Cell, Maze, MazeError, shortest_path, shortest_path_length
from tests.conftest import make_env


def test_mouse_starts_at_start_cell(small_maze):
    env = make_env(small_maze)
    assert env.reset() == small_maze.start == (1, 1)
    assert env.steps == 0


def test_mouse_cannot_pass_through_walls(corridor_maze):
    env = make_env(corridor_maze)
    env.reset()
    for action in (Action.UP, Action.DOWN, Action.LEFT):
        state, _, terminated, truncated, info = env.step(action)
        assert state == (1, 1)
        assert info.hit_wall
        assert not terminated and not truncated


def test_wall_collision_penalty(corridor_maze, rewards):
    env = make_env(corridor_maze)
    env.reset()
    _, reward, _, _, info = env.step(Action.UP)
    assert reward == pytest.approx(rewards.wall)
    assert info.reward_components == {"wall": rewards.wall}


def test_valid_move_costs_step_penalty(corridor_maze, rewards):
    env = make_env(corridor_maze)
    env.reset()
    state, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert state == (1, 2)
    assert reward == pytest.approx(rewards.step)
    assert not (terminated or truncated or info.hit_wall)


def test_reaching_cheese_terminates(corridor_maze, rewards):
    env = make_env(corridor_maze)
    env.reset()
    env.step(Action.RIGHT)
    state, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert state == corridor_maze.goal
    assert terminated and not truncated
    assert info.reached_goal and info.reason == "goal"
    assert reward == pytest.approx(rewards.goal)
    with pytest.raises(RuntimeError):
        env.step(Action.RIGHT)


def test_step_limit_truncates(corridor_maze, rewards):
    env = make_env(corridor_maze, max_steps=3)
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(Action.UP)
        assert not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(Action.UP)
    assert truncated and not terminated
    assert info.reason == "step_limit"
    assert info.reward_components == {"wall": rewards.wall, "timeout": rewards.timeout}
    assert reward == pytest.approx(rewards.wall + rewards.timeout)


def test_goal_on_last_step_is_terminal_not_truncated(corridor_maze):
    env = make_env(corridor_maze, max_steps=2)
    env.reset()
    env.step(Action.RIGHT)
    _, _, terminated, truncated, info = env.step(Action.RIGHT)
    assert terminated and not truncated and "timeout" not in info.reward_components


def test_trap_penalty_and_optional_termination(rewards):
    maze = Maze(["#####", "#STG#", "#####"])
    env = make_env(maze)
    env.reset()
    state, reward, terminated, _, info = env.step(Action.RIGHT)
    assert info.hit_trap and state == (1, 2) and not terminated
    assert reward == pytest.approx(rewards.trap)
    env = make_env(maze, trap_terminates=True)
    env.reset()
    _, _, terminated, _, info = env.step(Action.RIGHT)
    assert terminated and info.reason == "trap"


def test_reset_starts_new_episode(corridor_maze):
    env = make_env(corridor_maze)
    env.reset()
    env.step(Action.RIGHT)
    assert env.reset() == corridor_maze.start and env.steps == 0


def test_invalid_action_rejected(corridor_maze):
    env = make_env(corridor_maze)
    env.reset()
    with pytest.raises(ValueError):
        env.step(7)


@pytest.mark.parametrize("layout", [
    [],
    ["###", "#S"],
    ["#####", "#S.X#", "#####"],
    ["#####", "#S..#", "#####"],              # no goal
    ["#####", "#S#G#", "#####"],              # unreachable goal
    ["######", "#SSG.#", "######"],            # two starts
])
def test_invalid_layouts_rejected(layout):
    with pytest.raises(MazeError):
        Maze(layout)


def test_default_maze_is_interesting():
    maze = Maze.default()
    cells = maze.traversable_cells()
    dead_ends = [c for c in cells if len(maze.neighbours(c)) == 1 and c != maze.goal]
    edges = sum(len(maze.neighbours(c)) for c in cells) // 2
    loops = edges - len(cells) + 1
    assert len(dead_ends) >= 3
    assert loops >= 2                     # alternative routes exist
    assert any(maze.cell(c) == Cell.TRAP for c in cells)
    assert shortest_path_length(maze) == 44


def test_shortest_path_is_valid(small_maze):
    path = shortest_path(small_maze)
    assert path[0] == small_maze.start and path[-1] == small_maze.goal
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert not small_maze.is_wall(b)


@pytest.mark.parametrize("seed", range(10))
def test_generator_produces_solvable_mazes(seed):
    maze = generate_maze(11, 15, loops=4, traps=3, seed=seed)
    assert shortest_path_length(maze) is not None
    assert all(maze.is_wall((r, 0)) and maze.is_wall((r, maze.cols - 1))
               for r in range(maze.rows))
    traps = sum(maze.cell(c) == Cell.TRAP for c in maze.traversable_cells())
    assert traps <= 3


def test_generator_is_reproducible():
    a = generate_maze(13, 17, seed=7).to_layout()
    b = generate_maze(13, 17, seed=7).to_layout()
    c = generate_maze(13, 17, seed=8).to_layout()
    assert a == b
    assert a != c


def test_layout_file_roundtrip(tmp_path, small_maze):
    import json
    path = tmp_path / "maze.json"
    path.write_text(json.dumps({"layout": small_maze.to_layout()}))
    assert Maze.from_file(path).to_layout() == small_maze.to_layout()
    path.write_text("not json")
    with pytest.raises(MazeError):
        Maze.from_file(path)
