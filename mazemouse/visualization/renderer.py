"""Draws the maze, the mouse, the cheese, routes and the learned policy.

Policy arrows and the heat map are computed directly from the agent's
Q-table and visit counts — nothing here knows the optimal route.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from agent.q_learning import QLearningAgent
from environment.maze import ACTION_DELTAS, Cell, Maze, State
from visualization import theme


class MazeRenderer:
    """Renders a :class:`Maze` scaled to fit an arbitrary rectangle."""

    def __init__(self, maze: Maze) -> None:
        self.maze = maze
        self.cell = 10
        self.origin = (0, 0)
        self._mouse_cache: dict[int, pygame.Surface] = {}
        self._cheese_cache: dict[int, pygame.Surface] = {}

    # --------------------------------------------------------------- geometry
    def fit(self, area: pygame.Rect) -> pygame.Rect:
        self.cell = max(6, min(area.width // self.maze.cols, area.height // self.maze.rows))
        width, height = self.cell * self.maze.cols, self.cell * self.maze.rows
        self.origin = (area.x + (area.width - width) // 2, area.y + (area.height - height) // 2)
        return pygame.Rect(self.origin, (width, height))

    def cell_rect(self, state: State) -> pygame.Rect:
        r, c = state
        return pygame.Rect(self.origin[0] + c * self.cell, self.origin[1] + r * self.cell,
                           self.cell, self.cell)

    def center(self, state: State) -> tuple[int, int]:
        return self.cell_rect(state).center

    # ----------------------------------------------------------------- layers
    def draw_grid(self, surface: pygame.Surface) -> None:
        maze = self.maze
        for r in range(maze.rows):
            for c in range(maze.cols):
                rect = self.cell_rect((r, c))
                kind = maze.grid[r][c]
                if kind == Cell.WALL:
                    pygame.draw.rect(surface, theme.WALL, rect)
                    pygame.draw.rect(surface, theme.WALL_EDGE, rect, width=1)
                else:
                    color = theme.FLOOR if (r + c) % 2 == 0 else theme.FLOOR_ALT
                    pygame.draw.rect(surface, color, rect)

    def draw_heatmap(self, surface: pygame.Surface, agent: QLearningAgent) -> None:
        """Tint each explored cell by its max Q-value (state value estimate)."""
        cells = [s for s in self.maze.traversable_cells()
                 if agent.visit_counts[s].sum() > 0 and self.maze.cell(s) != Cell.GOAL]
        if not cells:
            return
        values = np.array([agent.action_values(s).max() for s in cells])
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else 1.0
        overlay = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        for state, value in zip(cells, values):
            overlay.fill((*theme.heat_color((value - lo) / span), 150))
            surface.blit(overlay, self.cell_rect(state))

    def draw_visited(self, surface: pygame.Surface, visited: set[State]) -> None:
        overlay = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        overlay.fill((*theme.VISITED, 120))
        for state in visited:
            surface.blit(overlay, self.cell_rect(state))

    def draw_special_cells(self, surface: pygame.Surface) -> None:
        maze = self.maze
        start = self.cell_rect(maze.start).inflate(-self.cell // 6, -self.cell // 6)
        pygame.draw.rect(surface, theme.START, start, width=max(2, self.cell // 12),
                         border_radius=4)
        label = theme.font(max(12, self.cell // 2), bold=True).render("S", True, theme.START)
        surface.blit(label, label.get_rect(center=start.center))
        for state in maze.traversable_cells():
            if maze.cell(state) == Cell.TRAP:
                self._draw_trap(surface, self.cell_rect(state))
        surface.blit(self._cheese_sprite(), self.cell_rect(maze.goal))

    def _draw_trap(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        inner = rect.inflate(-self.cell // 5, -self.cell // 5)
        pygame.draw.rect(surface, theme.TRAP, inner, border_radius=4)
        # Spikes
        n = 3
        w = inner.width / n
        for i in range(n):
            x0 = inner.x + i * w
            pygame.draw.polygon(surface, (250, 225, 225), [
                (x0 + w * 0.15, inner.bottom - inner.height * 0.2),
                (x0 + w * 0.5, inner.top + inner.height * 0.2),
                (x0 + w * 0.85, inner.bottom - inner.height * 0.2)])

    def draw_policy(self, surface: pygame.Surface, agent: QLearningAgent) -> None:
        """Arrow = action with the highest Q-value; dot = not explored enough."""
        for state in self.maze.traversable_cells():
            kind = self.maze.cell(state)
            if kind == Cell.GOAL:
                continue
            values = agent.action_values(state)
            counts = agent.visit_counts[state]
            best = np.flatnonzero(values == values.max())
            cx, cy = self.center(state)
            if counts.sum() == 0 or len(best) > 1 or counts[best[0]] == 0:
                pygame.draw.circle(surface, theme.UNEXPLORED, (cx, cy), max(2, self.cell // 10))
                continue
            action = int(best[0])
            confidence = min(1.0, counts[action] / 15.0)
            color = theme.lerp(theme.UNEXPLORED, theme.ARROW, 0.35 + 0.65 * confidence)
            self._draw_arrow(surface, (cx, cy), action, color)

    def _draw_arrow(self, surface: pygame.Surface, center: tuple[int, int], action: int,
                    color: theme.Color) -> None:
        dr, dc = ACTION_DELTAS[action]
        length = self.cell * 0.32
        head = self.cell * 0.18
        cx, cy = center
        tip = (cx + dc * length, cy + dr * length)
        tail = (cx - dc * length, cy - dr * length)
        pygame.draw.line(surface, color, tail, tip, max(2, self.cell // 14))
        px, py = -dr, dc  # perpendicular
        base = (tip[0] - dc * head, tip[1] - dr * head)
        pygame.draw.polygon(surface, color, [
            tip, (base[0] + px * head * 0.7, base[1] + py * head * 0.7),
            (base[0] - px * head * 0.7, base[1] - py * head * 0.7)])

    def draw_route(self, surface: pygame.Surface, path: list[State], color: theme.Color,
                   width: int, dashed: bool = False) -> None:
        if len(path) < 2:
            return
        points = [self.center(s) for s in path]
        if not dashed:
            pygame.draw.lines(surface, color, False, points, width)
            return
        for i, (a, b) in enumerate(zip(points, points[1:])):
            if i % 2 == 0:
                pygame.draw.line(surface, color, a, b, width)

    def draw_mouse(self, surface: pygame.Surface, state: State, facing: int,
                   bumped: bool = False) -> None:
        sprite = self._mouse_sprite()
        angle = {0: 90, 1: 0, 2: -90, 3: 180}[facing]
        rotated = pygame.transform.rotate(sprite, angle)
        rect = self.cell_rect(state)
        if bumped:
            pygame.draw.rect(surface, theme.BAD, rect, width=max(2, self.cell // 10),
                             border_radius=4)
        surface.blit(rotated, rotated.get_rect(center=rect.center))

    # ----------------------------------------------------------------- sprites
    def _mouse_sprite(self) -> pygame.Surface:
        """A simple grey mouse facing right, drawn procedurally."""
        size = self.cell
        if size in self._mouse_cache:
            return self._mouse_cache[size]
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        body = pygame.Rect(0, 0, size * 0.62, size * 0.42)
        body.center = (int(size * 0.46), size // 2)
        tail_pts = [(body.left + 2, body.centery)]
        for i in range(8):
            t = i / 7
            tail_pts.append((body.left - t * size * 0.32,
                             body.centery + math.sin(t * math.pi * 1.6) * size * 0.08))
        pygame.draw.lines(s, (200, 130, 140), False, tail_pts, max(1, size // 22))
        pygame.draw.ellipse(s, (150, 150, 160), body)
        head_c = (int(body.right - size * 0.04), body.centery)
        pygame.draw.circle(s, (165, 165, 175), head_c, int(size * 0.16))
        for dy in (-1, 1):
            ear = (int(head_c[0] - size * 0.08), int(head_c[1] + dy * size * 0.15))
            pygame.draw.circle(s, (165, 165, 175), ear, int(size * 0.1))
            pygame.draw.circle(s, (240, 170, 185), ear, int(size * 0.06))
        nose = (int(head_c[0] + size * 0.16), head_c[1])
        pygame.draw.circle(s, (240, 120, 150), nose, max(2, int(size * 0.04)))
        for dy in (-1, 1):
            eye = (int(head_c[0] + size * 0.05), int(head_c[1] + dy * size * 0.06))
            pygame.draw.circle(s, (20, 20, 25), eye, max(1, int(size * 0.03)))
            pygame.draw.line(s, (90, 90, 95), nose,
                             (nose[0] + size * 0.1, nose[1] + dy * size * 0.08), 1)
        self._mouse_cache[size] = s
        return s

    def _cheese_sprite(self) -> pygame.Surface:
        """A yellow wedge of cheese with holes."""
        size = self.cell
        if size in self._cheese_cache:
            return self._cheese_cache[size]
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 236, 160), (0, 0, size, size), border_radius=4)
        pts = [(size * 0.12, size * 0.78), (size * 0.88, size * 0.78),
               (size * 0.88, size * 0.42), (size * 0.12, size * 0.6)]
        pygame.draw.polygon(s, (250, 200, 40), pts)
        pygame.draw.polygon(s, (205, 150, 20), pts, max(1, size // 20))
        for hx, hy, hr in ((0.35, 0.66, 0.06), (0.62, 0.6, 0.08), (0.78, 0.7, 0.04)):
            pygame.draw.circle(s, (215, 160, 25), (int(size * hx), int(size * hy)),
                               max(1, int(size * hr)))
        self._cheese_cache[size] = s
        return s
