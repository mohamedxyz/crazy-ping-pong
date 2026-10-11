"""Statistics panel, Q-update explanation panel and embedded Matplotlib charts."""

from __future__ import annotations

import time

import pygame

from agent.q_learning import QUpdate
from environment.maze import Action
from training.metrics import MetricsTracker
from visualization import theme
from visualization.charts import build_training_figure, figure_to_rgba

ACTION_NAMES = {int(a): a.name for a in Action}


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, title: str) -> int:
    """Draw a titled panel and return the y coordinate where content starts."""
    pygame.draw.rect(surface, theme.PANEL, rect, border_radius=8)
    pygame.draw.rect(surface, theme.PANEL_BORDER, rect, width=1, border_radius=8)
    label = theme.font(20, bold=True).render(title.upper(), True, theme.HEADER)
    surface.blit(label, (rect.x + 10, rect.y + 8))
    return rect.y + 8 + label.get_height() + 6


def draw_rows(surface: pygame.Surface, rect: pygame.Rect, y: int,
              rows: list[tuple[str, str, theme.Color | None]], columns: int = 1,
              size: int = 19) -> int:
    """Draw ``label: value`` rows in one or more columns; return the next y."""
    f = theme.font(size)
    line_h = f.get_linesize() + 2
    col_w = (rect.width - 20) // columns
    per_col = -(-len(rows) // columns)
    for i, (label, value, color) in enumerate(rows):
        col, row = divmod(i, per_col)
        x = rect.x + 10 + col * col_w
        yy = y + row * line_h
        surface.blit(f.render(label, True, theme.TEXT_MUTED), (x, yy))
        val = f.render(value, True, color or theme.TEXT)
        surface.blit(val, (x + col_w - val.get_width() - 10, yy))
    return y + per_col * line_h


def wrap_text(text: str, font: pygame.font.Font, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_paragraph(surface: pygame.Surface, text: str, x: int, y: int, width: int,
                   size: int = 18, color: theme.Color = theme.TEXT, max_y: int | None = None
                   ) -> int:
    f = theme.font(size)
    for line in wrap_text(text, f, width):
        if max_y is not None and y + f.get_linesize() > max_y:
            break
        surface.blit(f.render(line, True, color), (x, y))
        y += f.get_linesize()
    return y


def fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def draw_q_update(surface: pygame.Surface, rect: pygame.Rect, update: QUpdate | None,
                  wall_history: list[float] | None, wall_key: tuple | None,
                  best_at_wall: tuple[int, float] | None = None) -> None:
    """Explain the most recent real Q-learning update with its actual numbers."""
    y = draw_panel(surface, rect, "Last Q-learning update")
    x, width = rect.x + 10, rect.width - 20
    if update is None:
        draw_paragraph(surface, "No update yet. Start training to see the real numbers "
                       "behind each step.", x, y, width, color=theme.TEXT_MUTED)
        return
    f = theme.font(18)
    mono = theme.font(19, bold=True)
    action = ACTION_NAMES[update.action]
    head = f"s = {update.state}, a = {action}" + ("   (terminal)" if update.terminated else "")
    surface.blit(f.render(head, True, theme.TEXT), (x, y))
    y += f.get_linesize() + 2
    bootstrap = ("0 (terminal)" if update.terminated
                 else f"{update.max_next_q:+.3f}")
    rows = [
        ("old Q(s,a)", f"{update.old_q:+.3f}", None),
        ("reward r", f"{update.reward:+.2f}",
         theme.GOOD if update.reward > 0 else theme.BAD if update.reward < -0.5 else None),
        ("max Q(s',a')", bootstrap, None),
        ("alpha / gamma", f"{update.alpha:.3f} / {update.gamma:.3f}", None),
    ]
    y = draw_rows(surface, rect, y, rows, columns=2, size=18)
    line1 = (f"target = r + γ·maxQ' = {update.reward:+.2f} + {update.gamma:.2f}·"
             f"{update.max_next_q:+.2f} = {update.target:+.3f}")
    line2 = (f"new Q = {update.old_q:+.3f} + {update.alpha:.2f}·({update.target:+.3f} − "
             f"{update.old_q:+.3f}) = {update.new_q:+.3f}")
    for line in (line1, line2):
        surface.blit(mono.render(line, True, theme.ACCENT), (x, y))
        y += mono.get_linesize()
    if wall_history and wall_key and len(wall_history) >= 2:
        (state, act) = wall_key
        values = " > ".join(f"{v:+.2f}" for v in wall_history[-6:])
        y += 2
        y = draw_paragraph(surface, f"Wall bumps at {state} going {ACTION_NAMES[act]}: "
                           f"Q = {values}", x, y, width, size=17, color=theme.WARN,
                           max_y=rect.bottom - 4)
        if best_at_wall is not None:
            best_action, best_q = best_at_wall
            y = draw_paragraph(surface, f"Best action in that cell: {ACTION_NAMES[best_action]} "
                               f"(Q = {best_q:+.2f})", x, y, width, size=17,
                               color=theme.TEXT, max_y=rect.bottom - 4)
        draw_paragraph(surface, "The mouse keeps no list of mistakes: each bump just lowers "
                       "the estimated long-term value of that action.", x, y, width, size=16,
                       color=theme.TEXT_MUTED, max_y=rect.bottom - 4)


class ChartPanel:
    """Caches a Matplotlib render of the training curves; refreshes on a timer."""

    def __init__(self, refresh_seconds: float = 1.5) -> None:
        self.refresh_seconds = refresh_seconds
        self.surface: pygame.Surface | None = None
        self._last_render = 0.0
        self._last_count = -1
        self._last_size = (0, 0)

    def invalidate(self) -> None:
        self._last_count = -1

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, metrics: MetricsTracker) -> None:
        y = draw_panel(surface, rect, "Training curves")
        area = pygame.Rect(rect.x + 6, y, rect.width - 12, rect.bottom - y - 6)
        if area.width < 80 or area.height < 60:
            return
        now = time.monotonic()
        size = area.size
        stale = len(metrics) != self._last_count or size != self._last_size
        if stale and (now - self._last_render >= self.refresh_seconds
                      or self.surface is None or size != self._last_size):
            dpi = 80
            fig = build_training_figure(metrics, size[0] / dpi, size[1] / dpi, dpi, dark=True)
            raw, dims = figure_to_rgba(fig)
            self.surface = pygame.image.frombuffer(raw, dims, "RGBA").copy()
            self._last_render = now
            self._last_count = len(metrics)
            self._last_size = size
        if self.surface is not None:
            surface.blit(self.surface, area.topleft)
