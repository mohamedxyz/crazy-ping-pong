"""Colour palette and font helpers shared by the GUI modules."""

from __future__ import annotations

import pygame

Color = tuple[int, int, int]

BACKGROUND: Color = (18, 22, 31)
PANEL: Color = (27, 34, 48)
PANEL_BORDER: Color = (52, 63, 84)
TEXT: Color = (229, 233, 240)
TEXT_MUTED: Color = (138, 149, 168)
HEADER: Color = (127, 178, 255)
ACCENT: Color = (98, 160, 255)
GOOD: Color = (72, 199, 116)
BAD: Color = (240, 98, 98)
WARN: Color = (246, 173, 85)

BUTTON: Color = (47, 59, 82)
BUTTON_TEXT: Color = (229, 233, 240)
BUTTON_ACTIVE: Color = (98, 160, 255)
BUTTON_ACTIVE_TEXT: Color = (12, 18, 30)
BUTTON_DANGER: Color = (120, 45, 52)
BUTTON_DISABLED: Color = (36, 42, 54)
INPUT_BG: Color = (20, 26, 38)

WALL: Color = (44, 52, 70)
WALL_EDGE: Color = (60, 70, 92)
FLOOR: Color = (236, 230, 216)
FLOOR_ALT: Color = (228, 221, 205)
VISITED: Color = (168, 205, 245)
START: Color = (120, 200, 150)
TRAP: Color = (214, 76, 76)
ROUTE: Color = (240, 140, 40)
BEST_ROUTE: Color = (40, 170, 90)
ARROW: Color = (35, 45, 70)
UNEXPLORED: Color = (160, 160, 160)


def lighten(color: Color, amount: int) -> Color:
    return tuple(min(255, c + amount) for c in color)  # type: ignore[return-value]


def lerp(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def heat_color(t: float) -> Color:
    """Map 0..1 to a blue → teal → yellow → orange ramp."""
    stops: list[Color] = [(49, 80, 170), (40, 160, 170), (240, 210, 70), (240, 120, 40)]
    t = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return lerp(stops[i], stops[i + 1], t - i)


_FONT_CACHE: dict[tuple[int, bool], pygame.font.Font] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _FONT_CACHE:
        f = pygame.font.Font(None, size)
        f.set_bold(bold)
        _FONT_CACHE[key] = f
    return _FONT_CACHE[key]


def clear_font_cache() -> None:
    """Drop cached fonts; they become invalid after ``pygame.quit()``."""
    _FONT_CACHE.clear()
