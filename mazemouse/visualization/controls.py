"""Minimal Pygame widgets: buttons and a numeric text field.

Widgets only report *which action* was requested; the application decides
what that action does, which keeps the controls easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pygame

from visualization import theme


@dataclass
class Button:
    """A clickable rectangle that triggers ``action`` when pressed."""

    action: str
    label: str | Callable[[], str]
    tooltip: str = ""
    active: Callable[[], bool] | None = None    # highlighted (toggle on)
    enabled: Callable[[], bool] | None = None
    danger: bool = False
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))

    def text(self) -> str:
        return self.label() if callable(self.label) else self.label

    def is_enabled(self) -> bool:
        return self.enabled() if self.enabled is not None else True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, hovered: bool) -> None:
        is_active = self.active() if self.active is not None else False
        if not self.is_enabled():
            bg, fg = theme.BUTTON_DISABLED, theme.TEXT_MUTED
        elif is_active:
            bg, fg = theme.BUTTON_ACTIVE, theme.BUTTON_ACTIVE_TEXT
        elif self.danger:
            bg, fg = theme.BUTTON_DANGER, theme.BUTTON_TEXT
        else:
            bg, fg = theme.BUTTON, theme.BUTTON_TEXT
        if hovered and self.is_enabled():
            bg = theme.lighten(bg, 18)
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        label = font.render(self.text(), True, fg)
        if label.get_width() > self.rect.width - 8:
            scale = (self.rect.width - 8) / label.get_width()
            label = pygame.transform.smoothscale(
                label, (int(label.get_width() * scale), int(label.get_height() * scale)))
        surface.blit(label, label.get_rect(center=self.rect.center))


@dataclass
class NumberField:
    """A small integer input box; type digits, Enter or click elsewhere to commit."""

    value: int
    minimum: int = 1
    maximum: int = 1_000_000
    focused: bool = False
    buffer: str = ""
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))

    def focus(self) -> None:
        self.focused = True
        self.buffer = str(self.value)

    def commit(self) -> None:
        if self.focused:
            try:
                self.value = max(self.minimum, min(self.maximum, int(self.buffer)))
            except ValueError:
                pass
        self.focused = False

    def handle_key(self, event: pygame.event.Event) -> bool:
        """Return True if the key was consumed by the field."""
        if not self.focused:
            return False
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
            self.commit()
        elif event.key == pygame.K_ESCAPE:
            self.focused = False
        elif event.key == pygame.K_BACKSPACE:
            self.buffer = self.buffer[:-1]
        elif event.unicode.isdigit() and len(self.buffer) < 7:
            self.buffer += event.unicode
        return True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        border = theme.ACCENT if self.focused else theme.PANEL_BORDER
        pygame.draw.rect(surface, theme.INPUT_BG, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=6)
        text = (self.buffer + "|") if self.focused else f"{self.value:,}"
        label = font.render(text, True, theme.TEXT)
        surface.blit(label, label.get_rect(midleft=(self.rect.x + 8, self.rect.centery)))
