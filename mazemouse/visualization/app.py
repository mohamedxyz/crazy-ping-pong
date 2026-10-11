"""Interactive Pygame laboratory for watching the mouse learn.

Concurrency model: everything runs on the main thread. Training advances in
small, time-budgeted batches inside the frame loop (a few milliseconds per
frame), so the window stays responsive during long runs and there is no
shared state between threads to protect. ``clock.tick`` sleeps between
frames, so the loop never busy-waits.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from agent.persistence import ModelLoadError
from environment.maze import MazeEnv, State, shortest_path_length
from session import Session
from training.evaluation import EvaluationReport, evaluate_agent
from training.metrics import EpisodeStats
from training.trainer import POLICY_EPSILON, POLICY_GREEDY, POLICY_RANDOM
from visualization import theme
from visualization.controls import Button, NumberField
from visualization.dashboard import (ChartPanel, draw_panel, draw_paragraph, draw_q_update,
                                     draw_rows, fmt)
from visualization.renderer import MazeRenderer

logger = logging.getLogger(__name__)

IDLE, TRAIN, BATCH, WATCH, DEMO = "idle", "train", "batch", "watch", "demo"
SPEEDS: list[int] = [1, 3, 8, 20, 60, 200, 800, 0]  # steps per second; 0 = turbo
DECAYS: list[float] = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 1.0]
FRAME_BUDGET_S = 0.025       # max time spent training per frame
DEMO_EPISODES = 1500         # training episodes used by demo stage B
DEMO_ANIMATED_EPISODES = 2   # stage B episodes shown step by step before fast-forward
DEMO_HOLD_S = 2.5            # pause between demo stages so the text can be read


@dataclass
class DemoState:
    stage: str = "A"                  # A, B_anim, B_batch, C, done
    hold_until: float = 0.0
    animated_done: int = 0
    stats_a: EpisodeStats | None = None
    stats_c: EpisodeStats | None = None
    path_a: list[State] | None = None
    report: EvaluationReport | None = None


class MazeMouseApp:
    """Owns the window, the active session and all UI state."""

    def __init__(self, session: Session, model_path: Path,
                 size: tuple[int, int] = (1440, 880)) -> None:
        pygame.display.init()
        pygame.font.init()
        theme.clear_font_cache()
        pygame.display.set_caption("MazeMouse — a mouse that learns from its mistakes")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.session = session
        self.main_session: Session | None = None   # stored while the demo runs
        self.model_path = model_path
        self.renderer = MazeRenderer(session.maze)
        self.charts = ChartPanel()
        self.mode = IDLE
        self.paused = False
        self.speed_index = 3
        self.step_credit = 0.0
        self.batch_target = 0
        self.batch_start = 0
        self.show_policy = False
        self.show_heatmap = False
        self.show_best = True
        self.show_visited = True
        self.facing = 1
        self.message = ("Welcome! Press 'Start training' or run the demo to see learning "
                        "from mistakes.")
        self.message_color = theme.TEXT
        self.eval_report: EvaluationReport | None = None
        self.eval_path: list[State] | None = None
        self.watch_result: EpisodeStats | None = None
        self.confirm_reset_until = 0.0
        self.random_maze_seed = (session.config.training.seed or 0) + 1
        self.demo: DemoState | None = None
        self.episodes_field = NumberField(value=500, minimum=1, maximum=1_000_000)
        self.running = True
        self.buttons: list[Button] = []
        self.section_labels: list[tuple[str, tuple[int, int]]] = []
        self.param_labels: list[tuple[str, pygame.Rect]] = []
        self._build_buttons()
        self.layout()

    # ================================================================ helpers
    @property
    def agent(self):
        return self.session.agent

    @property
    def trainer(self):
        return self.session.trainer

    def say(self, text: str, color: theme.Color = theme.TEXT) -> None:
        self.message = text
        self.message_color = color
        logger.info("UI: %s", text)

    def not_demo(self) -> bool:
        return self.demo is None

    def speed_label(self) -> str:
        speed = SPEEDS[self.speed_index]
        return "turbo" if speed == 0 else f"{speed} steps/s"

    def train_label(self) -> str:
        if self.mode in (TRAIN, BATCH) and not self.paused:
            return "Pause training"
        if self.mode in (TRAIN, BATCH) and self.paused:
            return "Resume training"
        return "Start training"

    # ================================================================ widgets
    def _build_buttons(self) -> None:
        nd = self.not_demo
        B = Button
        self.button_specs: list[tuple[str, list[Button | str]]] = [
            ("Training", [
                B("start_pause", self.train_label, active=lambda: self.mode in (TRAIN, BATCH)
                  and not self.paused, enabled=nd),
                B("pause", lambda: "Resume" if self.paused else "Pause all",
                  active=lambda: self.paused),
                B("reset_episode", "Reset episode", enabled=nd),
                B("reset_all", lambda: ("Click again to confirm" if time.monotonic()
                                        < self.confirm_reset_until else "Reset ALL learning"),
                  danger=True, enabled=nd),
                B("train_100", "Train 100", enabled=nd),
                B("train_1000", "Train 1,000", enabled=nd),
                "field",
                B("train_n", "Train N", enabled=nd),
                B("watch", "Watch trained mouse", enabled=nd,
                  active=lambda: self.mode == WATCH),
                B("evaluate", "Evaluate (20 runs)", enabled=nd),
            ]),
            ("View", [
                B("toggle_explore", lambda: "Exploration: " + (
                    "ON" if self.agent.exploration_enabled else "OFF"),
                  active=lambda: self.agent.exploration_enabled),
                B("toggle_policy", "Policy arrows", active=lambda: self.show_policy),
                B("toggle_heatmap", "Value heatmap", active=lambda: self.show_heatmap),
                B("toggle_best", "Best route", active=lambda: self.show_best),
            ]),
            ("Parameters", [
                "param:speed", B("speed_down", "−"), B("speed_up", "+"),
                "param:alpha", B("alpha_down", "−"), B("alpha_up", "+"),
                "param:gamma", B("gamma_down", "−"), B("gamma_up", "+"),
                "param:decay", B("decay_down", "−"), B("decay_up", "+"),
            ]),
            ("Model & maze", [
                B("save", "Save model", enabled=nd), B("load", "Load model", enabled=nd),
                B("random_maze", "New random maze", enabled=nd),
                B("default_maze", "Default maze", enabled=nd),
            ]),
            ("Demonstration", [
                B("demo", lambda: "Replay demo" if self.demo else "Run demo A-B-C",
                  active=lambda: self.demo is not None and self.demo.stage != "done"),
                B("exit_demo", "Exit demo", enabled=lambda: self.demo is not None),
            ]),
        ]

    def layout(self) -> None:
        width, height = self.screen.get_size()
        self.left = pygame.Rect(10, 10, 250, height - 20)
        right_w = max(380, min(520, int(width * 0.31)))
        self.right = pygame.Rect(width - right_w - 10, 10, right_w, height - 20)
        center_x = self.left.right + 10
        self.center = pygame.Rect(center_x, 10, self.right.x - center_x - 10, height - 20)
        self.header_rect = pygame.Rect(self.center.x, self.center.y, self.center.width, 40)
        narrative_h = 150
        self.narrative_rect = pygame.Rect(self.center.x, self.center.bottom - narrative_h,
                                          self.center.width, narrative_h)
        self.maze_area = pygame.Rect(self.center.x, self.header_rect.bottom + 6,
                                     self.center.width,
                                     self.narrative_rect.y - self.header_rect.bottom - 12)
        # Right column
        stats_h = 215
        q_h = 235
        self.stats_rect = pygame.Rect(self.right.x, self.right.y, self.right.width, stats_h)
        self.q_rect = pygame.Rect(self.right.x, self.stats_rect.bottom + 8, self.right.width, q_h)
        self.chart_rect = pygame.Rect(self.right.x, self.q_rect.bottom + 8, self.right.width,
                                      self.right.bottom - self.q_rect.bottom - 8)
        # Left column buttons
        self.buttons.clear()
        self.section_labels.clear()
        self.param_labels.clear()
        x0, y = self.left.x + 10, self.left.y + 10
        full_w = self.left.width - 20
        half_w = (full_w - 6) // 2
        btn_h = 28 if height >= 800 else 24
        gap = 5
        for title, items in self.button_specs:
            self.section_labels.append((title.upper(), (x0, y)))
            y += 20
            col = 0
            i = 0
            while i < len(items):
                item = items[i]
                if isinstance(item, str) and item.startswith("param:"):
                    label_rect = pygame.Rect(x0, y, full_w - 2 * 34 - 8, btn_h)
                    self.param_labels.append((item.split(":")[1], label_rect))
                    for k, button in enumerate(items[i + 1:i + 3]):
                        assert isinstance(button, Button)
                        button.rect = pygame.Rect(x0 + full_w - (2 - k) * 34 - (1 - k) * 4, y,
                                                  32, btn_h)
                        self.buttons.append(button)
                    y += btn_h + gap
                    i += 3
                    continue
                if item == "field":
                    self.episodes_field.rect = pygame.Rect(x0, y, half_w, btn_h)
                    col = 1
                    i += 1
                    continue
                assert isinstance(item, Button)
                wide = item.action in ("start_pause", "reset_all", "watch", "evaluate",
                                       "toggle_explore")
                if wide and col == 1:
                    y += btn_h + gap
                    col = 0
                w = full_w if wide else half_w
                item.rect = pygame.Rect(x0 + (half_w + 6) * col, y, w, btn_h)
                self.buttons.append(item)
                if wide or col == 1:
                    y += btn_h + gap
                    col = 0
                else:
                    col = 1
                i += 1
            if col == 1:
                y += btn_h + gap
            y += 6

    # ================================================================ actions
    def handle_action(self, action: str) -> None:
        """Perform the action bound to a button or key."""
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            raise ValueError(f"unknown action {action}")
        handler()

    def _act_start_pause(self) -> None:
        if self.mode in (TRAIN, BATCH):
            self.paused = not self.paused
            self.say("Training paused." if self.paused else "Training resumed.")
            return
        if self.mode == WATCH:
            self.trainer.abort_episode()
        self.mode = TRAIN
        self.paused = False
        self.say("Training: the mouse explores, collects rewards and penalties, and updates "
                 "its Q-table after every step.")

    def _act_pause(self) -> None:
        self.paused = not self.paused
        self.say("Paused." if self.paused else "Resumed.")

    def _act_reset_episode(self) -> None:
        self.trainer.abort_episode()
        if self.mode == WATCH:
            self.mode = IDLE
        self.step_credit = 0.0
        self.say("Current episode reset: the mouse is back at the start. "
                 "The Q-table (what it has learned) is unchanged.")

    def _act_reset_all(self) -> None:
        now = time.monotonic()
        if now >= self.confirm_reset_until:
            self.confirm_reset_until = now + 3.0
            self.say("Click 'Reset ALL learning' again within 3 s to erase the Q-table.",
                     theme.WARN)
            return
        self.confirm_reset_until = 0.0
        self.trainer.reset_learning()
        self.mode = IDLE
        self.paused = False
        self.eval_report = None
        self.eval_path = None
        self.charts.invalidate()
        self.say("All learning erased: Q-table is zero, epsilon reset, statistics cleared.",
                 theme.WARN)

    def _start_batch(self, episodes: int) -> None:
        self.trainer.abort_episode()
        self.batch_start = self.trainer.episode_count
        self.batch_target = self.trainer.episode_count + episodes
        self.mode = BATCH
        self.paused = False
        self.say(f"Fast training {episodes:,} episodes (rendering only the latest state)...")

    def _act_train_100(self) -> None:
        self._start_batch(100)

    def _act_train_1000(self) -> None:
        self._start_batch(1000)

    def _act_train_n(self) -> None:
        self.episodes_field.commit()
        self._start_batch(self.episodes_field.value)

    def _act_watch(self) -> None:
        self.trainer.abort_episode()
        self.trainer.begin_episode()
        self.mode = WATCH
        self.paused = False
        self.watch_result = None
        self.say("Watching the learned (greedy) policy. No learning happens in this mode.")

    def _act_evaluate(self) -> None:
        env = MazeEnv(self.session.maze, self.session.config.rewards,
                      self.session.config.environment)
        report = evaluate_agent(self.agent, env, 20)
        self.eval_report = report
        self.eval_path = report.example_path
        color = theme.GOOD if report.success_rate >= 0.95 else theme.WARN
        self.say(f"Evaluation (greedy, no learning, 20 runs): success "
                 f"{100 * report.success_rate:.0f}%, avg steps {fmt(report.avg_success_steps)},"
                 f" shortest possible {report.shortest_path_length}.", color)

    def _act_toggle_explore(self) -> None:
        self.agent.exploration_enabled = not self.agent.exploration_enabled
        self.say("Exploration ON: epsilon-greedy action selection." if
                 self.agent.exploration_enabled else
                 "Exploration OFF: the mouse always takes its best-known action "
                 "(it still learns while training).")

    def _act_toggle_policy(self) -> None:
        self.show_policy = not self.show_policy

    def _act_toggle_heatmap(self) -> None:
        self.show_heatmap = not self.show_heatmap

    def _act_toggle_best(self) -> None:
        self.show_best = not self.show_best

    def _act_speed_down(self) -> None:
        self.speed_index = max(0, self.speed_index - 1)

    def _act_speed_up(self) -> None:
        self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)

    def _act_alpha_down(self) -> None:
        cfg = self.agent.config
        cfg.learning_rate = round(max(0.01, cfg.learning_rate - 0.05), 3)

    def _act_alpha_up(self) -> None:
        cfg = self.agent.config
        cfg.learning_rate = round(min(1.0, cfg.learning_rate + 0.05), 3)

    def _act_gamma_down(self) -> None:
        cfg = self.agent.config
        cfg.discount = round(max(0.0, cfg.discount - 0.01), 3)

    def _act_gamma_up(self) -> None:
        cfg = self.agent.config
        cfg.discount = round(min(1.0, cfg.discount + 0.01), 3)

    def _decay_index(self) -> int:
        decay = self.agent.config.epsilon_decay
        return min(range(len(DECAYS)), key=lambda i: abs(DECAYS[i] - decay))

    def _act_decay_down(self) -> None:
        self.agent.config.epsilon_decay = DECAYS[max(0, self._decay_index() - 1)]

    def _act_decay_up(self) -> None:
        self.agent.config.epsilon_decay = DECAYS[min(len(DECAYS) - 1, self._decay_index() + 1)]

    def _act_save(self) -> None:
        try:
            path = self.session.save(self.model_path)
            self.say(f"Model saved to {path} ({self.trainer.episode_count:,} episodes).",
                     theme.GOOD)
        except OSError as exc:
            self.say(f"Could not save model: {exc}", theme.BAD)

    def _act_load(self) -> None:
        try:
            session = Session.from_model(self.model_path, copy.deepcopy(self.session.config))
        except ModelLoadError as exc:
            self.say(f"Could not load model: {exc}", theme.BAD)
            return
        self._replace_session(session)
        self.say(f"Loaded {self.model_path} — {self.trainer.episode_count:,} episodes, "
                 f"epsilon {self.agent.epsilon:.3f}. Training continues from here.", theme.GOOD)

    def _replace_session(self, session: Session) -> None:
        self.session = session
        self.renderer = MazeRenderer(session.maze)
        self.renderer.fit(self.maze_area)
        self.mode = IDLE
        self.paused = False
        self.eval_report = None
        self.eval_path = None
        self.charts.invalidate()

    def _new_maze_session(self, maze_name: str) -> Session:
        config = copy.deepcopy(self.session.config)
        config.environment.maze = maze_name
        if maze_name == "random":
            config.training.seed = self.random_maze_seed
            self.random_maze_seed += 1
        return Session(config)

    def _act_random_maze(self) -> None:
        self._replace_session(self._new_maze_session("random"))
        self.say(f"New random maze (seed {self.session.config.training.seed}). "
                 f"Learning starts from scratch.")

    def _act_default_maze(self) -> None:
        self._replace_session(self._new_maze_session("default"))
        self.say("Default maze loaded. Learning starts from scratch.")

    # ------------------------------------------------------------------ demo
    def _act_demo(self) -> None:
        if self.main_session is None:
            self.main_session = self.session
        config = copy.deepcopy(self.main_session.config)
        demo_session = Session(config, maze=self.main_session.maze)
        self._replace_session(demo_session)
        self.demo = DemoState()
        self.mode = DEMO
        self.show_policy = False
        self.trainer.begin_episode()
        self.say("Demo stage A: an untrained mouse. Its Q-table is all zeros, so every action "
                 "looks the same and it moves at random.")

    def _act_exit_demo(self) -> None:
        if self.main_session is not None:
            self._replace_session(self.main_session)
            self.main_session = None
        self.demo = None
        self.say("Demo closed. Back to your own training session.")

    def _demo_tick(self, now: float) -> None:
        demo = self.demo
        assert demo is not None
        if now < demo.hold_until:
            return
        if demo.stage == "A":
            stats = self._animate(learn=False, policy=POLICY_RANDOM)
            if stats:
                demo.stats_a = stats
                demo.path_a = list(self.trainer.last_episode_path)
                demo.stage = "B_anim"
                demo.hold_until = now + DEMO_HOLD_S
                self.say(f"Untrained mouse: {stats.steps} steps, {stats.wall_collisions} wall "
                         f"bumps, {'found' if stats.success else 'did NOT find'} the cheese. "
                         f"Next, stage B: learning from rewards and penalties.")
        elif demo.stage == "B_anim":
            stats = self._animate(learn=True, policy=POLICY_EPSILON)
            if stats:
                demo.animated_done += 1
                if demo.animated_done >= DEMO_ANIMATED_EPISODES:
                    demo.stage = "B_batch"
                    self.say(f"Fast-forwarding to {DEMO_EPISODES:,} training episodes. Watch "
                             f"the curves: reward rises, episode length falls.")
            else:
                self.say("Stage B: every step updates Q(s,a). Wall bumps and wasted steps push "
                         "values down; reaching the cheese pushes value back along the route.")
        elif demo.stage == "B_batch":
            self._run_batch(DEMO_EPISODES)
            if self.trainer.episode_count >= DEMO_EPISODES:
                demo.stage = "C"
                demo.hold_until = now + DEMO_HOLD_S
                self.show_policy = True
                self.trainer.begin_episode()
                self.say("Stage C: the learned mouse follows the highest-valued action in each "
                         "cell (exploration switched off). Arrows show its learned policy.")
        elif demo.stage == "C":
            stats = self._animate(learn=False, policy=POLICY_GREEDY)
            if stats:
                demo.stats_c = stats
                env = MazeEnv(self.session.maze, self.session.config.rewards,
                              self.session.config.environment)
                demo.report = evaluate_agent(self.agent, env, 20)
                demo.stage = "done"
                self.mode = IDLE
                self.say("Demo complete — compare the untrained and trained mouse below. "
                         "Press 'Replay demo' to run it again.", theme.GOOD)

    # ============================================================ simulation
    def _animate(self, learn: bool, policy: str) -> EpisodeStats | None:
        """Advance up to the number of steps allowed by the speed setting."""
        speed = SPEEDS[self.speed_index]
        dt = self.clock.get_time() / 1000.0
        if speed == 0:
            steps = 10_000
        else:
            self.step_credit = min(self.step_credit + dt * speed, max(1.0, speed / 10))
            steps = int(self.step_credit)
            self.step_credit -= steps
        deadline = time.perf_counter() + FRAME_BUDGET_S
        for _ in range(steps):
            transition, stats = self.trainer.step(learn=learn, policy=policy)
            self.facing = transition.action
            if stats is not None:
                return stats
            if time.perf_counter() > deadline:
                break
        return None

    def _run_batch(self, target: int) -> None:
        deadline = time.perf_counter() + FRAME_BUDGET_S
        while self.trainer.recorder is not None:   # finish an animated episode first
            self.trainer.step(learn=True, policy=POLICY_EPSILON)
        while self.trainer.episode_count < target and time.perf_counter() < deadline:
            self.trainer.train_episode()

    def update(self) -> None:
        if self.paused:
            return
        now = time.monotonic()
        if self.mode == TRAIN:
            if SPEEDS[self.speed_index] == 0:
                self._run_batch(self.trainer.episode_count + 10_000)
            else:
                self._animate(learn=True, policy=POLICY_EPSILON)
        elif self.mode == BATCH:
            self._run_batch(self.batch_target)
            if self.trainer.episode_count >= self.batch_target:
                self.mode = IDLE
                summary = self.trainer.metrics.summary(recent=100)
                self.say(f"Finished {self.batch_target - self.batch_start:,} episodes. "
                         f"Success over the last 100: "
                         f"{100 * float(summary['recent_success_rate'] or 0):.0f}%. "
                         f"Use 'Evaluate' for a learning-free test.", theme.GOOD)
        elif self.mode == WATCH:
            stats = self._animate(learn=False, policy=POLICY_GREEDY)
            if stats:
                self.mode = IDLE
                self.watch_result = stats
                bfs = shortest_path_length(self.session.maze)
                if stats.success:
                    self.say(f"Trained mouse reached the cheese in {stats.steps} steps "
                             f"(shortest possible: {bfs}).", theme.GOOD)
                else:
                    self.say(f"Trained mouse did not reach the cheese ({stats.reason}). "
                             f"It needs more training.", theme.WARN)
        elif self.mode == DEMO and self.demo is not None:
            self._demo_tick(now)

    # ================================================================ events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1100, event.w), max(720, event.h)),
                                                  pygame.RESIZABLE)
            self.layout()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.episodes_field.rect.collidepoint(event.pos):
                self.episodes_field.focus()
                return
            self.episodes_field.commit()
            for button in self.buttons:
                if button.rect.collidepoint(event.pos) and button.is_enabled():
                    self.handle_action(button.action)
                    break
        elif event.type == pygame.KEYDOWN:
            if self.episodes_field.handle_key(event):
                return
            keymap = {pygame.K_SPACE: "start_pause", pygame.K_r: "reset_episode",
                      pygame.K_w: "watch", pygame.K_p: "toggle_policy",
                      pygame.K_h: "toggle_heatmap", pygame.K_e: "toggle_explore",
                      pygame.K_b: "toggle_best", pygame.K_1: "train_100",
                      pygame.K_2: "train_1000", pygame.K_v: "evaluate",
                      pygame.K_d: "demo", pygame.K_s: "save", pygame.K_l: "load",
                      pygame.K_EQUALS: "speed_up", pygame.K_PLUS: "speed_up",
                      pygame.K_KP_PLUS: "speed_up", pygame.K_MINUS: "speed_down",
                      pygame.K_KP_MINUS: "speed_down"}
            if event.key == pygame.K_ESCAPE:
                if self.demo is not None:
                    self.handle_action("exit_demo")
                else:
                    self.running = False
            elif event.key in keymap:
                action = keymap[event.key]
                button = next((b for b in self.buttons if b.action == action), None)
                if button is None or button.is_enabled():
                    self.handle_action(action)

    # ================================================================ drawing
    def current_path(self) -> list[State]:
        recorder = self.trainer.recorder
        return recorder.path if recorder is not None else self.trainer.last_episode_path

    def current_visited(self) -> set[State]:
        recorder = self.trainer.recorder
        if recorder is not None:
            return recorder.visited
        return set(self.trainer.last_episode_path)

    def draw(self) -> None:
        screen = self.screen
        screen.fill(theme.BACKGROUND)
        self._draw_left()
        self._draw_header()
        self._draw_maze()
        self._draw_narrative()
        self._draw_stats()
        last_wall = self.trainer.last_wall_key
        history = list(self.trainer.wall_history.get(last_wall, [])) if last_wall else None
        best_at_wall = None
        if last_wall:
            values = self.agent.action_values(last_wall[0])
            best_at_wall = (int(values.argmax()), float(values.max()))
        draw_q_update(screen, self.q_rect, self.trainer.last_update, history, last_wall,
                      best_at_wall)
        self.charts.draw(screen, self.chart_rect, self.trainer.metrics)
        pygame.display.flip()

    def _draw_left(self) -> None:
        screen = self.screen
        pygame.draw.rect(screen, theme.PANEL, self.left, border_radius=8)
        for text, pos in self.section_labels:
            screen.blit(theme.font(17, bold=True).render(text, True, theme.HEADER), pos)
        values = {
            "speed": f"Speed: {self.speed_label()}",
            "alpha": f"Learning rate α: {self.agent.config.learning_rate:.2f}",
            "gamma": f"Discount γ: {self.agent.config.discount:.2f}",
            "decay": f"ε decay/episode: {self.agent.config.epsilon_decay:g}",
        }
        for name, rect in self.param_labels:
            label = theme.font(18).render(values[name], True, theme.TEXT)
            screen.blit(label, label.get_rect(midleft=(rect.x + 2, rect.centery)))
        mouse = pygame.mouse.get_pos()
        f = theme.font(18)
        for button in self.buttons:
            button.draw(screen, f, button.rect.collidepoint(mouse))
        self.episodes_field.draw(screen, f)
        hint = "Keys: Space train · W watch · P arrows · H heat · E explore · D demo"
        draw_paragraph(screen, hint, self.left.x + 10, self.left.bottom - 44,
                       self.left.width - 20, size=15, color=theme.TEXT_MUTED)

    def _draw_header(self) -> None:
        mode = {IDLE: "Idle", TRAIN: "Training (animated)", BATCH: "Fast training",
                WATCH: "Watching learned policy", DEMO: "Demonstration"}[self.mode]
        if self.demo is not None:
            mode += {"A": " — Stage A: untrained", "B_anim": " — Stage B: training",
                     "B_batch": " — Stage B: training (fast)",
                     "C": " — Stage C: learned", "done": " — complete"}[self.demo.stage]
        if self.paused:
            mode += "  [PAUSED]"
        title = theme.font(30, bold=True).render("MazeMouse", True, theme.TEXT)
        self.screen.blit(title, (self.header_rect.x, self.header_rect.y + 6))
        label = theme.font(22).render(mode, True, theme.WARN if self.paused else theme.ACCENT)
        self.screen.blit(label, (self.header_rect.x + title.get_width() + 20,
                                 self.header_rect.y + 10))

    def _draw_maze(self) -> None:
        screen = self.screen
        r = self.renderer
        r.fit(self.maze_area)
        r.draw_grid(screen)
        if self.show_heatmap:
            r.draw_heatmap(screen, self.agent)
        if self.show_visited and not self.show_heatmap:
            r.draw_visited(screen, self.current_visited())
        r.draw_special_cells(screen)
        if self.show_best and self.trainer.best_path:
            r.draw_route(screen, self.trainer.best_path, theme.BEST_ROUTE,
                         max(3, r.cell // 8), dashed=True)
        if self.show_policy:
            r.draw_policy(screen, self.agent)
        path = self.current_path()
        r.draw_route(screen, path[-400:], theme.ROUTE, max(2, r.cell // 12))
        transition = self.trainer.last_transition
        bumped = bool(transition and transition.info.hit_wall and self.trainer.recorder)
        position = self.trainer.state if self.trainer.state is not None else self.session.maze.start
        r.draw_mouse(screen, position, self.facing, bumped)
        legend = [("current route", theme.ROUTE), ("best route found", theme.BEST_ROUTE),
                  ("visited", theme.VISITED), ("trap", theme.TRAP)]
        x = self.maze_area.x
        y = self.maze_area.bottom - 2
        for text, color in legend:
            pygame.draw.rect(screen, color, (x, y - 12, 14, 10))
            label = theme.font(16).render(text, True, theme.TEXT_MUTED)
            screen.blit(label, (x + 18, y - 14))
            x += label.get_width() + 34

    def _draw_narrative(self) -> None:
        rect = self.narrative_rect
        y = draw_panel(self.screen, rect, "What is happening")
        x, width = rect.x + 10, rect.width - 20
        y = draw_paragraph(self.screen, self.message, x, y, width, size=19,
                           color=self.message_color, max_y=rect.bottom - 4)
        demo = self.demo
        if demo is not None and demo.stage == "done" and demo.stats_a and demo.stats_c:
            a, c = demo.stats_a, demo.stats_c
            rep = demo.report
            rows = [
                ("Untrained: steps / walls / reward",
                 f"{a.steps} / {a.wall_collisions} / {a.total_reward:+.1f}"
                 + ("" if a.success else " (failed)"), theme.BAD),
                ("Trained: steps / walls / reward",
                 f"{c.steps} / {c.wall_collisions} / {c.total_reward:+.1f}"
                 + ("" if c.success else " (failed)"), theme.GOOD),
                ("Greedy eval success (20 runs)",
                 f"{100 * rep.success_rate:.0f}%" if rep else "—", None),
                ("Shortest possible path (BFS)",
                 str(rep.shortest_path_length) if rep else "—", None),
            ]
            draw_rows(self.screen, rect, y + 2, rows, columns=2, size=17)
        elif self.eval_report is not None and demo is None:
            rep = self.eval_report
            rows = [("success / failure", f"{100 * rep.success_rate:.0f}% / "
                     f"{100 * rep.failure_rate:.0f}%", None),
                    ("avg reward", f"{rep.avg_reward:+.1f}", None),
                    ("avg successful steps", fmt(rep.avg_success_steps), None),
                    ("BFS shortest / gap", f"{rep.shortest_path_length} / "
                     f"{fmt(rep.gap_to_optimal, 1)}", None)]
            draw_rows(self.screen, rect, y + 2, rows, columns=2, size=17)

    def _draw_stats(self) -> None:
        rect = self.stats_rect
        y = draw_panel(self.screen, rect, "Statistics")
        summary = self.trainer.metrics.summary(recent=100)
        recorder = self.trainer.recorder
        cur_reward = recorder.total_reward if recorder else 0.0
        cur_steps = recorder.steps if recorder else 0
        cur_cells = len(recorder.visited) if recorder else 0
        if self.mode == BATCH:
            done = self.trainer.episode_count - self.batch_start
            total = self.batch_target - self.batch_start
            progress = f"{done:,}/{total:,}"
        else:
            progress = f"{self.trainer.episode_count:,} total"
        rows = [
            ("Episode", f"{self.trainer.episode_count + (1 if recorder else 0):,}", None),
            ("Progress", progress, None),
            ("Epsilon ε", f"{self.agent.epsilon:.3f}", None),
            ("Episode reward", f"{cur_reward:+.1f}", None),
            ("Episode length", f"{cur_steps}", None),
            ("Cells visited", f"{cur_cells}", None),
            ("Success rate", f"{100 * float(summary['success_rate'] or 0):.1f}%", None),
            ("Last 100", f"{100 * float(summary['recent_success_rate'] or 0):.0f}%",
             theme.GOOD if (summary['recent_success_rate'] or 0) > 0.9 else None),
            ("Avg steps (wins)", fmt(summary["avg_success_steps"]), None),
            ("Best win (steps)", str(summary["best_success_steps"] or "—"), None),
            ("Wall bumps", f"{summary['total_wall_collisions']:,}", None),
            ("Trap hits", f"{summary['total_trap_encounters']:,}", None),
        ]
        y = draw_rows(self.screen, rect, y, rows, columns=2, size=18)
        bfs = shortest_path_length(self.session.maze)
        comps = recorder.components if recorder else {}
        breakdown = "  ".join(f"{k} {v:+.1f}" for k, v in comps.items()) or "—"
        draw_paragraph(self.screen, f"Reward breakdown (this episode): {breakdown}",
                       rect.x + 10, y + 4, rect.width - 20, size=17, color=theme.TEXT_MUTED)
        draw_paragraph(self.screen, f"Benchmark (BFS, not given to the mouse): {bfs} steps",
                       rect.x + 10, rect.bottom - 22, rect.width - 20, size=17,
                       color=theme.TEXT_MUTED)

    # ================================================================== loop
    def run(self, max_frames: int | None = None, fps: int = 60) -> None:
        frames = 0
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            self.clock.tick(fps)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break


def run_app(session: Session, model_path: Path, max_frames: int | None = None) -> int:
    app = MazeMouseApp(session, model_path)
    try:
        app.run(max_frames=max_frames)
    finally:
        pygame.quit()
    return 0
