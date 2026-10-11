"""The GUI controls perform their advertised actions (runs with a dummy display)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from config import AppConfig  # noqa: E402
from session import Session  # noqa: E402
from visualization import app as app_module  # noqa: E402


@pytest.fixture
def app(tmp_path):
    config = AppConfig()
    config.environment.maze = "small"
    config.environment.max_steps = 100
    application = app_module.MazeMouseApp(Session(config), tmp_path / "model.json",
                                          size=(1200, 800))
    yield application
    pygame.quit()


def run_until(app, predicate, limit=5000):
    for _ in range(limit):
        if predicate():
            return
        app.update()
        app.clock.tick(1000)
    raise AssertionError("condition not reached")


def test_every_button_has_a_handler(app):
    for button in app.buttons:
        assert hasattr(app, f"_act_{button.action}"), button.action


def test_train_buttons_run_exact_episode_counts(app):
    app.handle_action("train_100")
    run_until(app, lambda: app.mode == app_module.IDLE)
    assert app.trainer.episode_count == 100
    app.episodes_field.value = 37
    app.handle_action("train_n")
    run_until(app, lambda: app.mode == app_module.IDLE)
    assert app.trainer.episode_count == 137


def test_start_pause_resume(app):
    app.handle_action("start_pause")
    assert app.mode == app_module.TRAIN and not app.paused
    app.speed_index = len(app_module.SPEEDS) - 1   # turbo
    app.update()
    episodes = app.trainer.episode_count
    assert episodes > 0
    app.handle_action("start_pause")
    assert app.paused
    app.update()
    assert app.trainer.episode_count == episodes   # paused: nothing happens
    app.handle_action("start_pause")
    assert not app.paused


def test_reset_episode_keeps_learning_reset_all_erases(app):
    app.handle_action("train_100")
    run_until(app, lambda: app.mode == app_module.IDLE)
    app.trainer.begin_episode()
    app.trainer.step()
    q = app.agent.q_table.copy()
    app.handle_action("reset_episode")
    assert app.trainer.recorder is None
    assert np.array_equal(q, app.agent.q_table)
    app.handle_action("reset_all")           # first click only asks for confirmation
    assert app.agent.q_table.any()
    app.handle_action("reset_all")
    assert not app.agent.q_table.any() and app.trainer.episode_count == 0


def test_watch_and_evaluate_do_not_learn(app):
    app.handle_action("train_1000")
    run_until(app, lambda: app.mode == app_module.IDLE)
    q = app.agent.q_table.copy()
    app.speed_index = len(app_module.SPEEDS) - 1
    app.handle_action("watch")
    run_until(app, lambda: app.mode == app_module.IDLE)
    assert app.watch_result is not None and app.watch_result.success
    app.handle_action("evaluate")
    assert app.eval_report.success_rate == 1.0
    assert np.array_equal(q, app.agent.q_table)
    assert app.trainer.episode_count == 1000


def test_toggles_and_parameters(app):
    app.handle_action("toggle_explore")
    assert not app.agent.exploration_enabled
    for name in ("toggle_policy", "toggle_heatmap"):
        app.handle_action(name)
    assert app.show_policy and app.show_heatmap
    alpha, gamma, decay = (app.agent.config.learning_rate, app.agent.config.discount,
                           app.agent.config.epsilon_decay)
    app.handle_action("alpha_up")
    app.handle_action("gamma_down")
    app.handle_action("decay_up")
    assert app.agent.config.learning_rate > alpha
    assert app.agent.config.discount < gamma
    assert app.agent.config.epsilon_decay > decay
    speed = app.speed_index
    app.handle_action("speed_up")
    assert app.speed_index == speed + 1
    app.draw()   # renders policy arrows + heatmap without errors


def test_save_then_load_restores_model(app):
    app.handle_action("train_100")
    run_until(app, lambda: app.mode == app_module.IDLE)
    q = app.agent.q_table.copy()
    app.handle_action("save")
    app.confirm_reset_until = 1e18
    app.handle_action("reset_all")
    assert not app.agent.q_table.any()
    app.handle_action("load")
    assert np.array_equal(q, app.agent.q_table)
    assert app.trainer.episode_count == 100


def test_demo_runs_all_stages_and_restores_session(app, monkeypatch):
    monkeypatch.setattr(app_module, "DEMO_HOLD_S", 0.0)
    monkeypatch.setattr(app_module, "DEMO_EPISODES", 300)
    app.handle_action("train_100")
    run_until(app, lambda: app.mode == app_module.IDLE)
    main_q = app.agent.q_table.copy()
    app.speed_index = len(app_module.SPEEDS) - 1
    app.handle_action("demo")
    assert not app.agent.q_table.any()           # stage A starts untrained
    run_until(app, lambda: app.demo.stage == "done", limit=20000)
    assert app.demo.stats_a is not None and app.demo.stats_c is not None
    assert app.demo.stats_c.success
    app.draw()
    app.handle_action("exit_demo")
    assert np.array_equal(main_q, app.agent.q_table)
