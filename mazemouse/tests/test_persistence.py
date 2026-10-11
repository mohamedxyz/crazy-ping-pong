"""Saving, loading and continuing training from model files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent.persistence import ModelLoadError, load_model
from config import AppConfig, ConfigError, load_config
from session import Session


def small_config(seed: int = 5) -> AppConfig:
    config = AppConfig()
    config.environment.maze = "small"
    config.environment.max_steps = 100
    config.agent.epsilon_decay = 0.99
    config.training.seed = seed
    return config


def test_save_and_load_roundtrip(tmp_path):
    session = Session(small_config())
    session.trainer.train(60)
    path = session.save(tmp_path / "models" / "m.json")
    loaded = load_model(path)
    assert np.array_equal(loaded.agent.q_table, session.agent.q_table)
    assert np.array_equal(loaded.agent.visit_counts, session.agent.visit_counts)
    assert loaded.agent.epsilon == session.agent.epsilon
    assert loaded.episode_count == 60
    assert loaded.maze.to_layout() == session.maze.to_layout()
    assert loaded.metrics.to_list() == session.trainer.metrics.to_list()
    assert loaded.agent.config == session.agent.config


def test_continue_training_matches_uninterrupted_run(tmp_path):
    uninterrupted = Session(small_config())
    uninterrupted.trainer.train(80)

    first = Session(small_config())
    first.trainer.train(40)
    first.save(tmp_path / "m.json")
    resumed = Session.from_model(tmp_path / "m.json")
    assert resumed.trainer.episode_count == 40
    resumed.trainer.train(40)

    assert resumed.trainer.episode_count == 80
    assert np.allclose(resumed.agent.q_table, uninterrupted.agent.q_table)
    assert resumed.agent.epsilon == pytest.approx(uninterrupted.agent.epsilon)


def test_load_preserves_values_not_reinitialised(tmp_path):
    session = Session(small_config())
    session.trainer.train(30)
    session.save(tmp_path / "m.json")
    restored = Session.from_model(tmp_path / "m.json")
    assert restored.agent.q_table.any()


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(format="other"), "not a MazeMouse"),
    (lambda d: d.update(version=99), "unsupported model version"),
    (lambda d: d.update(q_table=[[0.0]]), "shape"),
    (lambda d: d["q_table"][1][1].__setitem__(0, float("nan")), "NaN"),
    (lambda d: d["agent"].update(epsilon=3.0), "epsilon"),
    (lambda d: d["maze"].update(layout=["###"]), "invalid"),
    (lambda d: d.pop("rewards"), "invalid"),
    (lambda d: d["agent"]["config"].update(learning_rate=-1), "invalid"),
])
def test_invalid_models_rejected(tmp_path, mutate, message):
    session = Session(small_config())
    session.trainer.train(3)
    path = session.save(tmp_path / "m.json")
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ModelLoadError, match=message):
        load_model(path)


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_model(bad)


def test_config_file_loading_and_validation(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"rewards": {"goal": 50}, "agent": {"discount": 0.9}}))
    config = load_config(path)
    assert config.rewards.goal == 50 and config.agent.discount == 0.9
    assert config.rewards.wall == -2.0          # untouched defaults
    for bad in ({"rewards": {"step": 1.0}}, {"agent": {"bogus": 1}}, {"other": {}},
                {"environment": {"max_steps": 0}}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ConfigError):
            load_config(path)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")


def test_shipped_config_file_is_valid():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parent.parent / "config.json")
    assert config.agent.learning_rate == 0.1
