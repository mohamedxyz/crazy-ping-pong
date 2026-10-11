"""MazeMouse entry point.

Examples::

    python main.py                                   # interactive GUI
    python main.py gui --load models/mouse.json      # GUI with a saved model
    python main.py train --episodes 5000 --save models/mouse.json --charts logs/charts.png
    python main.py evaluate --load models/mouse.json --episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import AppConfig, ConfigError, load_config
from environment.maze import MazeError
from agent.persistence import ModelLoadError
from session import Session
from training.evaluation import evaluate_agent

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "mouse.json"
LOG_DIR = PROJECT_DIR / "logs"

logger = logging.getLogger("mazemouse")


def setup_logging(verbose: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(LOG_DIR / "mazemouse.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mazemouse",
        description="A mouse that learns to find the cheese with tabular Q-learning.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="JSON config file (see config.json)")
    common.add_argument("--maze", help="'default', 'small', 'random' or a JSON layout file")
    common.add_argument("--seed", type=int, help="random seed (use -1 for no seed)")
    common.add_argument("--max-steps", type=int, help="step limit per episode")
    common.add_argument("--load", help="load a saved model before running")
    common.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", parents=[common], help="interactive Pygame laboratory (default)")

    train = sub.add_parser("train", parents=[common], help="headless training")
    train.add_argument("--episodes", type=int, help="number of training episodes")
    train.add_argument("--alpha", type=float, help="learning rate")
    train.add_argument("--gamma", type=float, help="discount factor")
    train.add_argument("--epsilon-decay", type=float, help="per-episode epsilon decay")
    train.add_argument("--save", default=str(DEFAULT_MODEL), help="where to save the model")
    train.add_argument("--charts", help="save training charts to this PNG file")
    train.add_argument("--log-every", type=int, default=500)
    train.add_argument("--eval-episodes", type=int, help="evaluation episodes after training")

    evaluate = sub.add_parser("evaluate", parents=[common],
                              help="evaluate a saved model without learning")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.maze:
        config.environment.maze = args.maze
    if args.seed is not None:
        config.training.seed = None if args.seed < 0 else args.seed
    if args.max_steps is not None:
        config.environment.max_steps = args.max_steps
    for attr, section, key in (("episodes", "training", "episodes"),
                               ("alpha", "agent", "learning_rate"),
                               ("gamma", "agent", "discount"),
                               ("epsilon_decay", "agent", "epsilon_decay"),
                               ("eval_episodes", "training", "eval_episodes")):
        value = getattr(args, attr, None)
        if value is not None and not (attr == "episodes" and args.command == "evaluate"):
            setattr(getattr(config, section), key, value)
    config.validate()
    return config


def make_session(config: AppConfig, args: argparse.Namespace) -> Session:
    if args.load:
        session = Session.from_model(args.load, config)
        # Explicit command-line hyper-parameters override the saved ones.
        apply_overrides(session.config, args)
        session.env.config.max_steps = session.config.environment.max_steps
        logger.info("Continuing from %s at episode %d (epsilon %.3f)",
                    args.load, session.trainer.episode_count, session.agent.epsilon)
        return session
    return Session(config)


def cmd_train(config: AppConfig, args: argparse.Namespace) -> int:
    session = make_session(config, args)
    episodes = config.training.episodes
    logger.info("Training for %d episodes on a %dx%d maze (seed=%s, alpha=%.3f, "
                "gamma=%.3f, decay=%.4f)", episodes, session.maze.rows, session.maze.cols,
                config.training.seed, session.agent.config.learning_rate,
                session.agent.config.discount, session.agent.config.epsilon_decay)
    started = time.perf_counter()
    session.trainer.train(episodes, log_every=args.log_every)
    elapsed = time.perf_counter() - started
    summary = session.trainer.metrics.summary()
    logger.info("Training finished in %.1fs. Overall training success rate %.1f%%, "
                "best successful episode %s steps.", elapsed,
                100 * float(summary["success_rate"] or 0), summary["best_success_steps"])
    path = session.save(args.save)
    print(f"Model saved to {path}")
    if args.charts:
        from visualization.charts import save_training_charts
        print(f"Charts saved to {save_training_charts(session.trainer.metrics, args.charts)}")
    report = evaluate_agent(session.agent, session.env, config.training.eval_episodes)
    print("\nGreedy evaluation (no learning):")
    print(report.format())
    return 0


def cmd_evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    if not args.load:
        args.load = str(DEFAULT_MODEL)
    session = make_session(config, args)
    report = evaluate_agent(session.agent, session.env, args.episodes)
    if args.json:
        data = report.to_dict()
        data.pop("example_path")
        print(json.dumps(data, indent=2))
    else:
        print(f"Model: {args.load}  (trained episodes: {session.trainer.episode_count})")
        print(report.format())
    return 0


def cmd_gui(config: AppConfig, args: argparse.Namespace) -> int:
    from visualization.app import run_app
    session = make_session(config, args)
    return run_app(session, model_path=Path(args.load) if args.load else DEFAULT_MODEL)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["gui", *(argv if argv is not None else sys.argv[1:])])
    setup_logging(args.verbose)
    try:
        config = apply_overrides(load_config(args.config), args)
        handler = {"train": cmd_train, "evaluate": cmd_evaluate, "gui": cmd_gui}[args.command]
        return handler(config, args)
    except (ConfigError, MazeError, ModelLoadError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
