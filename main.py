"""Command-line entry point."""

from __future__ import annotations

import argparse

from evaluate import run_evaluate
from train import run_train
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MRA-RLEC reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="config.yaml")
    train_parser.add_argument("--resume", default=None)

    train_vector_parser = subparsers.add_parser("train-vector")
    train_vector_parser.add_argument("--config", default="config_vector_server.yaml")
    train_vector_parser.add_argument("--resume", default=None)

    pretrain_parser = subparsers.add_parser("pretrain")
    pretrain_parser.add_argument("--config", default="config.yaml")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", default="config.yaml")
    eval_parser.add_argument("--episodes", type=int, default=3)
    eval_parser.add_argument("--checkpoint", default=None)

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--config", default="config.yaml")
    metrics_parser.add_argument("--policy", choices=["guide", "checkpoint", "controller"], default="guide")
    metrics_parser.add_argument("--checkpoint", default=None)
    metrics_parser.add_argument("--episodes", type=int, default=10)
    metrics_parser.add_argument("--output-dir", default="results/metrics")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "train":
        run_train(config, resume=args.resume)
    elif args.command == "train-vector":
        from algorithms.vector_mappo import VectorMAPPOTrainer

        trainer = VectorMAPPOTrainer(config)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()
        if trainer.last_checkpoint_path:
            print(f"checkpoint={trainer.last_checkpoint_path}")
        if trainer.best_checkpoint_path:
            print(f"best_checkpoint={trainer.best_checkpoint_path}")
    elif args.command == "pretrain":
        from pretrain import run_pretrain

        run_pretrain(config)
    elif args.command == "evaluate":
        if args.checkpoint:
            from algorithms.mra_rlec import TorchMRARLECTrainer
            from evaluate import run_evaluate_actor

            trainer = TorchMRARLECTrainer(config)
            trainer.load_checkpoint(args.checkpoint)
            result = run_evaluate_actor(config, trainer.actor, episodes=args.episodes)
            print(
                f"episodes={int(result['episodes'])} mean_reward={result['mean_reward']:.3f} "
                f"success_rate={result['success_rate']:.3f} collision_rate={result['collision_rate']:.3f}"
            )
        else:
            run_evaluate(config, episodes=args.episodes)
    elif args.command == "metrics":
        from experiments.run_metrics import run as run_metrics

        run_metrics(args.config, args.policy, args.checkpoint, args.episodes, args.output_dir)


if __name__ == "__main__":
    main()
