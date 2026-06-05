"""Command-line entry for Stage-2 randomized vector training."""

from __future__ import annotations

import argparse

from stage2.trainer import Stage2VectorMAPPOTrainer
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 MRA-RLEC vector training")
    parser.add_argument("--config", default="config_stage2.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Stage2VectorMAPPOTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    if trainer.last_checkpoint_path:
        print(f"checkpoint={trainer.last_checkpoint_path}")
    if trainer.best_checkpoint_path:
        print(f"best_checkpoint={trainer.best_checkpoint_path}")


if __name__ == "__main__":
    main()
