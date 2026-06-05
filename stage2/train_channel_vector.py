"""Command-line entry for Stage-2 channel-guide residual training."""

from __future__ import annotations

import argparse

from stage2.channel_trainer import Stage2ChannelVectorMAPPOTrainer
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 channel-guide vector training")
    parser.add_argument("--config", default="config_stage2_channel_probe.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Stage2ChannelVectorMAPPOTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    if trainer.last_checkpoint_path:
        print(f"checkpoint={trainer.last_checkpoint_path}")
    if trainer.best_checkpoint_path:
        print(f"best_checkpoint={trainer.best_checkpoint_path}")


if __name__ == "__main__":
    main()
