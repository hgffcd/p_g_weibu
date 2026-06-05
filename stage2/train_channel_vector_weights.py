"""Fine-tune Stage-2 channel residual policy from weights only.

This entry point is intentionally separate from ``train_channel_vector.py``.
The normal trainer restores the checkpoint update counter; that is correct for
true continuation, but it skips training when a 3000-update checkpoint is loaded
with a new config that also has ``vector_training.updates: 3000``. This script
loads actor/critic weights only and starts the hard-scenario run from update 1.
"""

from __future__ import annotations

import argparse

import torch

from stage2.channel_trainer import Stage2ChannelVectorMAPPOTrainer
from utils.config import load_config


def load_weights_only(trainer: Stage2ChannelVectorMAPPOTrainer, checkpoint_path: str) -> None:
    """Load network weights without inheriting optimizer or update progress."""
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.actor.load_state_dict(checkpoint["actor"])
    trainer.critic.load_state_dict(checkpoint["critic"])
    trainer.start_update = 1
    trainer.completed_episodes = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 channel-guide weights-only fine-tuning")
    parser.add_argument("--config", default="config_stage2_channel_hard_v1_finetune.yaml")
    parser.add_argument("--weights", required=True, help="Checkpoint used as actor/critic initialization")
    args = parser.parse_args()

    trainer = Stage2ChannelVectorMAPPOTrainer(load_config(args.config))
    load_weights_only(trainer, args.weights)
    trainer.train()
    if trainer.last_checkpoint_path:
        print(f"checkpoint={trainer.last_checkpoint_path}")
    if trainer.best_checkpoint_path:
        print(f"best_checkpoint={trainer.best_checkpoint_path}")


if __name__ == "__main__":
    main()
