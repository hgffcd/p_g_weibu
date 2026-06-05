"""Training entry point for the minimal MRA-RLEC reproduction."""

from __future__ import annotations

from typing import Any, Dict, List

from algorithms.mra_rlec import MRARLECTrainer, TorchMRARLECTrainer


def run_train(config: Dict[str, Any], resume: str | None = None) -> List[Dict[str, Any]]:
    backend = str(config.get("training", {}).get("backend", "torch"))
    trainer = TorchMRARLECTrainer(config) if backend == "torch" else MRARLECTrainer(config)
    if resume:
        if not hasattr(trainer, "load_checkpoint"):
            raise ValueError("Resume is only supported for the torch trainer")
        trainer.load_checkpoint(resume)
    history = trainer.train()
    if len(history) <= 20:
        rows_to_print = history
    else:
        rows_to_print = history[:5] + history[-5:]
        print(f"showing {len(rows_to_print)} of {len(history)} episodes; full log is in logs/train_history.csv")
    for item in rows_to_print:
        losses = ""
        if "actor_loss" in item:
            losses = f" actor_loss={item['actor_loss']:.4f} critic_loss={item['critic_loss']:.4f}"
            if "bc_loss" in item:
                losses += f" bc_loss={item['bc_loss']:.4f}"
        print(
            f"episode={item['episode']} reward={item['reward']:.3f} "
            f"steps={item['steps']} guide_steps={item['guide_steps']} "
            f"success={item['success']} collision={item['collision']} timeout={item['timeout']}"
            f"{losses}"
        )
    if hasattr(trainer, "last_checkpoint_path") and trainer.last_checkpoint_path:
        print(f"checkpoint={trainer.last_checkpoint_path}")
    return history
