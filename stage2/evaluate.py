"""Stage-2 evaluation for fixed and randomized planar scenarios."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from agents.apf_guide import APFGuidePolicy
from agents.encirclement_controller import PlanarEncirclementController
from stage2.channel_guide import Stage2ChannelGuidePolicy
from stage2.envs import Stage2EncirclementEnv
from utils.config import load_config
from utils.metrics import (
    EpisodeMetric,
    aggregate_episodes,
    communication_meta_messages,
    step_metric,
    summarize_episode,
)


def configure_scenario(config: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Return a copied config with fixed or randomized reset behavior."""
    cfg = copy.deepcopy(config)
    if scenario == "fixed":
        stage2 = cfg.setdefault("stage2", {})
        for key in ("target_randomization", "pursuer_randomization", "obstacle_randomization"):
            stage2.setdefault(key, {})["enabled"] = False
    return cfg


def checkpoint_actor(config: Dict[str, Any], checkpoint: str):
    """Load an actor through the existing trainer without modifying it."""
    from algorithms.mra_rlec import TorchMRARLECTrainer

    trainer = TorchMRARLECTrainer(config)
    trainer.load_checkpoint(checkpoint)
    trainer.actor.eval()
    return trainer.actor, trainer.device


def run_episode(
    config: Dict[str, Any],
    policy: str,
    checkpoint: str | None,
    seed: int,
) -> Tuple[EpisodeMetric, Dict[str, Any]]:
    env = Stage2EncirclementEnv(config)
    guide = (
        Stage2ChannelGuidePolicy(config)
        if config.get("channel_guide", {}).get("enabled", False)
        else APFGuidePolicy(config)
    )
    controller = PlanarEncirclementController(config) if policy == "controller" else None
    actor = None
    device = None
    if policy == "checkpoint":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for checkpoint evaluation")
        actor, device = checkpoint_actor(config, checkpoint)

    obs, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(env.num_pursuers, device) if actor is not None else None
    step_metrics = []
    info: Dict[str, Any] = {"success": False, "collision": False, "timeout": False}
    for step in range(int(config["environment"]["max_steps"])):
        if policy == "guide":
            action = guide.act(env)
        elif policy == "controller":
            if controller is None:
                raise RuntimeError("Controller was not initialized")
            action = controller.act(env)
        elif policy == "checkpoint":
            import torch

            assert actor is not None and device is not None and hidden is not None
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, hidden = actor.act(obs_tensor, hidden, deterministic=True)
            actor_action = action_tensor.detach().cpu().numpy()
            policy_mode = str(config.get("training", {}).get("policy_mode", "direct")).lower()
            if policy_mode == "residual":
                residual_scale = float(config.get("training", {}).get("residual_scale", 1.0))
                action = guide.act(env) + residual_scale * actor_action
            else:
                action = actor_action
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, done, info = env.step(action)
        step_metrics.append(step_metric(env, reward))
        if done:
            break

    metric = summarize_episode(step_metrics, info, step + 1, float(config["environment"]["dt"]))
    reset_info = env.last_reset_info
    return metric, {
        "target_x": float(reset_info.get("target_pos", env.target_pos)[0]),
        "target_y": float(reset_info.get("target_pos", env.target_pos)[1]),
        "reset_attempt": int(reset_info.get("attempt", 1)),
        "randomized": bool(reset_info.get("randomized", False)),
    }


def write_outputs(summary: Dict[str, float], rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    fieldnames = [
        "episode",
        "seed",
        "randomized",
        "target_x",
        "target_y",
        "reset_attempt",
        "success",
        "collision",
        "timeout",
        "steps",
        "encirclement_time",
        "average_step_reward",
        "danger_rate",
        "final_distance_error",
        "final_angle_error",
        "min_obstacle_clearance",
        "min_pursuer_clearance",
    ]
    with open(output_dir / "episodes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(
    config_path: str,
    scenario: str,
    policy: str,
    checkpoint: str | None,
    episodes: int,
    output_dir: str,
    seed: int,
) -> Dict[str, float]:
    config = configure_scenario(load_config(config_path), scenario)
    metrics: List[EpisodeMetric] = []
    rows: List[Dict[str, Any]] = []
    for episode in range(int(episodes)):
        metric, reset_row = run_episode(config, policy, checkpoint, seed + episode)
        metrics.append(metric)
        rows.append({
            "episode": episode,
            "seed": seed + episode,
            **reset_row,
            "success": metric.success,
            "collision": metric.collision,
            "timeout": metric.timeout,
            "steps": metric.steps,
            "encirclement_time": metric.encirclement_time,
            "average_step_reward": metric.average_step_reward,
            "danger_rate": metric.danger_rate,
            "final_distance_error": metric.final_distance_error,
            "final_angle_error": metric.final_angle_error,
            "min_obstacle_clearance": metric.min_obstacle_clearance,
            "min_pursuer_clearance": metric.min_pursuer_clearance,
        })
    summary = aggregate_episodes(metrics)
    summary.update(communication_meta_messages(int(config["environment"]["num_pursuers"])))
    summary["scenario"] = scenario  # type: ignore[assignment]
    summary["policy"] = policy  # type: ignore[assignment]
    write_outputs(summary, rows, Path(output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 fixed/randomized evaluation")
    parser.add_argument("--config", default="config_stage2_moving.yaml")
    parser.add_argument("--scenario", choices=["fixed", "randomized"], default="fixed")
    parser.add_argument("--policy", choices=["guide", "controller", "checkpoint"], default="guide")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", default="results_stage2/eval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.scenario, args.policy, args.checkpoint, args.episodes, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
