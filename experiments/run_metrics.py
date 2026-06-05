"""Run paper-style evaluation metrics for trained or guide policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from agents.apf_guide import APFGuidePolicy
from agents.encirclement_controller import PlanarEncirclementController
from algorithms.mra_rlec import TorchMRARLECTrainer
from envs import EncirclementEnv
from utils.config import load_config
from utils.metrics import (
    EpisodeMetric,
    aggregate_episodes,
    communication_meta_messages,
    step_metric,
    summarize_episode,
)


def evaluate_guide(config: Dict[str, Any], episodes: int) -> List[EpisodeMetric]:
    env = EncirclementEnv(config)
    guide = APFGuidePolicy(config)
    metrics: List[EpisodeMetric] = []
    for episode in range(episodes):
        env.reset(seed=episode)
        step_metrics = []
        info = {"success": False, "collision": False, "timeout": False}
        for step in range(int(config["environment"]["max_steps"])):
            actions = guide.act(env)
            _, reward, done, info = env.step(actions)
            step_metrics.append(step_metric(env, reward))
            if done:
                break
        metrics.append(summarize_episode(step_metrics, info, step + 1, float(config["environment"]["dt"])))
    return metrics


def evaluate_controller(config: Dict[str, Any], episodes: int) -> List[EpisodeMetric]:
    env = EncirclementEnv(config)
    controller = PlanarEncirclementController(config)
    metrics: List[EpisodeMetric] = []
    for episode in range(episodes):
        env.reset(seed=episode)
        step_metrics = []
        info = {"success": False, "collision": False, "timeout": False}
        for step in range(int(config["environment"]["max_steps"])):
            actions = controller.act(env)
            _, reward, done, info = env.step(actions)
            step_metrics.append(step_metric(env, reward))
            if done:
                break
        metrics.append(summarize_episode(step_metrics, info, step + 1, float(config["environment"]["dt"])))
    return metrics


def evaluate_checkpoint(config: Dict[str, Any], checkpoint: str, episodes: int) -> List[EpisodeMetric]:
    trainer = TorchMRARLECTrainer(config)
    trainer.load_checkpoint(checkpoint)
    actor = trainer.actor
    actor.eval()
    env = EncirclementEnv(config)
    guide = APFGuidePolicy(config)
    device = trainer.device
    policy_mode = str(config.get("training", {}).get("policy_mode", "direct")).lower()
    residual_scale = float(config.get("training", {}).get("residual_scale", 1.0))
    metrics: List[EpisodeMetric] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        hidden = actor.initial_hidden(env.num_pursuers, device)
        step_metrics = []
        info = {"success": False, "collision": False, "timeout": False}
        for step in range(int(config["environment"]["max_steps"])):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, hidden = actor.act(obs_tensor, hidden, deterministic=True)
            actor_action = action_tensor.detach().cpu().numpy()
            if policy_mode == "residual":
                action = guide.act(env) + residual_scale * actor_action
            else:
                action = actor_action
            obs, reward, done, info = env.step(action)
            step_metrics.append(step_metric(env, reward))
            if done:
                break
        metrics.append(summarize_episode(step_metrics, info, step + 1, float(config["environment"]["dt"])))
    return metrics


def write_outputs(summary: Dict[str, float], episodes: List[EpisodeMetric], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(output_dir / "episodes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
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
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(episodes):
            writer.writerow({
                "episode": idx,
                "success": item.success,
                "collision": item.collision,
                "timeout": item.timeout,
                "steps": item.steps,
                "encirclement_time": item.encirclement_time,
                "average_step_reward": item.average_step_reward,
                "danger_rate": item.danger_rate,
                "final_distance_error": item.final_distance_error,
                "final_angle_error": item.final_angle_error,
                "min_obstacle_clearance": item.min_obstacle_clearance,
                "min_pursuer_clearance": item.min_pursuer_clearance,
            })


def run(config_path: str, policy: str, checkpoint: str | None, episodes: int, output_dir: str) -> Dict[str, float]:
    config = load_config(config_path)
    if policy == "checkpoint":
        if not checkpoint:
            raise ValueError("--checkpoint is required when --policy checkpoint")
        episode_metrics = evaluate_checkpoint(config, checkpoint, episodes)
    elif policy == "guide":
        episode_metrics = evaluate_guide(config, episodes)
    elif policy == "controller":
        episode_metrics = evaluate_controller(config, episodes)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    summary = aggregate_episodes(episode_metrics)
    summary.update(communication_meta_messages(int(config["environment"]["num_pursuers"])))
    write_outputs(summary, episode_metrics, Path(output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style MRA-RLEC metrics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--policy", choices=["guide", "checkpoint", "controller"], default="guide")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="results/metrics")
    args = parser.parse_args()
    run(args.config, args.policy, args.checkpoint, args.episodes, args.output_dir)


if __name__ == "__main__":
    main()
