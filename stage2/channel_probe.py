"""Evaluate the Stage-2 obstacle-channel guide."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from stage2.channel_guide import Stage2ChannelGuidePolicy
from stage2.envs import Stage2EncirclementEnv
from stage2.evaluate import configure_scenario
from stage2.visualize import save_animation
from utils.config import load_config
from utils.metrics import aggregate_episodes, step_metric, summarize_episode


def run_episode(config: Dict[str, Any], seed: int):
    env = Stage2EncirclementEnv(config)
    guide = Stage2ChannelGuidePolicy(config)
    obs, _ = env.reset(seed=seed)
    metrics = []
    info: Dict[str, Any] = {"success": False, "collision": False, "timeout": False}
    pursuer_frames = [env.pursuer_pos.copy()]
    target_frames = [env.target_pos.copy()]
    status_frames = [{"step": 0, "success": False, "collision": False, "timeout": False}]
    for step in range(int(config["environment"]["max_steps"])):
        obs, reward, done, info = env.step(guide.act(env))
        metrics.append(step_metric(env, reward))
        pursuer_frames.append(env.pursuer_pos.copy())
        target_frames.append(env.target_pos.copy())
        status_frames.append({
            "step": step + 1,
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "timeout": bool(info.get("timeout", False)),
        })
        if done:
            break
    metric = summarize_episode(metrics, info, step + 1, float(config["environment"]["dt"]))
    trajectory = {
        "pursuer_pos": np.asarray(pursuer_frames, dtype=float),
        "target_pos": np.asarray(target_frames, dtype=float),
        "status": status_frames,
        "obstacles": [(o.x, o.y, o.radius, o.buffer) for o in env.obstacles],
        "capture_distance": env.capture_distance,
        "pursuer_radius": env.pursuer_radius,
        "target_radius": env.target_radius,
        "policy": "channel_guide",
        "seed": seed,
        "final_info": info,
        "mean_reward": metric.average_step_reward,
        "reset_info": env.last_reset_info,
    }
    reset = env.last_reset_info
    return metric, trajectory, {
        "target_x": float(reset.get("target_pos", env.target_pos)[0]),
        "target_y": float(reset.get("target_pos", env.target_pos)[1]),
        "reset_attempt": int(reset.get("attempt", 1)),
        "randomized": bool(reset.get("randomized", False)),
    }


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path: str, speed: float, episodes: int, seed: int, output_dir: Path, gif: bool) -> Dict[str, Any]:
    config = configure_scenario(load_config(config_path), "randomized")
    config["environment"]["target_max_velocity"] = float(speed)
    metrics = []
    rows = []
    first_failure = None
    first_failure_traj = None
    for episode in range(int(episodes)):
        episode_seed = int(seed) + episode
        metric, trajectory, reset = run_episode(config, episode_seed)
        metrics.append(metric)
        rows.append({
            "episode": episode,
            "seed": episode_seed,
            **reset,
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
        if first_failure is None and not metric.success:
            first_failure = episode_seed
            first_failure_traj = trajectory

    summary = aggregate_episodes(metrics)
    summary["target_speed"] = float(speed)  # type: ignore[assignment]
    summary["first_failure_seed"] = float(first_failure if first_failure is not None else -1)  # type: ignore[assignment]
    run_dir = output_dir / Path(config_path).stem.replace("config_stage2_", "") / f"speed_{speed:.2f}".replace(".", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_rows(run_dir / "episodes.csv", rows)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if gif and first_failure_traj is not None:
        save_animation(first_failure_traj, run_dir / f"first_failure_seed{first_failure}.gif")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 channel guide probe")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-sweep", default="0.0,0.15")
    parser.add_argument("--output-dir", default="results_stage2/channel_probe")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    speeds = [float(item) for item in args.speed_sweep.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    results = []
    for config_path in args.configs:
        for speed in speeds:
            summary = evaluate(config_path, speed, args.episodes, args.seed, output_dir, args.gif)
            summary["config"] = config_path  # type: ignore[assignment]
            results.append(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary_all.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

