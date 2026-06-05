"""Per-seed failure probe for Stage-2 randomized guide runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from stage2.evaluate import configure_scenario, run_episode
from stage2.visualize import collect_trajectory, save_animation
from utils.config import load_config


def _config_name(path: str) -> str:
    return Path(path).stem.replace("config_stage2_", "")


def _row_from_metric(episode: int, seed: int, reset: Dict[str, Any], metric: Any) -> Dict[str, Any]:
    return {
        "episode": episode,
        "seed": seed,
        "randomized": reset["randomized"],
        "target_x": reset["target_x"],
        "target_y": reset["target_y"],
        "reset_attempt": reset["reset_attempt"],
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
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    episodes = max(len(rows), 1)
    success_rows = [r for r in rows if bool(r["success"])]
    enc_times = [float(r["encirclement_time"]) for r in success_rows if r["encirclement_time"] is not None]
    return {
        "episodes": float(len(rows)),
        "success_rate": sum(bool(r["success"]) for r in rows) / episodes,
        "collision_rate": sum(bool(r["collision"]) for r in rows) / episodes,
        "timeout_rate": sum(bool(r["timeout"]) for r in rows) / episodes,
        "encirclement_time": sum(enc_times) / len(enc_times) if enc_times else float("nan"),
        "avg_steps": sum(float(r["steps"]) for r in rows) / episodes,
        "avg_episode_reward": sum(float(r["average_step_reward"]) for r in rows) / episodes,
        "final_distance_error": sum(float(r["final_distance_error"]) for r in rows) / episodes,
        "final_angle_error": sum(float(r["final_angle_error"]) for r in rows) / episodes,
        "min_obstacle_clearance": min(float(r["min_obstacle_clearance"]) for r in rows),
        "min_pursuer_clearance": min(float(r["min_pursuer_clearance"]) for r in rows),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
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


def evaluate_config(
    config_path: str,
    speed: float,
    episodes: int,
    seed: int,
    output_dir: Path,
    make_gif: bool,
) -> Dict[str, float]:
    config = configure_scenario(load_config(config_path), "randomized")
    config["environment"]["target_max_velocity"] = float(speed)
    rows: List[Dict[str, Any]] = []
    first_failure_seed: int | None = None

    for episode in range(int(episodes)):
        episode_seed = int(seed) + episode
        metric, reset = run_episode(config, "guide", None, episode_seed)
        row = _row_from_metric(episode, episode_seed, reset, metric)
        rows.append(row)
        if first_failure_seed is None and not bool(row["success"]):
            first_failure_seed = episode_seed

    summary = _summary(rows)
    summary["target_speed"] = float(speed)
    summary["first_failure_seed"] = float(first_failure_seed if first_failure_seed is not None else -1)

    name = _config_name(config_path)
    speed_tag = f"speed_{speed:.2f}".replace(".", "p")
    run_dir = output_dir / name / speed_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "episodes.csv", rows)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if make_gif and first_failure_seed is not None:
        trajectory = collect_trajectory(config, "guide", None, first_failure_seed)
        save_animation(trajectory, run_dir / f"first_failure_seed{first_failure_seed}.gif")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Stage-2 randomized guide failures per seed")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-sweep", default="0.0,0.15")
    parser.add_argument("--output-dir", default="results_stage2/failure_probe")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    speeds = [float(item) for item in args.speed_sweep.split(",") if item.strip()]
    all_results = []
    for config_path in args.configs:
        for speed in speeds:
            summary = evaluate_config(
                config_path=config_path,
                speed=speed,
                episodes=args.episodes,
                seed=args.seed,
                output_dir=output_dir,
                make_gif=args.gif,
            )
            summary["config"] = config_path  # type: ignore[assignment]
            all_results.append(summary)

    with open(output_dir / "summary_all.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
