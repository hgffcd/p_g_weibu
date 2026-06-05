"""Planar-circle reproduction suite for the paper's Section V metrics.

This intentionally excludes ROS2/Gazebo/model-car simulation. Pursuers, target,
and obstacles are all circular entities in a 2-D plane, matching the simplified
scope requested for code reproduction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiments.run_metrics import evaluate_checkpoint, evaluate_controller, evaluate_guide
from utils.config import load_config
from utils.metrics import aggregate_episodes, communication_meta_messages


def regular_line_obstacles(count: int) -> list[list[float]]:
    """Create deterministic circular obstacles for planar smoke experiments."""
    if count <= 0:
        return []
    rows = []
    for idx in range(count):
        x = -1.4 + 0.7 * (idx % 5)
        y = 0.9 + 0.55 * (idx // 5)
        radius = 0.12 + 0.02 * (idx % 3)
        rows.append([x, y, radius, 0.05])
    return rows


def set_num_pursuers(config: Dict[str, Any], n: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["environment"]["num_pursuers"] = int(n)
    # Keep initial spacing large enough to avoid immediate buffer violations.
    cfg["environment"]["initial_spacing"] = max(0.55, 2.8 / max(n - 1, 1))
    return cfg


def set_target_complexity(config: Dict[str, Any], target_speed: float, target_policy: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["environment"]["target_max_velocity"] = float(target_speed)
    cfg["environment"]["target_policy"] = str(target_policy)
    return cfg


def set_obstacle_count(config: Dict[str, Any], count: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["environment"]["obstacles"] = regular_line_obstacles(count)
    return cfg


def evaluate_case(config: Dict[str, Any], policy: str, checkpoint: str | None, episodes: int) -> Dict[str, float]:
    if policy == "checkpoint":
        if not checkpoint:
            raise ValueError("checkpoint policy requires --checkpoint")
        episode_metrics = evaluate_checkpoint(config, checkpoint, episodes)
    elif policy == "guide":
        episode_metrics = evaluate_guide(config, episodes)
    elif policy == "controller":
        episode_metrics = evaluate_controller(config, episodes)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    summary = aggregate_episodes(episode_metrics)
    summary.update(communication_meta_messages(int(config["environment"]["num_pursuers"])))
    return summary


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(
    config_path: str,
    policy: str,
    checkpoint: str | None,
    episodes: int,
    output_dir: str,
    pursuer_counts: Iterable[int],
    target_speeds: Iterable[float],
    target_policies: Iterable[str],
    obstacle_counts: Iterable[int],
) -> dict[str, list[dict[str, Any]]]:
    base_config = load_config(config_path)
    out = Path(output_dir)
    tables: dict[str, list[dict[str, Any]]] = {
        "scalability": [],
        "complexity": [],
        "escape_policy": [],
        "obstacle_count": [],
    }

    for n in pursuer_counts:
        cfg = set_num_pursuers(base_config, int(n))
        summary = evaluate_case(cfg, policy, checkpoint, episodes)
        tables["scalability"].append({"num_pursuers": int(n), **summary})

    for speed in target_speeds:
        cfg = set_target_complexity(base_config, float(speed), str(base_config["environment"].get("target_policy", "static")))
        summary = evaluate_case(cfg, policy, checkpoint, episodes)
        tables["complexity"].append({"target_max_velocity": float(speed), **summary})

    for target_policy in target_policies:
        cfg = set_target_complexity(base_config, float(base_config["environment"]["target_max_velocity"]), target_policy)
        summary = evaluate_case(cfg, policy, checkpoint, episodes)
        tables["escape_policy"].append({"target_policy": target_policy, **summary})

    for count in obstacle_counts:
        cfg = set_obstacle_count(base_config, int(count))
        summary = evaluate_case(cfg, policy, checkpoint, episodes)
        tables["obstacle_count"].append({"obstacle_count": int(count), **summary})

    for name, rows in tables.items():
        write_table(out / f"{name}.csv", rows)
    with open(out / "summary_tables.json", "w", encoding="utf-8") as handle:
        json.dump(tables, handle, indent=2, ensure_ascii=False)
    return tables


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run planar-circle paper metrics suite")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--policy", choices=["guide", "checkpoint", "controller"], default="guide")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="results/planar_suite")
    parser.add_argument("--pursuer-counts", default="3,4,5,6")
    parser.add_argument("--target-speeds", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--target-policies", default="static,greedy,pfp")
    parser.add_argument("--obstacle-counts", default="0,4,8,12")
    args = parser.parse_args()

    tables = run_suite(
        config_path=args.config,
        policy=args.policy,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        output_dir=args.output_dir,
        pursuer_counts=parse_ints(args.pursuer_counts),
        target_speeds=parse_floats(args.target_speeds),
        target_policies=parse_strings(args.target_policies),
        obstacle_counts=parse_ints(args.obstacle_counts),
    )
    print(json.dumps(tables, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
