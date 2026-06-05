"""Quick Stage-2 guide diagnostics before expensive training."""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict

import numpy as np

from agents.apf_guide import APFGuidePolicy
from stage2.envs import Stage2EncirclementEnv
from stage2.evaluate import configure_scenario
from utils.config import load_config


def evaluate_guide(config: Dict[str, Any], episodes: int, seed: int, target_speed: float | None) -> Dict[str, float]:
    successes = 0
    collisions = 0
    timeouts = 0
    rewards = []
    steps = []
    distance_errors = []
    angle_errors = []
    attempts = []
    for episode in range(int(episodes)):
        env = Stage2EncirclementEnv(config)
        if target_speed is not None:
            env.target_max_velocity = float(target_speed)
        guide = APFGuidePolicy(config)
        obs, _ = env.reset(seed=seed + episode)
        info = {"success": False, "collision": False, "timeout": False}
        total_reward = 0.0
        for step in range(int(config["environment"]["max_steps"])):
            obs, reward, done, info = env.step(guide.act(env))
            total_reward += float(np.mean(reward))
            if done:
                break
        distance_error, angle_error = env.encirclement_errors()
        successes += int(info["success"])
        collisions += int(info["collision"])
        timeouts += int(info["timeout"])
        rewards.append(total_reward)
        steps.append(step + 1)
        distance_errors.append(distance_error)
        angle_errors.append(angle_error)
        attempts.append(int(env.last_reset_info.get("attempt", 1)))
    denom = max(int(episodes), 1)
    return {
        "episodes": float(episodes),
        "target_speed": float(target_speed if target_speed is not None else config["environment"]["target_max_velocity"]),
        "success_rate": successes / denom,
        "collision_rate": collisions / denom,
        "timeout_rate": timeouts / denom,
        "avg_steps": float(np.mean(steps)),
        "avg_episode_reward": float(np.mean(rewards)),
        "final_distance_error": float(np.mean(distance_errors)),
        "final_angle_error": float(np.mean(angle_errors)),
        "avg_reset_attempts": float(np.mean(attempts)),
    }


def run(config_path: str, scenario: str, episodes: int, seed: int, speed_sweep: str) -> None:
    base = configure_scenario(load_config(config_path), scenario)
    if speed_sweep:
        speeds = [float(item) for item in speed_sweep.split(",")]
    else:
        speeds = [float(base["environment"]["target_max_velocity"])]
    results = []
    for speed in speeds:
        cfg = copy.deepcopy(base)
        cfg["environment"]["target_max_velocity"] = speed
        results.append(evaluate_guide(cfg, episodes, seed, speed))
    print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage-2 guide feasibility")
    parser.add_argument("--config", default="config_stage2_moving.yaml")
    parser.add_argument("--scenario", choices=["fixed", "randomized"], default="fixed")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-sweep", default="")
    args = parser.parse_args()
    run(args.config, args.scenario, args.episodes, args.seed, args.speed_sweep)


if __name__ == "__main__":
    main()
