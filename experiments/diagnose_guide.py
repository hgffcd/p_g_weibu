"""Guide-policy diagnostics before expensive server training."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np

from agents.apf_guide import APFGuidePolicy
from algorithms.regulators import angle_distance_bias, obstacle_buffer, obstacle_radius
from envs.encirclement_env import EncirclementEnv, Obstacle
from utils.config import load_config


def apply_curriculum(env: EncirclementEnv, config: Dict[str, Any], step: int, total: int) -> None:
    delta_alpha, delta_distance = angle_distance_bias(step, total, config)
    env.delta_alpha = delta_alpha
    env.delta_distance = delta_distance
    env.base_obstacles = [
        Obstacle(
            o.x,
            o.y,
            obstacle_radius(step, total, o.radius, config),
            obstacle_buffer(step, total, o.radius, o.buffer, config),
        )
        for o in env.base_obstacles
    ]


def run(config_path: str, episodes: int, schedule_step: int, schedule_total: int, progress: bool = False) -> Dict[str, float]:
    config = load_config(config_path)
    guide = APFGuidePolicy(config)
    successes = 0
    collisions = 0
    timeouts = 0
    pursuer_collisions = 0
    obstacle_collisions = 0
    steps = []
    rewards = []
    distance_errors = []
    angle_errors = []

    for episode in range(int(episodes)):
        env = EncirclementEnv(config)
        apply_curriculum(env, config, int(schedule_step), int(schedule_total))
        obs, _ = env.reset(seed=episode)
        total_reward = 0.0
        info = {"success": False, "collision": False, "timeout": False}
        for step_idx in range(int(config["environment"]["max_steps"])):
            actions = guide.act(env)
            obs, reward, done, info = env.step(actions)
            total_reward += float(np.mean(reward))
            if done:
                break
        successes += int(info["success"])
        collisions += int(info["collision"])
        timeouts += int(info["timeout"])
        pursuer_collisions += int(info.get("collision_type") == "pursuer")
        obstacle_collisions += int(info.get("collision_type") == "obstacle")
        if "distance_error" in info and "angle_error" in info:
            distance_errors.append(float(info["distance_error"]))
            angle_errors.append(float(info["angle_error"]))
        steps.append(step_idx + 1)
        rewards.append(total_reward)
        if progress:
            print(
                f"episode={episode + 1}/{episodes} success={info['success']} "
                f"collision={info['collision']} timeout={info['timeout']} steps={step_idx + 1} "
                f"distance_error={info.get('distance_error', 0.0):.3f} "
                f"angle_error={info.get('angle_error', 0.0):.3f}",
                flush=True,
            )

    denom = max(int(episodes), 1)
    summary = {
        "episodes": float(episodes),
        "schedule_step": float(schedule_step),
        "schedule_total": float(schedule_total),
        "success_rate": successes / denom,
        "collision_rate": collisions / denom,
        "pursuer_collision_rate": pursuer_collisions / denom,
        "obstacle_collision_rate": obstacle_collisions / denom,
        "timeout_rate": timeouts / denom,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "final_distance_error": float(np.mean(distance_errors)) if distance_errors else 0.0,
        "final_angle_error": float(np.mean(angle_errors)) if angle_errors else 0.0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose APF guide policy under a curriculum stage")
    parser.add_argument("--config", default="config_vector_server.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--schedule-step", type=int, default=1)
    parser.add_argument("--schedule-total", type=int, default=2000)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    run(args.config, args.episodes, args.schedule_step, args.schedule_total, progress=args.progress)


if __name__ == "__main__":
    main()
