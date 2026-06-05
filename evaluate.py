"""Evaluation entry point."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from agents.apf_guide import APFGuidePolicy
from agents.torch_networks import RecurrentActor
from envs.encirclement_env import EncirclementEnv


def run_evaluate(config: Dict[str, Any], episodes: int = 3) -> Dict[str, float]:
    env = EncirclementEnv(config)
    guide = APFGuidePolicy(config)
    rewards = []
    successes = 0
    collisions = 0

    for episode in range(episodes):
        env.reset(seed=episode)
        total_reward = 0.0
        info = {"success": False, "collision": False}
        for _ in range(int(config["environment"]["max_steps"])):
            actions = guide.act(env)
            _, reward, done, info = env.step(actions)
            total_reward += float(np.mean(reward))
            if done:
                break
        rewards.append(total_reward)
        successes += int(info["success"])
        collisions += int(info["collision"])

    result = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
    }
    print(
        f"episodes={episodes} mean_reward={result['mean_reward']:.3f} "
        f"success_rate={result['success_rate']:.3f} collision_rate={result['collision_rate']:.3f}"
    )
    return result


def run_evaluate_actor(config: Dict[str, Any], actor: RecurrentActor, episodes: int = 3) -> Dict[str, float]:
    """Evaluate a trained actor with decentralized execution, Section IV-C.3.

    If `training.policy_mode` is `residual`, the actor output is interpreted as a
    bounded correction around the environment.py-style guide policy.
    """
    import torch

    env = EncirclementEnv(config)
    guide = APFGuidePolicy(config)
    device = next(actor.parameters()).device
    policy_mode = str(config.get("training", {}).get("policy_mode", "direct")).lower()
    residual_scale = float(config.get("training", {}).get("residual_scale", 1.0))
    rewards = []
    successes = 0
    collisions = 0
    actor.eval()

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        hidden = actor.initial_hidden(env.num_pursuers, device)
        total_reward = 0.0
        info = {"success": False, "collision": False}
        for _ in range(int(config["environment"]["max_steps"])):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, hidden = actor.act(obs_tensor, hidden, deterministic=True)
            actor_action = action_tensor.detach().cpu().numpy()
            if policy_mode == "residual":
                action = guide.act(env) + residual_scale * actor_action
            else:
                action = actor_action
            obs, reward, done, info = env.step(action)
            total_reward += float(np.mean(reward))
            if done:
                break
        rewards.append(total_reward)
        successes += int(info["success"])
        collisions += int(info["collision"])

    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
    }
