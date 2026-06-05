"""Stage-2 pretraining on moving/randomized reset distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from agents.apf_guide import APFGuidePolicy
from algorithms.mra_rlec import TorchMRARLECTrainer
from algorithms.regulators import angle_distance_bias, obstacle_buffer, obstacle_radius
from envs.encirclement_env import Obstacle
from stage2.curriculum import target_speed_for_progress
from stage2.envs import Stage2EncirclementEnv
from stage2.evaluate import configure_scenario
from utils.config import load_config


def run_pretrain(config: Dict[str, Any], scenario: str = "fixed") -> str:
    config = configure_scenario(config, scenario)
    train_cfg = config["training"]
    pretrain_cfg = config.get("pretrain", {})
    episodes = int(pretrain_cfg.get("episodes", 1000))
    epochs_per_episode = int(pretrain_cfg.get("epochs_per_episode", 4))
    lr = float(pretrain_cfg.get("learning_rate", 0.0005))
    checkpoint_path = str(pretrain_cfg.get("checkpoint_path", "checkpoints/pretrain_actor.pt"))
    policy_mode = str(train_cfg.get("policy_mode", "direct")).lower()

    trainer = TorchMRARLECTrainer(config)
    env = Stage2EncirclementEnv(config)
    original_obstacles = list(env.base_obstacles)
    guide = APFGuidePolicy(config)
    optimizer = torch.optim.Adam(trainer.actor.parameters(), lr=lr)
    mse = nn.MSELoss()
    max_steps = int(config["environment"]["max_steps"])
    seed = int(train_cfg.get("seed", 0))

    for episode in range(1, episodes + 1):
        delta_alpha, delta_distance = angle_distance_bias(episode, episodes, config)
        env.delta_alpha = delta_alpha
        env.delta_distance = delta_distance
        env.target_max_velocity = target_speed_for_progress(episode, episodes, config)
        env.base_obstacles = [
            Obstacle(
                o.x,
                o.y,
                obstacle_radius(episode, episodes, o.radius, config),
                obstacle_buffer(episode, episodes, o.radius, o.buffer, config),
            )
            for o in original_obstacles
        ]

        observations, _ = env.reset(seed=seed + episode)
        obs_list = []
        action_list = []
        for _ in range(max_steps):
            actions = guide.act(env).astype(np.float32)
            obs_list.append(observations.astype(np.float32))
            if policy_mode == "residual":
                action_list.append(np.zeros_like(actions, dtype=np.float32))
            else:
                action_list.append(actions)
            observations, _, done, _ = env.step(actions)
            if done:
                break

        obs_tensor = torch.as_tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32, device=trainer.device)
        action_tensor = torch.as_tensor(np.concatenate(action_list, axis=0), dtype=torch.float32, device=trainer.device)
        last_loss = 0.0
        for _ in range(epochs_per_episode):
            mean, _, _ = trainer.actor(obs_tensor, None)
            loss = mse(mean, action_tensor)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.actor.parameters(), float(train_cfg.get("max_grad_norm", 0.25)))
            optimizer.step()
            trainer.actor.clamp_distribution_parameters()
            last_loss = float(loss.detach().cpu())

        if episode == 1 or episode % int(pretrain_cfg.get("log_interval", 50)) == 0:
            print(
                f"stage2_pretrain_episode={episode} scenario={scenario} mode={policy_mode} "
                f"target_vmax={env.target_max_velocity:.3f} samples={obs_tensor.shape[0]} bc_mse={last_loss:.6f}"
            )

    if policy_mode == "residual":
        with torch.no_grad():
            trainer.actor.log_std.fill_(float(pretrain_cfg.get("residual_log_std", -2.0)))
            trainer.actor.clamp_distribution_parameters()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path=checkpoint_path, episode=0)
    print(f"stage2_pretrain_checkpoint={checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 pretraining")
    parser.add_argument("--config", default="config_stage2_moving.yaml")
    parser.add_argument("--scenario", choices=["fixed", "randomized"], default="fixed")
    args = parser.parse_args()
    run_pretrain(load_config(args.config), scenario=args.scenario)


if __name__ == "__main__":
    main()
