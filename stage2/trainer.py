"""Stage-2 vector trainer that reuses existing PPO code without editing it."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from agents.apf_guide import APFGuidePolicy
from algorithms.regulators import angle_distance_bias, obstacle_buffer, obstacle_radius
from algorithms.vector_mappo import VectorMAPPOTrainer
from envs.encirclement_env import Obstacle
from stage2.curriculum import target_speed_for_progress
from stage2.envs import Stage2EncirclementEnv, Stage2VectorizedEncirclementEnv


class Stage2VectorMAPPOTrainer(VectorMAPPOTrainer):
    """Drop-in vector trainer that swaps in Stage2 randomized environments."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.envs = Stage2VectorizedEncirclementEnv(config, self.num_envs)
        seed = int(config.get("training", {}).get("seed", 0))
        self.envs.reset(seed=seed)
        self.original_obstacles = list(self.envs.envs[0].base_obstacles)

    def _apply_curriculum(self, episode: int, total_episodes: int) -> None:
        delta_alpha, delta_distance = angle_distance_bias(episode, total_episodes, self.config)
        obstacles = [
            Obstacle(
                o.x,
                o.y,
                obstacle_radius(episode, total_episodes, o.radius, self.config),
                obstacle_buffer(episode, total_episodes, o.radius, o.buffer, self.config),
            )
            for o in self.original_obstacles
        ]
        self.envs.apply_curriculum(delta_alpha, delta_distance, obstacles)
        self.envs.set_target_max_velocity(target_speed_for_progress(episode, total_episodes, self.config))

    def evaluate_actor_only(self, episodes: int, seed: int = 0) -> Dict[str, float]:
        """Evaluate with the same Stage2 reset distribution used for training."""
        env = Stage2EncirclementEnv(self.config)
        guide = APFGuidePolicy(self.config)
        self.actor.eval()
        rewards: List[float] = []
        successes = 0
        collisions = 0
        timeouts = 0
        with torch.no_grad():
            for episode in range(int(episodes)):
                obs, _ = env.reset(seed=seed + episode)
                hidden = self.actor.initial_hidden(env.num_pursuers, self.device)
                total_reward = 0.0
                info = {"success": False, "collision": False, "timeout": False}
                for _ in range(int(self.config["environment"]["max_steps"])):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    action_tensor, _, hidden = self.actor.act(obs_tensor, hidden, deterministic=True)
                    actor_action = action_tensor.detach().cpu().numpy()
                    if self.policy_mode == "residual":
                        action = guide.act(env) + self.residual_scale * actor_action
                    else:
                        action = actor_action
                    obs, reward, done, info = env.step(action)
                    total_reward += float(np.mean(reward))
                    if done:
                        break
                rewards.append(total_reward)
                successes += int(info["success"])
                collisions += int(info["collision"])
                timeouts += int(info["timeout"])
        self.actor.train()
        denom = max(int(episodes), 1)
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "success_rate": successes / denom,
            "collision_rate": collisions / denom,
            "timeout_rate": timeouts / denom,
        }
