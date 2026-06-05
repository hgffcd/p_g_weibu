"""Stage-2 vector trainer that uses the obstacle-channel guide."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from algorithms.regulators import angle_distance_bias
from stage2.channel_guide import Stage2ChannelGuidePolicy
from stage2.curriculum import target_speed_for_progress
from stage2.envs import Stage2EncirclementEnv
from stage2.trainer import Stage2VectorMAPPOTrainer


class Stage2ChannelVectorMAPPOTrainer(Stage2VectorMAPPOTrainer):
    """Residual MAPPO trainer with the channel guide as nominal policy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.guide = Stage2ChannelGuidePolicy(config)

    def _apply_curriculum(self, episode: int, total_episodes: int) -> None:
        """Keep true obstacle geometry for channel waypoints during training.

        The base Stage-2 trainer applies Eq. (14)'s obstacle-radius curriculum,
        which sets early obstacle radii/buffers near zero. The channel guide
        computes safe left/right corridors from obstacle radius and buffer, so
        changing those values during rollout invalidates the geometry validated
        by ``stage2.channel_probe``. We still keep the paper-inspired
        angle/distance tolerance curriculum and target-speed curriculum.
        """
        delta_alpha, delta_distance = angle_distance_bias(episode, total_episodes, self.config)
        self.envs.apply_curriculum(delta_alpha, delta_distance, self.original_obstacles)
        self.envs.set_target_max_velocity(target_speed_for_progress(episode, total_episodes, self.config))

    def evaluate_actor_only(self, episodes: int, seed: int = 0) -> Dict[str, float]:
        """Evaluate residual actor around the same channel guide used in training."""
        env = Stage2EncirclementEnv(self.config)
        guide = Stage2ChannelGuidePolicy(self.config)
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
