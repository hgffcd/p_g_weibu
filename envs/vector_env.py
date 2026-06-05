"""Synchronous vectorized planar encirclement environments.

This keeps dependencies minimal for the server environment: pure Python + NumPy.
Each sub-environment is the same circular-entity EMOCA environment; batching is
used to collect PPO rollouts efficiently.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.encirclement_env import EncirclementEnv


class VectorizedEncirclementEnv:
    def __init__(self, config: Dict[str, Any], num_envs: int):
        self.config = config
        self.num_envs = int(num_envs)
        self.envs = [EncirclementEnv(copy.deepcopy(config)) for _ in range(self.num_envs)]
        self.num_pursuers = self.envs[0].num_pursuers
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].flat_state().shape[0]
        self.max_steps = self.envs[0].max_steps
        self.step_in_episode = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        observations = []
        states = []
        for idx, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + idx)
            observations.append(obs)
            states.append(env.flat_state())
        self.step_in_episode[:] = 0
        self.episode_counts[:] = 0
        return np.stack(observations), np.stack(states)

    def flat_states(self) -> np.ndarray:
        return np.stack([env.flat_state() for env in self.envs]).astype(np.float32)

    def observations(self) -> np.ndarray:
        return np.stack([env.compute_observations() for env in self.envs]).astype(np.float32)

    def step(self, actions: np.ndarray, seed_base: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        observations = []
        rewards = []
        dones = []
        infos: List[Dict[str, Any]] = []
        states = []
        for idx, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[idx])
            self.step_in_episode[idx] += 1
            if done:
                self.episode_counts[idx] += 1
                info = dict(info)
                info["terminal_step_count"] = int(self.step_in_episode[idx])
                obs, _ = env.reset(seed=seed_base + int(self.episode_counts[idx]) * self.num_envs + idx)
                self.step_in_episode[idx] = 0
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            states.append(env.flat_state())
        return (
            np.stack(observations).astype(np.float32),
            np.stack(rewards).astype(np.float32),
            np.asarray(dones, dtype=bool),
            infos,
            np.stack(states).astype(np.float32),
        )

    def guide_actions(self, guide_policy) -> np.ndarray:
        return np.stack([guide_policy.act(env) for env in self.envs]).astype(np.float32)

    def apply_curriculum(self, delta_alpha: float, delta_distance: float, obstacles) -> None:
        for env in self.envs:
            env.delta_alpha = float(delta_alpha)
            env.delta_distance = float(delta_distance)
            env.base_obstacles = [copy.copy(o) for o in obstacles]
            env.obstacles = [copy.copy(o) for o in obstacles]
