"""Rollout buffer for vectorized MAPPO-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from algorithms.rollout_buffer import RolloutBatch


class VectorRolloutBuffer:
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.states: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.guide_masks: List[np.ndarray] = []

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        states: np.ndarray,
        values: np.ndarray,
        guide_mask: np.ndarray,
    ) -> None:
        self.observations.append(np.asarray(observations, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.float32))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=bool))
        self.states.append(np.asarray(states, dtype=np.float32))
        self.values.append(np.asarray(values, dtype=np.float32))
        self.guide_masks.append(np.asarray(guide_mask, dtype=bool))

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        obs = np.stack(self.observations)          # [T, E, N, D]
        actions = np.stack(self.actions)          # [T, E, N, A]
        log_probs = np.stack(self.log_probs)      # [T, E, N]
        rewards = np.stack(self.rewards)          # [T, E]
        dones = np.stack(self.dones).astype(np.float32)
        states = np.stack(self.states)            # [T, E, S]
        values = np.stack(self.values)            # [T, E]
        guide_masks = np.stack(self.guide_masks)  # [T, E, N]

        rewards = np.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        last_values = np.nan_to_num(last_values.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)

        t_len, n_envs = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(n_envs, dtype=np.float32)
        next_values = last_values
        for t in reversed(range(t_len)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_values = values[t]
        returns = advantages + values
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        returns = np.nan_to_num(returns, nan=0.0, posinf=1e6, neginf=-1e6)

        obs_t = torch.as_tensor(obs, device=device)
        actions_t = torch.as_tensor(actions, device=device)
        old_log_probs_t = torch.as_tensor(log_probs, device=device)
        states_t = torch.as_tensor(states, device=device)
        advantages_t = torch.as_tensor(advantages, device=device)
        returns_t = torch.as_tensor(returns, device=device)
        guide_masks_t = torch.as_tensor(guide_masks, device=device)

        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        return RolloutBatch(
            observations=obs_t.reshape(-1, obs_t.shape[-1]),
            actions=actions_t.reshape(-1, actions_t.shape[-1]),
            old_log_probs=old_log_probs_t.reshape(-1),
            advantages=advantages_t.reshape(-1).repeat_interleave(obs_t.shape[2]),
            guide_mask=guide_masks_t.reshape(-1),
            states=states_t.reshape(-1, states_t.shape[-1]),
            returns=returns_t.reshape(-1),
        )
