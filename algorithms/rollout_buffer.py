"""Rollout storage for the MRA-RLEC rMAPPO implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    guide_mask: torch.Tensor
    states: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Stores one on-policy rollout collected by Algorithm 1 lines 5-8."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.states: List[np.ndarray] = []
        self.values: List[float] = []
        self.guide_masks: List[np.ndarray] = []

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        flat_state: np.ndarray,
        value: float,
        is_guide: bool,
    ) -> None:
        self.observations.append(np.asarray(observations, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.float32))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.states.append(np.asarray(flat_state, dtype=np.float32))
        self.values.append(float(value))
        self.guide_masks.append(np.full(log_probs.shape, bool(is_guide), dtype=bool))

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        """Generalized advantage estimation referenced by Algorithm 1 line 9."""
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        returns = np.nan_to_num(returns, nan=0.0, posinf=1e6, neginf=-1e6)

        observations = torch.as_tensor(np.stack(self.observations), device=device)
        actions = torch.as_tensor(np.stack(self.actions), device=device)
        old_log_probs = torch.as_tensor(np.stack(self.log_probs), device=device)
        states = torch.as_tensor(np.stack(self.states), device=device)
        guide_masks = torch.as_tensor(np.stack(self.guide_masks), device=device)
        advantages_t = torch.as_tensor(advantages, device=device)
        returns_t = torch.as_tensor(returns, device=device)
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        num_agents = observations.shape[1]
        return RolloutBatch(
            observations=observations.reshape(-1, observations.shape[-1]),
            actions=actions.reshape(-1, actions.shape[-1]),
            old_log_probs=old_log_probs.reshape(-1),
            advantages=advantages_t.repeat_interleave(num_agents),
            guide_mask=guide_masks.reshape(-1),
            states=states,
            returns=returns_t,
        )
