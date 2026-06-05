"""PyTorch actor and critic networks for the paper's rMAPPO setup.

The architecture follows Section IV-C.3 and Table II:
- actor: MLP hidden units 128, 64 plus a GRU hidden state of 32;
- critic: MLP hidden units 256, 128, 64 plus a GRU hidden state of 32;
- tanh activations are used as stated in Section IV-C.3.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(input_dim: int, hidden_units: Iterable[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_units:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers), last_dim


class RecurrentActor(nn.Module):
    """Shared actor network mapping Eq. (19) observations to Eq. (1) actions."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_units: Iterable[int] = (128, 64),
        rnn_hidden_units: int = 32,
    ):
        super().__init__()
        self.action_limit = float(action_limit)
        self.feature, feature_dim = build_mlp(obs_dim, hidden_units)
        self.gru = nn.GRU(feature_dim, int(rnn_hidden_units), batch_first=True)
        self.mean = nn.Linear(int(rnn_hidden_units), action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one recurrent step.

        Args:
            obs: shape [batch, obs_dim].
            hidden: shape [1, batch, actor_rnn_hidden_units].

        Returns:
            action mean, action std, next hidden.
        """
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if hidden is None:
            hidden = self.initial_hidden(obs.shape[0], obs.device)
        else:
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e6, neginf=-1e6)
        features = self.feature(obs).unsqueeze(1)
        rnn_out, next_hidden = self.gru(features, hidden)
        mean = torch.tanh(self.mean(rnn_out[:, -1])) * self.action_limit
        mean = torch.nan_to_num(mean, nan=0.0, posinf=self.action_limit, neginf=-self.action_limit)
        log_std = torch.nan_to_num(self.log_std, nan=-0.5, posinf=self.LOG_STD_MAX, neginf=self.LOG_STD_MIN)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).clamp_min(1e-6).expand_as(mean)
        return mean, std, next_hidden

    def distribution(self, obs: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[Normal, torch.Tensor]:
        mean, std, next_hidden = self.forward(obs, hidden)
        return Normal(mean, std), next_hidden

    @torch.no_grad()
    def clamp_distribution_parameters(self) -> None:
        """Keep learnable exploration std in a numerically valid range."""
        self.log_std.data = torch.nan_to_num(self.log_std.data, nan=-0.5, posinf=self.LOG_STD_MAX, neginf=self.LOG_STD_MIN)
        self.log_std.data.clamp_(self.LOG_STD_MIN, self.LOG_STD_MAX)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: torch.Tensor | None = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, next_hidden = self.distribution(obs, hidden)
        raw_action = dist.mean if deterministic else dist.rsample()
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=self.action_limit, neginf=-self.action_limit)
        action = torch.clamp(raw_action, -self.action_limit, self.action_limit)
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        return action, log_prob, next_hidden

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=self.action_limit, neginf=-self.action_limit)
        actions = torch.clamp(actions, -self.action_limit, self.action_limit)
        dist, _ = self.distribution(obs, None)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob, entropy


class RecurrentCritic(nn.Module):
    """Centralized critic evaluating the overall state, as in Section IV-C.3."""

    def __init__(
        self,
        state_dim: int,
        hidden_units: Iterable[int] = (256, 128, 64),
        rnn_hidden_units: int = 32,
    ):
        super().__init__()
        self.feature, feature_dim = build_mlp(state_dim, hidden_units)
        self.gru = nn.GRU(feature_dim, int(rnn_hidden_units), batch_first=True)
        self.value_head = nn.Linear(int(rnn_hidden_units), 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)

    def forward(self, state: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        if hidden is None:
            hidden = self.initial_hidden(state.shape[0], state.device)
        else:
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e6, neginf=-1e6)
        features = self.feature(state).unsqueeze(1)
        rnn_out, next_hidden = self.gru(features, hidden)
        value = self.value_head(rnn_out[:, -1]).squeeze(-1)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        return value, next_hidden
