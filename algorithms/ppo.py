"""PPO update used by the rMAPPO-style MRA-RLEC trainer.

This implements the clipped objective in Section III-A, Eq. (6). The critic loss
is MSE because the paper denotes it as L^V(phi) but does not give a formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agents.torch_networks import RecurrentActor, RecurrentCritic
from algorithms.rollout_buffer import RolloutBatch


@dataclass
class PPOStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    bc_loss: float


class PPOUpdater:
    def __init__(
        self,
        actor: RecurrentActor,
        critic: RecurrentCritic,
        learning_rate: float,
        clip_coef: float,
        entropy_coef: float,
        value_coef: float,
        bc_coef: float,
        max_grad_norm: float,
    ):
        self.actor = actor
        self.critic = critic
        self.clip_coef = float(clip_coef)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.bc_coef = float(bc_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=float(learning_rate),
        )
        self.skipped_updates = 0

    def update(self, batch: RolloutBatch, epochs: int, minibatch_size: int) -> PPOStats:
        self.actor.clamp_distribution_parameters()
        batch.observations = torch.nan_to_num(batch.observations, nan=0.0, posinf=1e6, neginf=-1e6)
        batch.actions = torch.nan_to_num(batch.actions, nan=0.0, posinf=1e6, neginf=-1e6)
        batch.old_log_probs = torch.nan_to_num(batch.old_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        batch.advantages = torch.nan_to_num(batch.advantages, nan=0.0, posinf=0.0, neginf=0.0)
        batch.guide_mask = batch.guide_mask.bool()
        batch.states = torch.nan_to_num(batch.states, nan=0.0, posinf=1e6, neginf=-1e6)
        batch.returns = torch.nan_to_num(batch.returns, nan=0.0, posinf=1e6, neginf=-1e6)
        actor_losses = []
        critic_losses = []
        entropies = []
        bc_losses = []
        total_actor_items = batch.observations.shape[0]
        total_critic_items = batch.states.shape[0]

        for _ in range(int(epochs)):
            actor_perm = torch.randperm(total_actor_items, device=batch.observations.device)
            critic_perm = torch.randperm(total_critic_items, device=batch.states.device)
            num_minibatches = max(1, (total_actor_items + minibatch_size - 1) // minibatch_size)

            for mb in range(num_minibatches):
                actor_idx = actor_perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                if actor_idx.numel() == 0:
                    continue

                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch.observations[actor_idx],
                    batch.actions[actor_idx],
                )
                log_ratio = torch.clamp(new_log_probs - batch.old_log_probs[actor_idx], -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                advantages = batch.advantages[actor_idx]
                guide_mask = batch.guide_mask[actor_idx]
                learned_mask = ~guide_mask

                # Eq. (6): PPO clipped surrogate objective.
                if learned_mask.any():
                    unclipped = ratio[learned_mask] * advantages[learned_mask]
                    clipped = torch.clamp(ratio[learned_mask], 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages[learned_mask]
                    actor_loss = -torch.min(unclipped, clipped).mean()
                else:
                    actor_loss = torch.zeros((), device=batch.observations.device)

                if guide_mask.any() and self.bc_coef > 0.0:
                    guide_mean, _, _ = self.actor(batch.observations[actor_idx][guide_mask], None)
                    bc_loss = nn.functional.mse_loss(guide_mean, batch.actions[actor_idx][guide_mask])
                else:
                    bc_loss = torch.zeros((), device=batch.observations.device)

                critic_idx = critic_perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                if critic_idx.numel() == 0:
                    critic_idx = critic_perm[: min(minibatch_size, total_critic_items)]
                values, _ = self.critic(batch.states[critic_idx], None)
                critic_loss = nn.functional.mse_loss(values, batch.returns[critic_idx])

                loss = actor_loss + self.value_coef * critic_loss + self.bc_coef * bc_loss - self.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    self.skipped_updates += 1
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.skipped_updates += 1
                    continue
                self.optimizer.step()
                self.actor.clamp_distribution_parameters()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.mean().detach().cpu()))
                bc_losses.append(float(bc_loss.detach().cpu()))

        return PPOStats(
            actor_loss=sum(actor_losses) / max(len(actor_losses), 1),
            critic_loss=sum(critic_losses) / max(len(critic_losses), 1),
            entropy=sum(entropies) / max(len(entropies), 1),
            bc_loss=sum(bc_losses) / max(len(bc_losses), 1),
        )
