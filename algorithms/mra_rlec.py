"""Minimal Algorithm 1 training loop for MRA-RLEC."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from agents.apf_guide import APFGuidePolicy
from agents.simple_actor import LinearGaussianActor
from agents.torch_networks import RecurrentActor, RecurrentCritic
from algorithms.ppo import PPOUpdater
from algorithms.regulators import angle_distance_bias, guide_steps, obstacle_buffer, obstacle_radius
from algorithms.rollout_buffer import RolloutBuffer
from envs.encirclement_env import EncirclementEnv, Obstacle


class MRARLECTrainer:
    """Runs the structure of Algorithm 1 without full PPO gradient updates."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = EncirclementEnv(config)
        self.original_obstacles = list(self.env.base_obstacles)
        self.guide = APFGuidePolicy(config)
        self.actor = LinearGaussianActor(
            obs_dim=self.env.obs_dim,
            action_dim=2,
            action_limit=float(config["environment"]["max_acceleration"]),
            seed=int(config["training"].get("actor_seed", 0)),
        )

    def train(self) -> List[Dict[str, Any]]:
        total_episodes = int(self.config["training"]["episodes"])
        max_steps = int(self.config["environment"]["max_steps"])
        seed = int(self.config["training"].get("seed", 0))
        history: List[Dict[str, Any]] = []

        for episode in range(1, total_episodes + 1):
            self._apply_regulators(episode, total_episodes)
            m_g = guide_steps(episode, total_episodes, max_steps, self.config)
            observations, _ = self.env.reset(seed=seed + episode)
            total_reward = 0.0
            info = {"success": False, "collision": False, "timeout": False}

            for step in range(max_steps):
                # Algorithm 1 line 6: mixed pi_g and pi_theta by guide-step horizon.
                if step < m_g or self.config["training"].get("use_guide_after_jsrl", True):
                    actions = self.guide.act(self.env)
                else:
                    actions = self.actor.act(observations)
                observations, rewards, done, info = self.env.step(actions)
                total_reward += float(np.mean(rewards))
                if done:
                    break

            history.append({
                "episode": episode,
                "reward": total_reward,
                "steps": step + 1,
                "guide_steps": m_g,
                "success": bool(info["success"]),
                "collision": bool(info["collision"]),
                "timeout": bool(info["timeout"]),
            })
        return history

    def _apply_regulators(self, episode: int, total_episodes: int) -> None:
        # Algorithm 1 line 3: update Delta_alpha, Delta_d, and obstacle radii.
        delta_alpha, delta_distance = angle_distance_bias(episode, total_episodes, self.config)
        self.env.delta_alpha = delta_alpha
        self.env.delta_distance = delta_distance
        self.env.base_obstacles = [
            Obstacle(
                o.x,
                o.y,
                obstacle_radius(episode, total_episodes, o.radius, self.config),
                obstacle_buffer(episode, total_episodes, o.radius, o.buffer, self.config),
            )
            for o in self.original_obstacles
        ]


class TorchMRARLECTrainer:
    """PyTorch implementation of Algorithm 1 with rMAPPO-style PPO updates.

    Guide policy and curriculum regulators are from Section IV-A/B. The actor and
    centralized critic follow Section IV-C.3 and Table II. PPO policy loss follows
    Section III-A, Eq. (6).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = EncirclementEnv(config)
        self.original_obstacles = list(self.env.base_obstacles)
        self.guide = APFGuidePolicy(config)

        device_name = str(config["training"].get("device", "auto"))
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        obs, _ = self.env.reset(seed=int(config["training"].get("seed", 0)))
        state_dim = self.env.flat_state().shape[0]
        network_cfg = config["network"]
        env_cfg = config["environment"]
        self.actor = RecurrentActor(
            obs_dim=obs.shape[-1],
            action_dim=2,
            action_limit=float(env_cfg["max_acceleration"]),
            hidden_units=network_cfg["actor_hidden_units"],
            rnn_hidden_units=int(network_cfg["actor_rnn_hidden_units"]),
        ).to(self.device)
        self.critic = RecurrentCritic(
            state_dim=state_dim,
            hidden_units=network_cfg["critic_hidden_units"],
            rnn_hidden_units=int(network_cfg["critic_rnn_hidden_units"]),
        ).to(self.device)

        train_cfg = config["training"]
        self.updater = PPOUpdater(
            actor=self.actor,
            critic=self.critic,
            learning_rate=float(train_cfg["learning_rate_base"]),
            clip_coef=float(train_cfg["ppo_clip"]),
            entropy_coef=float(train_cfg["entropy_coef"]),
            value_coef=float(train_cfg["value_coef"]),
            bc_coef=float(train_cfg.get("bc_coef", 1.0)),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
        )
        self.last_checkpoint_path: str | None = None
        self.start_episode = 1
        self.log_dir = Path(str(train_cfg.get("log_dir", "logs")))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.log_dir / "train_history.csv"

    def train(self) -> List[Dict[str, Any]]:
        total_episodes = int(self.config["training"]["episodes"])
        max_steps = int(self.config["environment"]["max_steps"])
        seed = int(self.config["training"].get("seed", 0))
        gamma = float(self.config["training"]["gamma"])
        gae_lambda = float(self.config["training"]["gae_lambda"])
        ppo_epochs = int(self.config["training"]["ppo_epochs"])
        batch_size = int(self.config["training"]["batch_size"])
        history: List[Dict[str, Any]] = []

        checkpoint_interval = int(self.config["training"].get("checkpoint_interval", 0))
        log_interval = int(self.config["training"].get("log_interval", 10))

        for episode in range(self.start_episode, total_episodes + 1):
            self._apply_regulators(episode, total_episodes)
            m_g = guide_steps(episode, total_episodes, max_steps, self.config)
            observations, _ = self.env.reset(seed=seed + episode)
            actor_hidden = self.actor.initial_hidden(self.env.num_pursuers, self.device)
            buffer = RolloutBuffer()
            total_reward = 0.0
            info = {"success": False, "collision": False, "timeout": False}

            for step in range(max_steps):
                flat_state = self.env.flat_state()
                state_tensor = torch.as_tensor(flat_state[None, :], dtype=torch.float32, device=self.device)
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    value, _ = self.critic(state_tensor, None)

                is_guide = step < m_g
                if is_guide:
                    # JSRL guide segment from Algorithm 1 line 6.
                    action_np = self.guide.act(self.env).astype(np.float32)
                    action_tensor = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        log_probs, _ = self.actor.evaluate_actions(obs_tensor, action_tensor)
                else:
                    with torch.no_grad():
                        action_tensor, log_probs, actor_hidden = self.actor.act(obs_tensor, actor_hidden)
                    action_np = action_tensor.detach().cpu().numpy().astype(np.float32)

                next_observations, rewards, done, info = self.env.step(action_np)
                mean_reward = float(np.mean(rewards))
                buffer.add(
                    observations=observations,
                    actions=action_np,
                    log_probs=log_probs.detach().cpu().numpy(),
                    reward=mean_reward,
                    done=done,
                    flat_state=flat_state,
                    value=float(value.detach().cpu().item()),
                    is_guide=is_guide,
                )
                total_reward += mean_reward
                observations = next_observations
                if done:
                    break

            with torch.no_grad():
                if info["success"] or info["collision"] or info["timeout"]:
                    last_value = 0.0
                else:
                    last_state = torch.as_tensor(self.env.flat_state()[None, :], dtype=torch.float32, device=self.device)
                    last_value = float(self.critic(last_state, None)[0].detach().cpu().item())
            batch = buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda, self.device)
            stats = self.updater.update(batch, epochs=ppo_epochs, minibatch_size=batch_size)

            history.append({
                "episode": episode,
                "reward": total_reward,
                "steps": step + 1,
                "guide_steps": m_g,
                "success": bool(info["success"]),
                "collision": bool(info["collision"]),
                "timeout": bool(info["timeout"]),
                "actor_loss": stats.actor_loss,
                "critic_loss": stats.critic_loss,
                "entropy": stats.entropy,
                "bc_loss": stats.bc_loss,
            })
            self._append_history(history[-1])
            if log_interval > 0 and (episode == self.start_episode or episode % log_interval == 0):
                print(
                    f"episode={episode} reward={total_reward:.3f} steps={step + 1} "
                    f"success={bool(info['success'])} collision={bool(info['collision'])} timeout={bool(info['timeout'])} "
                    f"actor_loss={stats.actor_loss:.4f} critic_loss={stats.critic_loss:.4f} bc_loss={stats.bc_loss:.4f}"
                )
            if checkpoint_interval > 0 and bool(self.config["training"].get("save_checkpoint", True)):
                if episode % checkpoint_interval == 0:
                    self.last_checkpoint_path = self.save_checkpoint(episode=episode)
        if bool(self.config["training"].get("save_checkpoint", True)):
            self.last_checkpoint_path = self.save_checkpoint()
        return history

    def _apply_regulators(self, episode: int, total_episodes: int) -> None:
        delta_alpha, delta_distance = angle_distance_bias(episode, total_episodes, self.config)
        self.env.delta_alpha = delta_alpha
        self.env.delta_distance = delta_distance
        self.env.base_obstacles = [
            Obstacle(
                o.x,
                o.y,
                obstacle_radius(episode, total_episodes, o.radius, self.config),
                obstacle_buffer(episode, total_episodes, o.radius, o.buffer, self.config),
            )
            for o in self.original_obstacles
        ]

    def save_checkpoint(self, path: str | None = None, episode: int | None = None) -> str:
        """Save actor/critic parameters after training."""
        if path is None:
            checkpoint_dir = Path(str(self.config["training"].get("checkpoint_dir", "checkpoints")))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            name = "mra_rlec_latest.pt" if episode is None else f"mra_rlec_ep{episode:07d}.pt"
            path = str(checkpoint_dir / name)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.updater.optimizer.state_dict(),
                "episode": episode,
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load actor/critic parameters for continued training or evaluation."""
        # PyTorch 2.6 defaults to weights_only=True; this checkpoint contains
        # optimizer state and config from our own training run, so load explicitly.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.updater.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("episode") is not None:
            self.start_episode = int(checkpoint["episode"]) + 1

    def _append_history(self, row: Dict[str, Any]) -> None:
        fieldnames = [
            "episode",
            "reward",
            "steps",
            "guide_steps",
            "success",
            "collision",
            "timeout",
            "actor_loss",
            "critic_loss",
            "entropy",
            "bc_loss",
        ]
        exists = self.history_csv.exists()
        with open(self.history_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})
