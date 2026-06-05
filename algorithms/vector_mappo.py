"""Vectorized MAPPO-style trainer for the planar MRA-RLEC reproduction.

This is the server-oriented trainer: many synchronous planar environments collect
rollouts before each PPO update. It is still dependency-light and compatible with
the provided phf_env package versions.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from agents.apf_guide import APFGuidePolicy
from agents.torch_networks import RecurrentActor, RecurrentCritic
from algorithms.ppo import PPOUpdater
from algorithms.regulators import angle_distance_bias, guide_steps, obstacle_buffer, obstacle_radius
from algorithms.vector_rollout_buffer import VectorRolloutBuffer
from envs.encirclement_env import EncirclementEnv, Obstacle
from envs.vector_env import VectorizedEncirclementEnv


class VectorMAPPOTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        train_cfg = config["training"]
        vec_cfg = config.get("vector_training", {})
        torch_num_threads = int(train_cfg.get("torch_num_threads", 1))
        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
            try:
                torch.set_num_interop_threads(max(1, int(train_cfg.get("torch_num_interop_threads", 1))))
            except RuntimeError:
                # PyTorch allows setting interop threads only once per process.
                pass
        self.num_envs = int(vec_cfg.get("num_envs", 32))
        self.rollout_length = int(vec_cfg.get("rollout_length", 128))
        self.total_updates = int(vec_cfg.get("updates", 1000))
        self.curriculum_progress = str(vec_cfg.get("curriculum_progress", "episodes"))
        self.eval_interval = int(vec_cfg.get("eval_interval", 0))
        self.eval_episodes = int(vec_cfg.get("eval_episodes", 10))
        self.best_checkpoint_name = str(vec_cfg.get("best_checkpoint_name", "mra_rlec_best.pt"))
        self.policy_mode = str(train_cfg.get("policy_mode", "direct")).lower()
        self.residual_scale = float(train_cfg.get("residual_scale", 1.0))
        self.eval_only_device = str(train_cfg.get("device", "auto"))
        device_name = self.eval_only_device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.envs = VectorizedEncirclementEnv(config, self.num_envs)
        self.guide = APFGuidePolicy(config)
        obs, states = self.envs.reset(seed=int(train_cfg.get("seed", 0)))
        self.original_obstacles = list(self.envs.envs[0].base_obstacles)

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
            state_dim=states.shape[-1],
            hidden_units=network_cfg["critic_hidden_units"],
            rnn_hidden_units=int(network_cfg["critic_rnn_hidden_units"]),
        ).to(self.device)

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

        self.start_update = 1
        self.completed_episodes = 0
        self.last_checkpoint_path: str | None = None
        self.best_checkpoint_path: str | None = None
        self.best_eval_success = -1.0
        self.best_eval_collision = 1.0
        self.best_eval_timeout = 1.0
        self.best_eval_reward = float("-inf")
        self.log_dir = Path(str(train_cfg.get("log_dir", "logs")))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.log_dir / "vector_train_history.csv"

    def train(self) -> List[Dict[str, Any]]:
        train_cfg = self.config["training"]
        env_cfg = self.config["environment"]
        seed = int(train_cfg.get("seed", 0))
        gamma = float(train_cfg["gamma"])
        gae_lambda = float(train_cfg["gae_lambda"])
        ppo_epochs = int(train_cfg["ppo_epochs"])
        batch_size = int(train_cfg["batch_size"])
        checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
        log_interval = int(train_cfg.get("log_interval", 10))
        max_steps = int(env_cfg["max_steps"])
        total_episodes_for_curriculum = int(train_cfg.get("episodes", self.total_updates * self.num_envs))
        history: List[Dict[str, Any]] = []

        observations, states = self.envs.reset(seed=seed)
        hidden = self.actor.initial_hidden(self.num_envs * self.envs.num_pursuers, self.device)

        for update in range(self.start_update, self.total_updates + 1):
            schedule_step, schedule_total = self._curriculum_step(update, total_episodes_for_curriculum)
            self._apply_curriculum(schedule_step, schedule_total)
            m_g = guide_steps(schedule_step, schedule_total, max_steps, self.config)
            buffer = VectorRolloutBuffer()
            reward_sum = 0.0
            success_count = 0
            collision_count = 0
            timeout_count = 0
            done_count = 0
            terminal_step_sum = 0.0
            guide_action_count = 0

            for _ in range(self.rollout_length):
                states = self.envs.flat_states()
                obs_tensor = torch.as_tensor(observations.reshape(-1, observations.shape[-1]), dtype=torch.float32, device=self.device)
                state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    values, _ = self.critic(state_tensor, None)
                    sampled_actions, _, next_hidden = self.actor.act(obs_tensor, hidden)

                actor_actions = sampled_actions.detach().cpu().numpy().reshape(self.num_envs, self.envs.num_pursuers, 2)
                guide_actions = self.envs.guide_actions(self.guide)
                guide_env_mask = self.envs.step_in_episode < m_g
                if self.policy_mode == "residual":
                    # Engineering assumption for stable handoff: the environment.py-style
                    # guide remains the nominal controller, and the network learns a bounded
                    # residual correction. During pure-guide JSRL steps, the residual target
                    # is zero and PPO excludes those samples from the actor objective.
                    policy_action_np = actor_actions.copy()
                    action_np = guide_actions + self.residual_scale * policy_action_np
                    policy_action_np[guide_env_mask] = 0.0
                    action_np[guide_env_mask] = guide_actions[guide_env_mask]
                    action_np = np.clip(action_np, -float(env_cfg["max_acceleration"]), float(env_cfg["max_acceleration"]))
                else:
                    policy_action_np = actor_actions.copy()
                    action_np = actor_actions.copy()
                    action_np[guide_env_mask] = guide_actions[guide_env_mask]
                    policy_action_np[guide_env_mask] = guide_actions[guide_env_mask]
                guide_mask = np.repeat(guide_env_mask[:, None], self.envs.num_pursuers, axis=1)
                guide_action_count += int(guide_mask.sum())

                action_tensor = torch.as_tensor(policy_action_np.reshape(-1, 2), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    log_probs, _ = self.actor.evaluate_actions(obs_tensor, action_tensor)
                log_probs_np = log_probs.detach().cpu().numpy().reshape(self.num_envs, self.envs.num_pursuers)

                next_obs, rewards, dones, infos, next_states = self.envs.step(action_np, seed_base=seed + update * 100000)
                team_rewards = rewards.mean(axis=1)
                buffer.add(
                    observations=observations,
                    actions=policy_action_np,
                    log_probs=log_probs_np,
                    rewards=team_rewards,
                    dones=dones,
                    states=states,
                    values=values.detach().cpu().numpy(),
                    guide_mask=guide_mask,
                )

                reward_sum += float(team_rewards.mean())
                for info in infos:
                    if info.get("success"):
                        success_count += 1
                    if info.get("collision"):
                        collision_count += 1
                    if info.get("timeout"):
                        timeout_count += 1
                    if "terminal_step_count" in info:
                        terminal_step_sum += float(info["terminal_step_count"])
                done_count += int(dones.sum())
                self.completed_episodes += int(dones.sum())

                # Reset recurrent hidden state for sub-envs that ended.
                hidden = next_hidden.detach()
                for env_idx, done in enumerate(dones):
                    if done:
                        start = env_idx * self.envs.num_pursuers
                        end = start + self.envs.num_pursuers
                        hidden[:, start:end, :] = 0.0
                observations = next_obs

            with torch.no_grad():
                last_states = torch.as_tensor(self.envs.flat_states(), dtype=torch.float32, device=self.device)
                last_values, _ = self.critic(last_states, None)
            batch = buffer.compute_returns_and_advantages(
                last_values.detach().cpu().numpy(),
                gamma,
                gae_lambda,
                self.device,
            )
            stats = self.updater.update(batch, epochs=ppo_epochs, minibatch_size=batch_size)

            denom = max(done_count, 1)
            row = {
                "update": update,
                "completed_episodes": self.completed_episodes,
                "mean_reward_per_step": reward_sum / self.rollout_length,
                "success_rate": success_count / denom,
                "collision_rate": collision_count / denom,
                "timeout_rate": timeout_count / denom,
                "done_count": done_count,
                "avg_terminal_steps": terminal_step_sum / denom,
                "guide_fraction": guide_action_count / max(self.rollout_length * self.num_envs * self.envs.num_pursuers, 1),
                "schedule_step": schedule_step,
                "schedule_total": schedule_total,
                "actor_loss": stats.actor_loss,
                "critic_loss": stats.critic_loss,
                "entropy": stats.entropy,
                "bc_loss": stats.bc_loss,
            }

            if self.eval_interval > 0 and (update == self.start_update or update % self.eval_interval == 0):
                eval_stats = self.evaluate_actor_only(self.eval_episodes, seed=seed + update * 1000)
                row.update({
                    "eval_success_rate": eval_stats["success_rate"],
                    "eval_collision_rate": eval_stats["collision_rate"],
                    "eval_mean_reward": eval_stats["mean_reward"],
                })
                self._save_best_if_needed(eval_stats, update)
            else:
                row.update({
                    "eval_success_rate": "",
                    "eval_collision_rate": "",
                    "eval_mean_reward": "",
                })
            history.append(row)
            self._append_history(row)

            if log_interval > 0 and (update == self.start_update or update % log_interval == 0):
                print(
                    f"update={update} mode={self.policy_mode} episodes={self.completed_episodes} "
                    f"reward={row['mean_reward_per_step']:.3f} success={row['success_rate']:.3f} "
                    f"collision={row['collision_rate']:.3f} timeout={row['timeout_rate']:.3f} "
                    f"steps={row['avg_terminal_steps']:.1f} guide={row['guide_fraction']:.3f} actor_loss={stats.actor_loss:.4f} "
                    f"critic_loss={stats.critic_loss:.4f} bc_loss={stats.bc_loss:.4f} "
                    f"eval_success={row['eval_success_rate']} eval_collision={row['eval_collision_rate']}"
                )

            if checkpoint_interval > 0 and bool(train_cfg.get("save_checkpoint", True)):
                if update % checkpoint_interval == 0:
                    self.last_checkpoint_path = self.save_checkpoint(update=update)

        if bool(train_cfg.get("save_checkpoint", True)):
            self.last_checkpoint_path = self.save_checkpoint()
        return history

    def _curriculum_step(self, update: int, total_episodes_for_curriculum: int) -> tuple[int, int]:
        """Map vectorized updates to Algorithm 1 curriculum progress.

        The paper schedules by episode count. In synchronous vector training, one
        update contains many short parallel episodes; using raw completed episode
        count makes Eqs. (11)-(14) decay far too quickly. The update mode is an
        engineering assumption for the vectorized server trainer.
        """
        if self.curriculum_progress == "updates":
            return max(1, update), max(1, self.total_updates)
        return max(1, min(self.completed_episodes + 1, total_episodes_for_curriculum)), total_episodes_for_curriculum

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

    def evaluate_actor_only(self, episodes: int, seed: int = 0) -> Dict[str, float]:
        """Deterministic decentralized execution for checkpoint selection.

        In direct mode this is actor-only. In residual mode the learned policy is
        evaluated as a residual around the environment.py-style guide controller.
        """
        env = EncirclementEnv(self.config)
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
                        guide_action = self.guide.act(env)
                        action = guide_action + self.residual_scale * actor_action
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

    def _save_best_if_needed(self, eval_stats: Dict[str, float], update: int) -> None:
        success = float(eval_stats["success_rate"])
        collision = float(eval_stats["collision_rate"])
        timeout = float(eval_stats.get("timeout_rate", 1.0))
        mean_reward = float(eval_stats.get("mean_reward", float("-inf")))
        improved = (
            success > self.best_eval_success
            or (success == self.best_eval_success and collision < self.best_eval_collision)
            or (
                success == self.best_eval_success
                and collision == self.best_eval_collision
                and timeout < self.best_eval_timeout
            )
            or (
                success == self.best_eval_success
                and collision == self.best_eval_collision
                and timeout == self.best_eval_timeout
                and mean_reward > self.best_eval_reward
            )
        )
        if not improved or not bool(self.config["training"].get("save_checkpoint", True)):
            return
        self.best_eval_success = success
        self.best_eval_collision = collision
        self.best_eval_timeout = timeout
        self.best_eval_reward = mean_reward
        checkpoint_dir = Path(str(self.config["training"].get("checkpoint_dir", "checkpoints")))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.save_checkpoint(str(checkpoint_dir / self.best_checkpoint_name), update=update)

    def save_checkpoint(self, path: str | None = None, update: int | None = None) -> str:
        if path is None:
            checkpoint_dir = Path(str(self.config["training"].get("checkpoint_dir", "checkpoints")))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            name = "mra_rlec_latest.pt" if update is None else f"mra_rlec_update{update:07d}.pt"
            path = str(checkpoint_dir / name)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.updater.optimizer.state_dict(),
                "update": update,
                "completed_episodes": self.completed_episodes,
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.updater.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("update") is not None:
            self.start_update = int(checkpoint["update"]) + 1
        if checkpoint.get("completed_episodes") is not None:
            self.completed_episodes = int(checkpoint["completed_episodes"])

    def _append_history(self, row: Dict[str, Any]) -> None:
        fieldnames = [
            "update",
            "completed_episodes",
            "mean_reward_per_step",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "done_count",
            "avg_terminal_steps",
            "guide_fraction",
            "schedule_step",
            "schedule_total",
            "eval_success_rate",
            "eval_collision_rate",
            "eval_mean_reward",
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
