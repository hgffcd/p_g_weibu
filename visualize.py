"""Visualize one planar encirclement episode as a browser HTML animation.

This file is intentionally independent from the user-provided ``environment.py``
renderer. The reproduction uses a simplified 2-D circle environment, so a
Matplotlib animation is enough to inspect trajectories, obstacles, the target,
and the capture ring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from agents.apf_guide import APFGuidePolicy
from agents.encirclement_controller import PlanarEncirclementController
from envs.encirclement_env import EncirclementEnv
from utils.config import load_config


def checkpoint_policy_action(
    config: Dict[str, Any],
    checkpoint: str,
    env: EncirclementEnv,
    obs: np.ndarray,
    hidden: Any,
    guide: APFGuidePolicy,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Run the trained recurrent actor used by the paper-style rMAPPO setup.

    The residual policy branch follows the server training implementation:
    the guide policy is the nominal controller and the network outputs a
    bounded residual correction around it.
    """
    import torch
    from algorithms.mra_rlec import TorchMRARLECTrainer

    if not hasattr(checkpoint_policy_action, "_cache"):
        checkpoint_policy_action._cache = {}  # type: ignore[attr-defined]
    cache = checkpoint_policy_action._cache  # type: ignore[attr-defined]
    cache_key = (Path(checkpoint).resolve(), id(config))
    if cache_key not in cache:
        trainer = TorchMRARLECTrainer(config)
        trainer.load_checkpoint(checkpoint)
        trainer.actor.eval()
        cache[cache_key] = (trainer.actor, trainer.device)
    actor, device = cache[cache_key]
    if hidden is None:
        hidden = actor.initial_hidden(env.num_pursuers, device)

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_tensor, _, next_hidden = actor.act(obs_tensor, hidden, deterministic=True)
    actor_action = action_tensor.detach().cpu().numpy()
    policy_mode = str(config.get("training", {}).get("policy_mode", "direct")).lower()
    if policy_mode == "residual":
        residual_scale = float(config.get("training", {}).get("residual_scale", 1.0))
        action = guide.act(env) + residual_scale * actor_action
    else:
        action = actor_action
    return action, obs, next_hidden


def rollout_episode(
    config: Dict[str, Any],
    policy: str,
    checkpoint: str | None,
    episode: int,
) -> Dict[str, Any]:
    """Collect one episode trajectory for guide/controller/checkpoint policy."""
    env = EncirclementEnv(config)
    guide = APFGuidePolicy(config)
    controller = PlanarEncirclementController(config) if policy == "controller" else None
    obs, _ = env.reset(seed=episode)
    hidden = None
    positions = [env.pursuer_pos.copy()]
    target_positions = [env.target_pos.copy()]
    rewards = []
    info: Dict[str, Any] = {"success": False, "collision": False, "timeout": False}

    for _ in range(int(config["environment"]["max_steps"])):
        if policy == "guide":
            actions = guide.act(env)
        elif policy == "controller":
            if controller is None:
                raise RuntimeError("Controller policy was not initialized")
            actions = controller.act(env)
        elif policy == "checkpoint":
            if not checkpoint:
                raise ValueError("--checkpoint is required for checkpoint visualization")
            actions, _, hidden = checkpoint_policy_action(config, checkpoint, env, obs, hidden, guide)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, done, info = env.step(actions)
        positions.append(env.pursuer_pos.copy())
        target_positions.append(env.target_pos.copy())
        rewards.append(float(np.mean(reward)))
        if done:
            break

    return {
        "pursuer_pos": np.asarray(positions, dtype=float),
        "target_pos": np.asarray(target_positions, dtype=float),
        "obstacles": [(o.x, o.y, o.radius, o.buffer) for o in env.obstacles],
        "capture_distance": env.capture_distance,
        "pursuer_radius": env.pursuer_radius,
        "target_radius": env.target_radius,
        "safety_buffer": env.safety_buffer,
        "dt": env.dt,
        "info": info,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def save_animation(trajectory: Dict[str, Any], output_path: str | Path, fps: int = 12) -> Path:
    """Save a planar episode animation.

    Supported suffixes:
    - .html: self-contained browser animation, no external encoder needed;
    - .gif: uses Matplotlib's Pillow writer;
    - .mp4: uses Matplotlib's ffmpeg writer and requires system ffmpeg.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pursuers = trajectory["pursuer_pos"]
    targets = trajectory["target_pos"]
    obstacles = trajectory["obstacles"]
    all_xy = np.concatenate([pursuers.reshape(-1, 2), targets.reshape(-1, 2)], axis=0)
    margin = 0.8 + float(trajectory["capture_distance"])
    x_min, y_min = np.min(all_xy, axis=0) - margin
    x_max, y_max = np.max(all_xy, axis=0) + margin
    for x, y, radius, buffer in obstacles:
        x_min = min(x_min, x - radius - buffer - margin * 0.25)
        x_max = max(x_max, x + radius + buffer + margin * 0.25)
        y_min = min(y_min, y - radius - buffer - margin * 0.25)
        y_max = max(y_max, y + radius + buffer + margin * 0.25)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    for x, y, radius, buffer in obstacles:
        ax.add_patch(Circle((x, y), radius + buffer, color="#d95f02", alpha=0.18, lw=0))
        ax.add_patch(Circle((x, y), radius, color="#d95f02", alpha=0.65, lw=1.0))

    capture_ring = Circle((targets[0, 0], targets[0, 1]), trajectory["capture_distance"], fill=False, ls="--", lw=1.5, color="#1b9e77")
    target_circle = Circle((targets[0, 0], targets[0, 1]), trajectory["target_radius"], color="#1b9e77", alpha=0.85)
    ax.add_patch(capture_ring)
    ax.add_patch(target_circle)

    pursuer_patches = []
    trails = []
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#17becf", "#8c564b", "#e377c2"]
    for idx in range(pursuers.shape[1]):
        color = colors[idx % len(colors)]
        patch = Circle((pursuers[0, idx, 0], pursuers[0, idx, 1]), trajectory["pursuer_radius"], color=color, alpha=0.9)
        line, = ax.plot([], [], color=color, lw=1.2, alpha=0.75)
        ax.add_patch(patch)
        pursuer_patches.append(patch)
        trails.append(line)

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def update(frame: int):
        target_xy = targets[frame]
        capture_ring.center = (target_xy[0], target_xy[1])
        target_circle.center = (target_xy[0], target_xy[1])
        for idx, patch in enumerate(pursuer_patches):
            xy = pursuers[frame, idx]
            patch.center = (xy[0], xy[1])
            trails[idx].set_data(pursuers[: frame + 1, idx, 0], pursuers[: frame + 1, idx, 1])
        status.set_text(
            f"step={frame}  success={trajectory['info'].get('success', False)}  "
            f"collision={trajectory['info'].get('collision', False)}  "
            f"timeout={trajectory['info'].get('timeout', False)}"
        )
        return [capture_ring, target_circle, status, *pursuer_patches, *trails]

    anim = animation.FuncAnimation(fig, update, frames=len(pursuers), interval=max(1, int(1000 / fps)), blit=True)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        output.write_text(anim.to_jshtml(), encoding="utf-8")
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer, dpi=120)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError("MP4 output requires the system ffmpeg command. Use .html/.gif or install ffmpeg.")
        writer = animation.FFMpegWriter(fps=fps, metadata={"artist": "mra-rlec-reproduction"})
        anim.save(output, writer=writer, dpi=150)
    else:
        plt.close(fig)
        raise ValueError("Unsupported output suffix. Use .html, .gif, or .mp4.")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one planar MRA-RLEC episode")
    parser.add_argument("--config", default="config_vector_server.yaml")
    parser.add_argument("--policy", choices=["guide", "controller", "checkpoint"], default="guide")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default="results/visualization/episode.html")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    trajectory = rollout_episode(config, args.policy, args.checkpoint, args.episode)
    output = save_animation(trajectory, args.output, fps=args.fps)
    info = trajectory["info"]
    print(
        f"saved={output} steps={len(trajectory['pursuer_pos']) - 1} "
        f"success={info.get('success', False)} collision={info.get('collision', False)} "
        f"timeout={info.get('timeout', False)} mean_reward={trajectory['mean_reward']:.3f}"
    )


if __name__ == "__main__":
    main()
