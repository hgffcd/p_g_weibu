"""Stage-2 visualization with per-frame status text."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from agents.apf_guide import APFGuidePolicy
from agents.encirclement_controller import PlanarEncirclementController
from stage2.channel_guide import Stage2ChannelGuidePolicy
from stage2.envs import Stage2EncirclementEnv
from stage2.evaluate import checkpoint_actor, configure_scenario
from utils.config import load_config


def collect_trajectory(
    config: Dict[str, Any],
    policy: str,
    checkpoint: str | None,
    seed: int,
) -> Dict[str, Any]:
    env = Stage2EncirclementEnv(config)
    guide = (
        Stage2ChannelGuidePolicy(config)
        if config.get("channel_guide", {}).get("enabled", False)
        else APFGuidePolicy(config)
    )
    controller = PlanarEncirclementController(config) if policy == "controller" else None
    actor = None
    device = None
    if policy == "checkpoint":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for checkpoint visualization")
        actor, device = checkpoint_actor(config, checkpoint)

    obs, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(env.num_pursuers, device) if actor is not None else None
    pursuer_frames = [env.pursuer_pos.copy()]
    target_frames = [env.target_pos.copy()]
    status_frames: List[Dict[str, Any]] = [{"step": 0, "success": False, "collision": False, "timeout": False}]
    rewards = []
    final_info: Dict[str, Any] = {"success": False, "collision": False, "timeout": False}

    for step in range(1, int(config["environment"]["max_steps"]) + 1):
        if policy == "guide":
            action = guide.act(env)
        elif policy == "controller":
            if controller is None:
                raise RuntimeError("Controller was not initialized")
            action = controller.act(env)
        elif policy == "checkpoint":
            import torch

            assert actor is not None and device is not None and hidden is not None
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, hidden = actor.act(obs_tensor, hidden, deterministic=True)
            actor_action = action_tensor.detach().cpu().numpy()
            policy_mode = str(config.get("training", {}).get("policy_mode", "direct")).lower()
            if policy_mode == "residual":
                residual_scale = float(config.get("training", {}).get("residual_scale", 1.0))
                action = guide.act(env) + residual_scale * actor_action
            else:
                action = actor_action
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, done, info = env.step(action)
        final_info = info
        rewards.append(float(np.mean(reward)))
        pursuer_frames.append(env.pursuer_pos.copy())
        target_frames.append(env.target_pos.copy())
        status_frames.append({
            "step": step,
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "timeout": bool(info.get("timeout", False)),
        })
        if done:
            break

    return {
        "pursuer_pos": np.asarray(pursuer_frames, dtype=float),
        "target_pos": np.asarray(target_frames, dtype=float),
        "status": status_frames,
        "obstacles": [(o.x, o.y, o.radius, o.buffer) for o in env.obstacles],
        "capture_distance": env.capture_distance,
        "pursuer_radius": env.pursuer_radius,
        "target_radius": env.target_radius,
        "policy": policy,
        "seed": seed,
        "final_info": final_info,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reset_info": env.last_reset_info,
    }


def save_animation(trajectory: Dict[str, Any], output_path: str | Path, fps: int = 12) -> Path:
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
        x_min = min(x_min, x - radius - buffer - 0.4)
        x_max = max(x_max, x + radius + buffer + 0.4)
        y_min = min(y_min, y - radius - buffer - 0.4)
        y_max = max(y_max, y + radius + buffer + 0.4)

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
    target_trail, = ax.plot([], [], color="#1b9e77", lw=1.2, alpha=0.55)

    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#17becf", "#8c564b", "#e377c2"]
    pursuer_patches = []
    trails = []
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
        target_trail.set_data(targets[: frame + 1, 0], targets[: frame + 1, 1])
        for idx, patch in enumerate(pursuer_patches):
            xy = pursuers[frame, idx]
            patch.center = (xy[0], xy[1])
            trails[idx].set_data(pursuers[: frame + 1, idx, 0], pursuers[: frame + 1, idx, 1])
        item = trajectory["status"][frame]
        status.set_text(
            f"policy={trajectory['policy']} seed={trajectory['seed']}\n"
            f"step={item['step']} success={item['success']} "
            f"collision={item['collision']} timeout={item['timeout']}"
        )
        return [capture_ring, target_circle, target_trail, status, *pursuer_patches, *trails]

    anim = animation.FuncAnimation(fig, update, frames=len(pursuers), interval=max(1, int(1000 / fps)), blit=True)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        output.write_text(anim.to_jshtml(), encoding="utf-8")
    elif suffix == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError("MP4 output requires ffmpeg. Use .gif/.html if ffmpeg is unavailable.")
        anim.save(output, writer=animation.FFMpegWriter(fps=fps), dpi=150)
    else:
        plt.close(fig)
        raise ValueError("Unsupported output suffix. Use .html, .gif, or .mp4.")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 trajectory visualization")
    parser.add_argument("--config", default="config_stage2_moving.yaml")
    parser.add_argument("--scenario", choices=["fixed", "randomized"], default="fixed")
    parser.add_argument("--policy", choices=["guide", "controller", "checkpoint"], default="guide")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results_stage2/visualization/stage2_episode0.gif")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    config = configure_scenario(load_config(args.config), args.scenario)
    trajectory = collect_trajectory(config, args.policy, args.checkpoint, args.seed)
    output = save_animation(trajectory, args.output, fps=args.fps)
    final_info = trajectory["final_info"]
    print(
        f"saved={output} frames={len(trajectory['pursuer_pos'])} policy={args.policy} "
        f"success={final_info.get('success', False)} collision={final_info.get('collision', False)} "
        f"timeout={final_info.get('timeout', False)} mean_reward={trajectory['mean_reward']:.3f}"
    )


if __name__ == "__main__":
    main()
