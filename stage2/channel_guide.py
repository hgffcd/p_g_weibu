"""Obstacle-channel guide for Stage-2 randomized target probes.

This is an engineering extension around the existing APF guide. The fixed guide
often leaves the middle pursuers trapped below the obstacle band when the target
initial position changes. This wrapper adds a temporary corridor waypoint before
returning control to the original formation guide.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from agents.apf_guide import APFGuidePolicy, limit_action_inf_norm
from envs.encirclement_env import EncirclementEnv
from utils.geometry import norm


class Stage2ChannelGuidePolicy:
    """APF guide plus obstacle-band corridor waypoints.

    The channel rule is intentionally simple and only activates while pursuers
    are below the obstacle band. It does not replace the paper-derived guide;
    it creates a feasible approach phase for randomized target positions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base = APFGuidePolicy(config)
        guide_cfg = config.get("channel_guide", {})
        env_cfg = config["environment"]
        self.enabled = bool(guide_cfg.get("enabled", True))
        self.channel_margin = float(guide_cfg.get("channel_margin", 0.45))
        self.channel_top_margin = float(guide_cfg.get("channel_top_margin", 0.75))
        self.channel_gain = float(guide_cfg.get("channel_gain", 3.0))
        self.channel_damping = float(guide_cfg.get("channel_damping", 1.4))
        self.channel_weight = float(guide_cfg.get("channel_weight", 1.0))
        self.release_y_margin = float(guide_cfg.get("release_y_margin", 0.12))
        self.post_formation_boost = float(guide_cfg.get("post_formation_boost", 1.5))
        self.max_acceleration = float(env_cfg["max_acceleration"])

    def act(self, env: EncirclementEnv) -> np.ndarray:
        base_actions = self.base.act(env)
        if not self.enabled or not env.obstacles:
            return base_actions

        channel_y = self._channel_y(env)
        channel_actions = base_actions.copy()
        ranks = self._rank_by_x(env)
        corridor_x = self._corridor_x(env)
        active_channel = False

        for rank, pursuer_idx in enumerate(ranks):
            if rank == env.num_pursuers // 2:
                continue
            if env.pursuer_pos[pursuer_idx, 1] >= channel_y - self.release_y_margin:
                continue
            active_channel = True
            waypoint = self._waypoint(env, pursuer_idx, corridor_x[rank], channel_y)
            desired_vel = env.target_vel
            waypoint_action = self.channel_gain * (waypoint - env.pursuer_pos[pursuer_idx])
            waypoint_action -= self.channel_damping * (env.pursuer_vel[pursuer_idx] - desired_vel)
            channel_actions[pursuer_idx] = (
                (1.0 - self.channel_weight) * base_actions[pursuer_idx]
                + self.channel_weight * waypoint_action
            )

        if not active_channel and self.post_formation_boost > 0.0:
            channel_actions = channel_actions + self.post_formation_boost * self.base._formation_slot_actions(env)

        # Reuse the existing short-range safety layer after waypoint injection.
        if getattr(self.base, "safety_filter", False):
            channel_actions = self.base._safety_filter(env, channel_actions)
        return np.vstack([limit_action_inf_norm(action, self.max_acceleration) for action in channel_actions])

    def _rank_by_x(self, env: EncirclementEnv) -> np.ndarray:
        return np.argsort(env.pursuer_pos[:, 0])

    def _channel_y(self, env: EncirclementEnv) -> float:
        top = max(o.y + o.radius + o.buffer for o in env.obstacles)
        return min(float(env.target_pos[1] - 0.65), float(top + self.channel_top_margin))

    def _low_pass_y(self, env: EncirclementEnv) -> float:
        bottom = min(o.y - o.radius - o.buffer for o in env.obstacles)
        return float(bottom - 0.25)

    def _waypoint(self, env: EncirclementEnv, pursuer_idx: int, corridor_x: float, channel_y: float) -> np.ndarray:
        pos = env.pursuer_pos[pursuer_idx]
        low_y = self._low_pass_y(env)
        if pos[1] <= low_y and abs(pos[0] - corridor_x) > 0.25:
            # First move laterally below the obstacle band. A direct diagonal
            # route to the channel top cuts through the obstacle safety region.
            return np.array([corridor_x, pos[1]], dtype=float)
        return np.array([corridor_x, channel_y], dtype=float)

    def _corridor_x(self, env: EncirclementEnv) -> np.ndarray:
        left = min(o.x - o.radius - o.buffer for o in env.obstacles) - self.channel_margin
        right = max(o.x + o.radius + o.buffer for o in env.obstacles) + self.channel_margin
        center = float(env.target_pos[0])
        offsets = np.array([-0.24, 0.24], dtype=float)
        return np.array([left + offsets[0], left + offsets[1], center, right + offsets[0], right + offsets[1]], dtype=float)
