"""Stage-2 curriculum helpers."""

from __future__ import annotations

from typing import Any, Dict


def target_speed_for_progress(step: int, total_steps: int, config: Dict[str, Any]) -> float:
    """Linearly increase target speed for moving-target fine tuning.

    This follows the paper-level Fine Tune idea: target speed is gradually
    increased instead of starting from the hardest moving target.
    """
    env_cfg = config["environment"]
    cfg = config.get("stage2", {}).get("target_speed_curriculum", {})
    end_speed = float(cfg.get("end", env_cfg.get("target_max_velocity", 0.0)))
    if not bool(cfg.get("enabled", False)):
        return end_speed

    start_speed = float(cfg.get("start", 0.0))
    start_fraction = float(cfg.get("start_fraction", 0.0))
    end_fraction = float(cfg.get("end_fraction", 1.0))
    if total_steps <= 1:
        return end_speed
    progress = max(0.0, min(1.0, (float(step) - 1.0) / max(float(total_steps) - 1.0, 1.0)))
    if progress <= start_fraction:
        return start_speed
    if progress >= end_fraction:
        return end_speed
    span = max(end_fraction - start_fraction, 1e-8)
    alpha = (progress - start_fraction) / span
    return start_speed + alpha * (end_speed - start_speed)
