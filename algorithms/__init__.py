"""Algorithm package exports with lazy PyTorch training imports."""

from algorithms.regulators import angle_distance_bias, guide_steps, jsrl_regulator, obstacle_radius

__all__ = [
    "MRARLECTrainer",
    "TorchMRARLECTrainer",
    "angle_distance_bias",
    "guide_steps",
    "jsrl_regulator",
    "obstacle_radius",
]


def __getattr__(name: str):
    if name in {"MRARLECTrainer", "TorchMRARLECTrainer"}:
        from algorithms.mra_rlec import MRARLECTrainer, TorchMRARLECTrainer

        return {"MRARLECTrainer": MRARLECTrainer, "TorchMRARLECTrainer": TorchMRARLECTrainer}[name]
    raise AttributeError(f"module 'algorithms' has no attribute {name!r}")
