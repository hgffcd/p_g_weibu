# Reproduction Results

## Hardware / Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, about 8 GB VRAM.
- Driver/CUDA from `nvidia-smi`: CUDA 12.9.
- Conda env: `rl_env`.
- PyTorch: `2.8.0+cu129`.

Conclusion: the simplified planar-circle reproduction can run on this laptop GPU. Full paper-scale training is expected to be long, but the current rMAPPO-sized networks and 5-pursuer environment fit in memory.

## Implemented Result Metrics

The code now outputs the paper's Section V style indicators:

- average step reward
- encirclement time
- danger rate
- success rate
- collision rate
- timeout rate
- final distance error
- final angle error
- minimum pursuer-obstacle clearance
- minimum pursuer-pursuer clearance
- bidirectional vs fully connected communication meta-message cost

Outputs are written to:

- `results/*/summary.json`
- `results/*/episodes.csv`
- `results/planar_controller_suite/*.csv`
- `results/planar_controller_suite/summary_tables.json`

## Commands Run

```bash
conda run -n rl_env python -m pytest tests -q
conda run -n rl_env python main.py train --config config.yaml
conda run -n rl_env python main.py metrics --config config.yaml --policy checkpoint --checkpoint checkpoints\mra_rlec_latest.pt --episodes 3 --output-dir results\checkpoint_200_after_train
conda run -n rl_env python -m experiments.run_planar_suite --config config.yaml --policy controller --episodes 3 --output-dir results\planar_controller_suite --pursuer-counts 3,4,5,6 --target-speeds 0.05,0.15,0.25 --target-policies static,greedy,pfp --obstacle-counts 0,4
```

## Test Result

```text
11 passed
```

## Current Short-Run Findings

The current short training run is only an end-to-end validation, not a paper-level trained model.

- rMAPPO checkpoint after 3 episodes:
  - success rate: 0.0
  - collision rate: 0.0
  - timeout rate: 1.0
  - final distance error: 14.13
  - final angle error: 1.18

- Planar deterministic controller sanity check, no obstacles:
  - success rate: 1.0
  - encirclement time: 10.6 s
  - collision rate: 0.0
  - final distance error: 0.051
  - final angle error: 0.083

- Planar deterministic controller with generated 4-obstacle case:
  - success rate: 0.0
  - collision rate: 0.0
  - timeout rate: 1.0
  - final distance error: 0.178
  - final angle error: 0.363

Interpretation: the environment, success condition, metrics, checkpointing, and result table pipeline are working. The learned policy has not yet been trained enough to match the paper's multiobstacle results.

## Full Reproduction Next Run

To move toward paper-level results on the simplified planar-circle environment:

1. Increase `training.episodes` substantially.
2. Keep `environment.max_steps = 200`.
3. Use curriculum regulators already in `config.yaml`.
4. Evaluate with `experiments.run_planar_suite`.

Suggested first long run:

```bash
conda activate rl_env
python main.py train --config config.yaml
python -m experiments.run_planar_suite --config config.yaml --policy checkpoint --checkpoint checkpoints/mra_rlec_latest.pt --episodes 100 --output-dir results/planar_checkpoint_suite
```

For a real paper-scale attempt, set training episodes so total environment steps approach the paper's Base and Fine Tune scales. The original paper reports Base `7.68e6` steps and Fine Tune `5.12e6` steps; that is much larger than the current smoke run.
