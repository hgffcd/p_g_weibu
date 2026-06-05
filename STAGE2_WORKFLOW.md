# Stage-2 Reproduction Workflow

Stage 2 adds harder experiments without modifying the existing Stage 1 source.

## New Files

- `stage2/envs.py`: randomized reset environment and vector wrapper.
- `stage2/trainer.py`: vector MAPPO trainer using Stage 2 environments.
- `stage2/train_vector.py`: Stage 2 training entry point.
- `stage2/evaluate.py`: fixed/randomized evaluation entry point.
- `stage2/visualize.py`: per-frame status visualization for guide/controller/checkpoint.
- `stage2/launch_4gpu_vector.sh`: 4-GPU launch script.
- `stage2/evaluate_4gpu.sh`: 4-GPU checkpoint evaluation script.
- `config_stage2_moving.yaml`: fixed obstacles and moving PFP target.
- `config_stage2_randomized.yaml`: moving PFP target plus randomized target/pursuer/obstacle resets.

## Recommended Order

1. Verify Stage 2 fixed moving-target guide. The moving configuration enables
   delayed formation correction because the plain APF guide reaches the target
   but does not satisfy the strict per-agent angular condition at target speed
   `0.10-0.15`.

```bash
python -m stage2.diagnose --config config_stage2_moving.yaml --scenario fixed --episodes 20 --speed-sweep 0.0,0.05,0.10,0.15
python -m stage2.evaluate --config config_stage2_moving.yaml --scenario fixed --policy guide --episodes 20 --output-dir results_stage2/guide_moving_fixed
python -m stage2.visualize --config config_stage2_moving.yaml --scenario fixed --policy guide --seed 0 --output results_stage2/visualization/guide_moving_fixed.gif
```

2. Train fixed moving-target residual policy:

```bash
python -m stage2.pretrain --config config_stage2_moving.yaml --scenario fixed
MONITOR_INTERVAL=30 STARTUP_TIMEOUT=600 bash stage2/launch_4gpu_vector.sh config_stage2_moving.yaml checkpoints/pretrain_actor.pt stage2_moving
```

3. Evaluate trained moving-target checkpoints:

```bash
bash stage2/evaluate_4gpu.sh stage2_moving fixed 200 results_stage2/stage2_moving_fixed_eval
python -m stage2.visualize --config stage2_generated_configs/stage2_moving_gpu1.yaml --scenario fixed --policy checkpoint --checkpoint checkpoints_stage2_moving/gpu1/mra_rlec_best.pt --seed 0 --output results_stage2/visualization/gpu1_moving_fixed.gif
```

4. Only after fixed moving target is stable, train randomized Stage 2:

```bash
MONITOR_INTERVAL=30 STARTUP_TIMEOUT=600 bash stage2/launch_4gpu_vector.sh config_stage2_randomized.yaml checkpoints/pretrain_actor.pt stage2_randomized
bash stage2/evaluate_4gpu.sh stage2_randomized randomized 300 results_stage2/stage2_randomized_eval
```

5. Before expensive randomized training, run the randomized ablation probes.
   These configs keep the original implementation untouched and test whether
   failures come from target randomization alone or from the harder pursuer /
   obstacle randomization distribution.

```bash
bash stage2/randomized_ablation.sh 100 results_stage2/randomized_ablation
```

If `config_stage2_randomized_easy.yaml` is stable enough, use it for the first
randomized training run:

```bash
MONITOR_INTERVAL=30 STARTUP_TIMEOUT=600 bash stage2/launch_4gpu_vector.sh config_stage2_randomized_easy.yaml checkpoints/pretrain_actor.pt stage2_randomized_easy
bash stage2/evaluate_4gpu.sh stage2_randomized_easy randomized 300 results_stage2/stage2_randomized_easy_eval
```

If both randomized ablation probes remain below about 0.85 guide success, do
not train yet. Run the narrower curriculum probes first:

```bash
bash stage2/curriculum_probe.sh 100 results_stage2/curriculum_probe
```

Only start training from the first probe whose guide reaches a stable high
success rate. This keeps the curriculum anchored to a feasible guide policy.

6. If the narrow probes still timeout, use the obstacle-channel guide probe. It
   adds a two-stage corridor approach below the obstacle band, then returns to
   the original formation guide with a stronger capture-ring contraction.

```bash
python -m stage2.channel_probe \
  --configs config_stage2_channel_probe.yaml \
  --episodes 100 \
  --speed-sweep 0.0,0.15 \
  --output-dir results_stage2/channel_probe \
  --gif
```

Use this only as a guide feasibility probe first. Integrate it into training
after the server-side probe confirms high success without collision.

7. Train a residual policy around the obstacle-channel guide:

```bash
MONITOR_INTERVAL=30 STARTUP_TIMEOUT=600 bash stage2/launch_4gpu_channel_vector.sh config_stage2_channel_probe.yaml checkpoints/pretrain_actor.pt stage2_channel
bash stage2/evaluate_4gpu_channel.sh stage2_channel randomized 300 results_stage2/stage2_channel_eval
```

This is not the unmodified paper guide. Report it as the Stage-2 engineering
extension: channel waypoint guide plus residual MAPPO.

The channel trainer keeps the validated obstacle geometry during rollout. Do
not reuse generated configs from an older `stage2_channel` launch after editing
`stage2/channel_trainer.py`; remove or overwrite `stage2_generated_configs`
with a new run name.

## Interpretation

Stage 2 should be reported separately from Stage 1:

- Stage 1: fixed static target, fixed obstacles.
- Stage 2 moving: fixed obstacles, moving target.
- Stage 2 randomized: moving target plus randomized feasible reset distribution.

Do not claim arbitrary-scene guarantees. These are empirical evaluations over the configured reset distribution.
