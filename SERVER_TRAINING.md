# Server Training Guide

Target environment: `/home/user/anaconda3/envs/phf_env`.

## Dependency Check

Your listed environment already contains all packages needed for training:

- Python 3.10.16
- torch 2.6.0
- numpy 1.26.4
- PyYAML 6.0.3
- scipy / pandas / matplotlib are available if later plotting is needed

No extra package is required for training or metrics. `pytest` is not listed; only install it if you want to run pytest tests:

```bash
pip install pytest
```

You can also run tests without pytest:

```bash
python -m unittest discover -s tests
```

## Recommended Smoke Check

```bash
conda activate phf_env
python -m unittest discover -s tests
python main.py metrics --config config_server.yaml --policy controller --episodes 3 --output-dir results/controller_smoke
```

## Start Training

Recommended four-GPU multi-seed run with APF behavior-cloning pretraining:

```bash
conda activate phf_env
python main.py pretrain --config config_vector_server.yaml
python -m experiments.diagnose_guide --config config_vector_server.yaml --episodes 20 --schedule-step 1 --schedule-total 2000
bash scripts/launch_4gpu_vector.sh config_vector_server.yaml checkpoints/pretrain_actor.pt
```

If the diagnostic prints a high `collision_rate` at `schedule-step 1`, do not
start four-GPU training; send the diagnostic output back for analysis.

The four-GPU script stays in the foreground and prints the latest log line for
each GPU every 60 seconds. To change the interval:

```bash
MONITOR_INTERVAL=30 bash scripts/launch_4gpu_vector.sh config_vector_server.yaml checkpoints/pretrain_actor.pt
```

The server vector defaults are intentionally CPU-aware: `num_envs=16`,
`rollout_length=64`, and one PyTorch CPU thread per process. This keeps the
Python planar environment responsive enough that `update=1` should appear
quickly in the logs.

Vectorized single-GPU fallback:

```bash
conda activate phf_env
python main.py pretrain --config config_vector_server.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/server_train_vector.sh config_vector_server.yaml server_runs_vector checkpoints/pretrain_actor.pt
```

Legacy single-environment run:

```bash
conda activate phf_env
python main.py pretrain --config config_server.yaml
bash scripts/server_train.sh config_server.yaml server_runs checkpoints/pretrain_actor.pt
```

Manual equivalent:

```bash
python main.py train --config config_server.yaml 2>&1 | tee server_runs/train_stdout.log
```

Resume from a checkpoint:

```bash
python main.py train --config config_server.yaml --resume checkpoints/mra_rlec_ep0000500.pt
```

## Evaluate Trained Checkpoint

Evaluate all four GPU runs. The script uses `mra_rlec_best.pt` when it exists,
otherwise `mra_rlec_latest.pt`:

```bash
bash scripts/evaluate_4gpu_vector.sh 100 results/vector_eval
```

```bash
python main.py metrics \
  --config config_vector_gpu0.yaml \
  --policy checkpoint \
  --checkpoint checkpoints_vector/gpu0/mra_rlec_best.pt \
  --episodes 100 \
  --output-dir results/gpu0_eval
```

Batch planar suite:

```bash
python -m experiments.run_planar_suite \
  --config config_vector_server.yaml \
  --policy checkpoint \
  --checkpoint checkpoints_vector/mra_rlec_latest.pt \
  --episodes 100 \
  --output-dir results/planar_checkpoint_suite
```

## Files to Send Back for Analysis

I cannot directly read your server unless you expose/mount it into this chat environment. Send back one archive containing:

- `logs/train_history.csv`
- `server_runs/train_stdout.log`
- `server_runs/server_probe.txt`
- `server_runs/nvidia-smi.txt`
- `server_runs/conda-list.txt`
- `results/**`
- `checkpoints_vector/**/mra_rlec_best.pt` and `checkpoints_vector/**/mra_rlec_latest.pt` if you want checkpoint-level debugging
- the exact `config_vector_server.yaml` and generated `config_vector_gpu*.yaml` used

Create the archive:

```bash
bash scripts/collect_server_artifacts.sh mra_rlec_server_artifacts.tar.gz
```

Then upload `mra_rlec_server_artifacts.tar.gz` here, or paste:

```bash
tail -n 80 server_runs/train_stdout.log
tail -n 20 logs/train_history.csv
cat results/checkpoint_eval/summary.json
```
