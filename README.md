# Geometry-Aware Sharpness Minimization: Invariance and Generalization in SAM Variants

## Motivation

Recent advances in deep learning optimization highlight the importance of sharpness-aware methods
that seek flat minima to improve generalization. One such method, Sharpness-Aware Minimization
(SAM) [Foret et al., 2021], has shown promising results. However, it is sensitive to the choice of
model parameterization, as its perturbation radius is defined in the Euclidean metric.
Adaptive SAM (ASAM) [Kwon et al., 2021] and Monge SAM (M-SAM) [Jacobsen and Arvanitidis,
2025] try to address these limitations by introducing scale invariance and geometry-aware metrics,
respectively. However, existing evaluations of M-SAM are limited to a single fine-tuning task on
ResNet-18 and a multi-modal alignment task; a systematic comparison of all three optimizers under
controlled reparametrization across multiple architectures has not been done.
Understanding how these methods behave under reparametrization is crucial for developing more
robust optimization techniques, and this gap motivates our empirical work on benchmarking SAM,
ASAM, and M-SAM on ResNet-18 and ViT-B/32 architectures with a focus on reparametrization
invariance and generalization performance.


## Running the experiments

GPU instances are rented on [Vast.ai](https://vast.ai).
Each experiment script is self-contained and writes results under `results/{model}/`.
If a run is interrupted, re-running the same script will skip already-completed checkpoints automatically.

### Setup (on the GPU instance)

```bash
# 1. Clone the repo
git clone <repo-url> && cd sam-opt-research

# 2. Install dependencies
uv sync

# 3. Start a tmux session so training survives SSH disconnects
touch ~/.no_auto_tmux   # required on some Vast.ai templates
tmux new-session -A -s monitor_gpu # create a new session named 'monitor_gpu' or attach if it already exists
watch -n 1 nvidia-smi         # monitor GPU usage every second

tmux attach -t monitor_gpu    # to re-attach to the GPU monitoring session
# Ctrl+B d  to detach
```

### Experiment 1 — Baseline accuracy (ResNet-18 and ViT)

Run each model on a same GPU instance in parallel. The models are small enough that they can be trained simultaneously without significant slowdown, and this will speed up the overall experiment time.

```bash
tmux new-session -A -s resnet_baseline_train  # create a new session named 'resnet_baseline_train' or attach if it already exists
# Ctrl+B d  to detach

# Instance 1 — ResNet-18
bash experiments/run_resnet18_baseline.sh

tmux attach -t resnet_baseline_train          # to re-attach to the training session
# Ctrl+B d  to detach

tmux new-session -A -s vit_baseline_train  # create a new session named 'vit_baseline_train' or attach if it already exists
# Ctrl+B d  to detach

# Instance 2 — ViT-B/32
bash experiments/run_vit_baseline.sh

tmux attach -t vit_baseline_train          # to re-attach to the training session
# Ctrl+B d  to detach
```

Results are saved to `results/resnet18/` and `results/vit_b_32/` respectively.

### Experiment 2 — Reparametrisation invariance

```bash
tmux new-session -A -s resnet_reparam_train  # create a new session named 'resnet_reparam_train' or attach if it already exists
# Ctrl+B d  to detach

# Instance 1 — ResNet-18
bash experiments/run_resnet18_reparam.sh

tmux attach -t resnet_reparam_train          # to re-attach to the training session
# Ctrl+B d  to detach

tmux new-session -A -s vit_reparam_train  # create a new session named 'vit_reparam_train' or attach if it already exists
# Ctrl+B d  to detach

# Instance 2 — ViT-B/32
bash experiments/run_vit_reparam.sh

tmux attach -t vit_reparam_train          # to re-attach to the training session
# Ctrl+B d  to detach
```

### Experiment 3 — Flatness / sharpness analysis

Requires checkpoints from Experiment 1 to exist first.

```bash
# Single checkpoint
python experiments/run_flatness.py \
    --config configs/resnet18_baseline_run.yaml \
    --checkpoint results/resnet18/experiments/baseline/resnet18/checkpoints/sam_rho0.05_seed42.pt

# Batch mode (all checkpoints)
python experiments/run_flatness.py \
    --config configs/resnet18_baseline_run.yaml \
    --ckpt-dir results/resnet18/experiments/baseline/resnet18/checkpoints \
    --out-dir  results/resnet18/experiments/flatness/resnet18
```

### GPU utilization tips

- Training uses BF16 mixed precision and `torch.compile` automatically on CUDA.
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Container disk: **30 GB** minimum (ViT checkpoints alone are ~10 GB).
- Configs are under `configs/`. Each `*_run.yaml` variant has model-specific result paths.
- Use large batch sizes (e.g. 256) to maximize GPU utilization, especially for ViT.
- If GPU utlisation is low, check for CPU bottlenecks (data loading, logging) and consider increasing `num_workers` in the config.
- For A100 80 GB, you can run both ResNet-18 and ViT experiments simultaneously without significant slowdown, which will speed up overall experiment time.


### Important notes from the discussion with Mentor:

- SAM approaches require two forward-backward passes per iteration, which can be computationally expensive. We should consider the trade-off between the potential performance gains from SAM and the increased computational cost. It might be worth experimenting with a smaller subset of the data or a simpler model to see if SAM provides significant benefits before fully committing to it.
