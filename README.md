# Invariance and Generalization in Sharpness-Aware Minimization (SAM) Variants

Stanford CS229 Project — Manav Dahra

## Overview

We study whether the generalisation benefits of SAM variants survive function-preserving weight
reparametrisation. We train ResNet-18 and ViT-B/32 on CIFAR-10 and compare four optimisers—
SGD, SAM, Adaptive SAM (ASAM), and Monge SAM (M-SAM)—across two initialisation scales
($\alpha \in \{1.0, 5.0\}$), three perturbation radii ($\rho \in \{0.05, 0.1, 0.5\}$), and three random seeds,
evaluating convergence speed, generalisation gap, and loss-landscape curvature.

M-SAM consistently dominates on all metrics, with its advantage widening on the more complex
ViT-B/32 architecture, confirming that geometry-aware, reparametrisation-invariant perturbations
yield measurably better optimisation than Euclidean or diagonal-scaling alternatives.

## Key findings

- **M-SAM** converges in the fewest epochs and fewest total GFLOPs, achieves the smallest
  generalisation gap, and reaches the flattest minima (lowest Hutchinson trace estimate).
- **ASAM** offers a partial improvement over SAM but is sensitive to $\rho$ and hampered by its
  diagonal scaling operator, which ignores inter-parameter correlations.
- **SAM** matches M-SAM on ResNet-18 where the loss landscape is relatively smooth, but falls
  behind on ViT-B/32.
- **SGD** consistently converges to the sharpest minima and the largest generalisation gap.
- M-SAM's advantages are **preserved under adversarial weight rescaling** ($\alpha = 5.0$), while
  SAM's performance degrades.

## Repository structure

```
configs/          Experiment YAML configs (model, optimizer, $\rho$, $\alpha$ settings)
experiments/      Training and analysis scripts
  run_resnet18_baseline.sh
  run_resnet18_reparam.sh
  run_vit_baseline.sh
  run_vit_reparam.sh
  run_flatness.py
  plot_baseline.py
  plot_convergence.py
src/              Model definitions, optimiser wrappers, metrics
  models/         ResNet-18, ViT-B/32
  optimizers/     SAM, ASAM, M-SAM
  metrics/        Hutchinson trace, loss landscape, generalisation gap
results/          Output directory (created by training scripts)
  resnet18/
  vit_b_32/
tests/            Unit tests (reparametrisation correctness, etc.)
docs/             CS229 final report (LaTeX)
```

## Infrastructure

Experiments were run on four **NVIDIA RTX 4090** GPUs (24 GB VRAM each) rented on
[Runpod](https://www.runpod.io/). Four training scripts ran in parallel—one per architecture × $\alpha$
combination—each distributing its 36 runs (4 optimisers × 3 $\rho$ values × 3 seeds) sequentially
on its assigned GPU. Mixed-precision training (`torch.cuda.amp`) was enabled for ViT-B/32.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd sam-opt-research

# 2. Install dependencies
uv sync

# 3. Start a tmux session so training survives SSH disconnects
# If you get "sessions should be nested with care", you're already inside tmux.
tmux new-session -A -s monitor_gpu
```

## Running the experiments

Each script is self-contained and writes results under `results/{model}/`.
Re-running a script skips already-completed checkpoints automatically.

### Experiment 1 — Baseline ($\alpha = 1.0$)

```bash
# ResNet-18
python experiments/run_baseline.py --config configs/resnet18_baseline_run.yaml

# ViT-B/32 (run in parallel on a separate GPU)
tmux new-session -A -s vit_baseline_train
python experiments/run_baseline.py --config configs/vit_baseline_run.yaml
```

Results: `results/resnet18/` and `results/vit_b_32/`

### Experiment 2 — Reparametrisation ($\alpha = 5.0$)

```bash
# ResNet-18
tmux new-session -A -s resnet_reparam_train
python experiments/run_reparam.py --config configs/resnet18_reparam_run.yaml

# ViT-B/32
tmux new-session -A -s vit_reparam_train
python experiments/run_reparam.py --config configs/vit_reparam_run.yaml
```

### Experiment 3 — Flatness / sharpness analysis

Requires checkpoints from Experiment 1 and 2.

```bash
# Plot loss landscapes around each checkpoint (ResNet-18)
python experiments/plot_landscape.py \
  --config configs/vit_baseline_run_flatness.yaml \
  --checkpoint results/vit_b_32/experiments/baseline/vit_b_32/checkpoints/<checkpoint_name>.pt \
  --out-dir results/vit_b_32/experiments/flatness/vit_b_32

# Plot loss landscapes around each checkpoint
python experiments/plot_landscape.py \
  --config configs/vit_baseline_run_flatness.yaml \
  --ckpt-dir results/vit_b_32/experiments/baseline/vit_b_32/checkpoints \
  --out-dir results/vit_b_32/experiments/flatness/vit_b_32
```

```bash
# Single checkpoint
python experiments/run_flatness.py \
  --config configs/resnet18_baseline_run.yaml \
  --checkpoint results/resnet18/experiments/baseline/resnet18/checkpoints/<checkpoint_name>.pt \
  --out-dir results/resnet18/experiments/flatness/resnet18

# Batch mode (all checkpoints)
# Note that this may fail due to GPU memory constraints, especially for ViT-B/32. In that case, run the above command sequentially for each checkpoint.
python experiments/run_flatness.py \
  --config configs/resnet18_baseline_run.yaml \
  --ckpt-dir results/resnet18/experiments/baseline/resnet18/checkpoints \
  --out-dir results/resnet18/experiments/flatness/resnet18
```

## GPU tips

- I wasted a lot of time with running instances on vastai, use runpod. It offers more reliable and consistent performance.
  - I found training to run much faster on runpod than vastai, even with the same GPU model (RTX 4090). Runpod's infrastructure seems better optimised for deep learning workloads, and their pricing is competitive.
  - runpod allows you to reuse network volume, this is crucial for saving checkpoints and results without needing to re-upload them every time you start a new instance.
- Consider training uses BF16 mixed precision and `torch.compile` automatically on CUDA.
- Monitor GPU: `watch -n 1 nvidia-smi`. If utilisation is low, check for CPU bottlenecks (data loading) and increase `num_workers` in the config.
- Container instances should have at least **30 GB** of disk space (ViT checkpoints alone are ~10 GB).
- Configs are under `configs/`. Each `*_run.yaml` has model-specific result paths.
- SAM variants require **two forward-backward passes per step** — expect ~2× wall-clock
  time vs SGD. M-SAM's per-step cost is comparable to SAM (the Monge metric reduces to
  a simple gradient rescaling via Sherman-Morrison), but fewer epochs are needed overall.

## Reparametrisation details

**ResNet-18:** A positive scaling factor α is injected at the output of BN1 in each BasicBlock
and compensated with 1/α at conv2. BN absorbs any pre-BN scaling, so the factor must be
applied *after* BN1 to propagate through ReLU into conv2.

**ViT-B/32:** GELU is not positively homogeneous (f(αx) ≠ αf(x)), so naive MLP weight
scaling does not preserve the block output. We replace each GELU with its first-order Taylor
approximation at zero, f(x) ≈ 0.5x (positively homogeneous), then scale linear1 by α and
linear2 by 1/α. Function preservation is verified by unit tests in `tests/`.

## Citation

```bibtex
@misc{dahra2026sam,
  title  = {Invariance and Generalization in Sharpness-Aware Minimization (SAM) Variants},
  author = {Dahra, Manav},
  year   = {2026},
  note   = {Stanford CS229 Project}
}
```

## Future work

- [ ] **Hutchinson trace on ViT checkpoints.** The current implementation becomes intractable for
  ViT-B/32 in a low-resource setting. Find an efficient approximation (e.g. low-rank or block-diagonal
  Hessian, stochastic Lanczos quadrature) that makes curvature estimation feasible for large
  transformer checkpoints.
- [ ] **Broader $\alpha$ sweep.** Experiments currently use only $\alpha \in \{1.0, 5.0\}$. Run a finer grid
  (e.g. $\alpha \in \{1.0, 2.0, 5.0, 10.0, 20.0\}$) to characterise how reparametrisation magnitude affects
  each optimiser's convergence and generalisation, and identify any threshold beyond which
  invariance breaks down.
- [ ] **Optimiser trajectory analysis.** Track and visualise the parameter-space trajectory of SGD,
  SAM, ASAM, and M-SAM throughout training (e.g. via projected 2D PCA of weight updates,
  gradient norm evolution, or sharpness along the path). This would give a clearer picture of *how*
  geometry-aware perturbations steer optimisation differently from Euclidean ones.
- [ ] **Larger-scale evaluation.** Experiments are limited to CIFAR-10; extend to ImageNet or
  domain-shift benchmarks to test whether M-SAM's geometry-aware perturbation retains its
  advantage when the label space and data distribution become more complex.
- [ ] **Adaptive $\rho$ scheduling.** All three SAM variants are sensitive to the choice of ρ; develop a
  principled, curvature-driven scheduler that tightens the perturbation ball as training progresses
  to eliminate the need for a manual grid search.
- [ ] **Broader reparametrisation families.** We studied a single α value and a linearised GELU;
  explore the full range of function-preserving transforms—including layer-norm rescaling and
  attention-head rotations in transformers—to give a more complete picture of invariance under
  realistic model manipulations.

## Acknowledgements

Thanks to **Bradley Moon** for mentorship, general guidance on experimental design, and
qualitative validation of results.

