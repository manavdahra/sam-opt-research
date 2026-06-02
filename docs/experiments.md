# Experiments Notes

## Batch Size Differences: ResNet-18 vs ViT-B/32

ResNet-18 uses `batch_size: 256`; ViT-B/32 uses `batch_size: 64` due to GPU memory constraints
(224×224 inputs + two forward passes for SAM-family optimizers exhaust Apple M4 memory at 256).

**This does not invalidate cross-architecture comparisons** because:

1. **Primary comparisons are within-architecture.** SGD vs SAM vs ASAM vs M-SAM are each
   evaluated at the same batch size within their architecture (256 for ResNet-18, 64 for ViT).
   Optimizer rankings within each architecture remain fair.

2. **Cross-architecture comparison is qualitative, not quantitative.** The research question is
   whether reparametrization invariance and sharpness patterns *generalise* from ResNet-18 to
   ViT — e.g. does M-SAM still achieve the lowest tr(H)/d, does ASAM still show a lower
   divergence rate? Absolute accuracy numbers are not compared across architectures.

**Learning rate note:** The linear scaling rule suggests lr ∝ batch_size. ViT already uses
`lr: 0.01` vs ResNet-18's `lr: 0.1` (10× lower vs 4× batch reduction), which is appropriate
since ViT is fine-tuned from pretrained ImageNet weights and warrants a lower LR regardless.

**For the final report:** include a sentence such as:
> "ViT-B/32 experiments use batch size 64 due to GPU memory constraints; all four optimizers
> are evaluated under identical conditions within each architecture."

---

## Experiment Walkthrough (2-GPU Setup)

Assign GPU 0 to ResNet-18 and GPU 1 to ViT-B/32 so all four phases run in parallel.

### Phase 0 — Sanity Checks (both GPUs idle, ~1 min)

Run the full test suite before starting any training:

```bash
uv run pytest tests/ -v
```

Covers: ResNet-18 reparam correctness, ViT reparam + Taylor bound, metrics tracking
(including rho/alpha key presence in run_reparam results).
All tests must pass before proceeding.

---

### Phase 1 + 2 — Baseline Training (run in parallel)

The `rho_sweep` is specified **per-optimizer** inside each optimizer block. SGD has no
`rho_sweep` and runs once (rho=0.0). SAM, ASAM, and M-SAM each have their own
`rho_sweep: [0.005, 0.01, 0.02, 0.03, 0.05]` → **16 runs per model per seed** (1 + 5 + 5 + 5).

**GPU 0 — ResNet-18** (≈ 3–4 h at 200 epochs, batch 256)
```bash
uv run python main.py baseline \
    --config configs/resnet18_baseline.yaml
```

**GPU 1 — ViT-B/32** (≈ 2–3 h at 50 epochs, batch 64, pretrained)
```bash
uv run python main.py baseline \
    --config configs/vit_baseline.yaml
```

Outputs per run:
- `results/runs/<timestamp>-<model>-<opt>-rho<ρ>-seed<seed>/` — checkpoint + metrics
- `results/experiments/baseline/<model>/checkpoints/` — flat checkpoint dir for flatness phase
- `results/experiments/baseline/<model>/baseline_results.json` — aggregated mean ± SEM

---

### Phase 3 + 4 — Reparametrisation Invariance (run in parallel, after Phase 0)

These are independent of Phases 1/2 and can start immediately on the same GPUs.
If Phases 1/2 are still running, queue these to start when the GPUs free up.

**GPU 0 — ResNet-18 reparam** (same wall-clock cost as Phase 1, times len(alpha_values) × len(rho_sweep))
```bash
uv run python main.py reparam \
    --config configs/resnet18_reparam.yaml
```

**GPU 1 — ViT-B/32 reparam**
```bash
uv run python main.py reparam \
    --config configs/vit_reparam.yaml
```

For each optimizer × ρ ∈ `rho_sweep` × α ∈ {0.1, 0.5, 1.0, 2.0, 10.0}: apply the
function-preserving weight rescaling to initial weights, train from scratch, record final test
accuracy.  SGD has no `rho_sweep` and runs once (rho=0.0); SAM, ASAM, and M-SAM each sweep
their own rho values → **80 training runs per model per seed** (5 + 25 + 25 + 25).
Reparam configs use the same dict-based optimizer format as the baseline configs.
Key metric: **variance of mean test accuracy across α** — lower = more invariant.

Output: `results/experiments/reparam/<model>/reparam_results.json`

---

### Phase 5 — Flatness / Sharpness Analysis (after Phase 1/2)

Requires the checkpoints produced by the baseline phase.

**GPU 0 — ResNet-18**
```bash
uv run python experiments/run_flatness.py \
    --config configs/resnet18_baseline.yaml \
    --ckpt-dir results/experiments/baseline/resnet18/checkpoints \
    --out-dir  results/experiments/flatness/resnet18
```

**GPU 1 — ViT-B/32**
```bash
uv run python experiments/run_flatness.py \
    --config configs/vit_baseline.yaml \
    --ckpt-dir results/experiments/baseline/vit_b_32/checkpoints \
    --out-dir  results/experiments/flatness/vit_b_32
```

Per checkpoint: Hutchinson tr(H)/d (20 Rademacher samples, 64-image batch, **training data**)
and a 31×31 2D loss landscape (filter-normalised, Gram-Schmidt orthogonalised, **training data**).

Outputs: `sharpness_all.json`, `landscape_all.json`, `sharpness_bars.html`,
`sharpness_vs_rho.html`, `landscape_all.html`, `landscape_best.html`

---

### Phase 6 — Plots & Analysis (after all phases complete, CPU only)

```bash
uv run python experiments/plot_baseline.py
uv run python experiments/plot_convergence.py
```

Key things to inspect:

| Plot | Question |
|---|---|
| Accuracy vs ρ | Does SAM/M-SAM beat SGD? Optimal ρ? |
| Generalization gap | Do SAM variants reduce `test_loss − train_loss`? |
| Sharpness vs ρ | Does tr(H)/d decrease with ρ, or is there a sweet spot? |
| Reparam variance | SGD > SAM > M-SAM ≈ ASAM ≈ 0 is the expected ordering |
| Loss landscapes | Are SAM minima visually broader than SGD? |

---

### Dependency & Parallelism Overview

```
Phase 0 (tests) ────────────────────────────────────┐
                                                     │
GPU 0:  Phase 1 (ResNet-18 baseline) ──→ Phase 5 (ResNet-18 flatness)
        Phase 3 (ResNet-18 reparam)   ──┘ (queue after Phase 1)

GPU 1:  Phase 2 (ViT baseline)       ──→ Phase 5 (ViT flatness)
        Phase 4 (ViT reparam)         ──┘ (queue after Phase 2)

CPU:    Phase 6 (plots) ← waits for all JSON results
```

Phases 1 & 3 share GPU 0 sequentially (or interleave if memory allows).
Phases 2 & 4 share GPU 1 sequentially.


## Instances:

C.38440613 - resnet18 reparam
C.38447562 - vit reparam
C.38448435 - resnet18 baseline
C.38455116 - vit baseline

Generate loss landscape plots for vit baseline and reparam models
```python
uv run experiments/run_flatness.py --config configs/vit_baseline_run.yaml --ckpt-dir results/vit_b_32/experiments/baseline/vit_b_32/checkpoints/
```

