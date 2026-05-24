# Plan: SAM Optimization Research Implementation

## Overview
Empirical benchmarking of SGD, SAM, ASAM, and M-SAM on ResNet-18 and ViT-B/16 with CIFAR-10.
Three experiments: (1) baseline comparison, (2) reparametrization invariance, (3) flatness analysis.
Stack: Python 3.13, PyTorch, torchvision, timm, YAML configs, Jupyter for plots.

---

## Phase 0: Project Setup
1. Update `pyproject.toml` — add torch, torchvision, timm, numpy, matplotlib, seaborn, pyyaml, tqdm
2. Create directory tree:
   - `src/optimizers/`, `src/models/`, `src/data/`, `src/training/`, `src/analysis/`
   - `configs/`, `experiments/`, `notebooks/`, `results/` (git-ignored)
3. Update `main.py` as CLI dispatch entry point (calls experiment scripts by name)

## Phase 1: Optimizers (src/optimizers/)
4. `sam.py` — SAM: two-step API (`first_step` / `second_step`). Step 1: ε = ρ·∇ℓ/‖∇ℓ‖; Step 2: apply base optimizer at θ+ε, restore θ.
5. `asam.py` — ASAM: adaptive perturbation ε = ρ·T_w²∇ℓ/‖T_w∇ℓ‖ where T_w = diag(|w|+η). Same two-step API.
6. `msam.py` — M-SAM: ε = δ_SAM / (1 + ‖∇ℓ‖²₂). Thin wrapper over SAM's first_step with rescaling. Same API.

## Phase 2: Data (src/data/)
7. `cifar10.py` — returns (train_loader, test_loader). Standard augmentation (RandomCrop 32 pad=4, RandomHorizontalFlip, Normalize). Seed-controlled worker init.

## Phase 3: Models (src/models/)
8. `resnet18.py` — wraps `torchvision.models.resnet18(num_classes=10)`.
9. `vit.py` — wraps `torchvision.models.vit_b_32` (patch size 32). Resize CIFAR-10 inputs to 224×224 in the data pipeline so (224/32)²=49 patches are produced. Expose `get_model(name, num_classes)` factory.

## Phase 4: Trainer (src/training/)
10. `trainer.py` — `train_one_epoch(model, optimizer, loader, loss_fn)` and `evaluate(model, loader, loss_fn)`. Returns dict: {train_acc, test_acc, train_nll, test_nll} per epoch. Handles SAM two-step pattern via duck typing.

## Phase 5: Analysis Modules (src/analysis/)
11. `reparam.py` — two functions:
    - `apply_relu_reparam(model, alpha)` for ResNet-18: for each consecutive Conv/Linear pair separated by ReLU, scale layer_i weights (and bias) by α, scale layer_{i+1} input weights by 1/α. Exact function-preservation; verified by `assert f(x) ≈ reparam_f(x)`.
    - `apply_mlp_reparam(model, alpha)` for ViT-B/32: targets each Transformer MLP block's Linear→GELU→Linear pair. Scale first Linear's weight and bias by α; scale second Linear's input weight by 1/α. GELU is not scale-homogeneous so this is an *approximate* reparametrization. Log the pre-training output deviation (rather than asserting equality) to characterize the approximation quality.
12. `flatness.py` — `hutchinson_trace(model, loss_fn, loader, n_samples)`: stochastic trace estimation tr(H)/d via Rademacher vectors (z ~ ±1, estimate = z·Hz / d). `loss_landscape_1d(model, loss_fn, loader, direction, steps, range)`: perturb model along direction, record losses.
13. `metrics.py` — `compute_sem(values)`: standard error of mean. Shared helpers for logging.

## Phase 6: Configs (configs/)
14. YAML per experiment:
    - `resnet18_baseline.yaml`, `vit_baseline.yaml` — seeds, epochs, lr, momentum, wd, rho_sweep
    - `resnet18_reparam.yaml`, `vit_reparam.yaml` — alpha_values: [0.1, 0.5, 1, 2, 10]
    - Single rho for reparam test; multiple seeds

## Phase 7: Experiment Scripts (experiments/)
15. `run_baseline.py` — for each (model, optimizer, rho, seed): train, log metrics to results/baseline/. Saves checkpoint per run.
16. `run_reparam.py` — for each (model, optimizer, alpha, seed): call `apply_relu_reparam` (ResNet-18) or `apply_mlp_reparam` (ViT-B/32), train from scratch, record final test acc. Saves to results/reparam/.
17. `run_flatness.py` — loads saved checkpoints, computes Hutchinson trace and 1D landscape, saves to results/flatness/.

## Phase 8: Notebook (notebooks/)
18. `analysis.ipynb` — load CSVs from results/, generate all figures: (a) accuracy/NLL curves with SEM bands, (b) divergence rate vs ρ, (c) reparam variance bar chart, (d) loss landscape plots, (e) sharpness bar chart.

---

## Relevant Files
- `pyproject.toml` — add all dependencies
- `main.py` — CLI entry: `python main.py baseline --config configs/resnet18_baseline.yaml`
- `src/optimizers/sam.py` — SAM first_step/second_step
- `src/optimizers/asam.py` — ASAM, adaptive T_w norm
- `src/optimizers/msam.py` — M-SAM = SAM / (1 + ||g||²)
- `src/models/vit.py` — ViT-B/32 (`torchvision.models.vit_b_32`), 224×224 resize in data pipeline
- `src/analysis/reparam.py` — `apply_relu_reparam` (ResNet-18, exact) + `apply_mlp_reparam` (ViT-B/32 MLP blocks, approximate)
- `src/analysis/flatness.py` — Hutchinson estimator with autograd

## Verification
1. Unit test each optimizer: confirm perturbation formula matches paper equations
2. Sanity check reparam: `assert model(x) ≈ reparam_model(x)` for random batch
3. Run 1 epoch of each optimizer on ResNet-18 with CIFAR-10 and confirm loss decreases
4. Confirm Hutchinson estimator converges for a small known quadratic
5. Run full baseline on ResNet-18 (fast architecture) to confirm results look sensible before running ViT

## Decisions / Scope Boundaries
- Dataset: CIFAR-10 only (no CIFAR-100 or ImageNet)
- Architectures: ResNet-18 (from-scratch) and ViT-B/32 (`torchvision.models.vit_b_32`, inputs resized to 224×224)
- Stretch goal (metric ablation) is explicitly out of scope for initial implementation
- No early stopping — fixed epoch count per config
- Results stored as CSV + pickle, not a database
- M-SAM implemented from paper eq. 9 (no official code available)
- Device selection: `cuda` → `mps` → `cpu` (priority order, auto-detected at runtime; no hard-coded device IDs)
- ViT-B/32: **fine-tune from pre-trained ImageNet weights** — none of the three papers train a ViT from scratch on CIFAR-10; M-SAM itself uses a pre-trained ResNet-18; training ViT from scratch on CIFAR-10 is impractical and not supported by prior work
- Reparam for ResNet-18: exact (ReLU homogeneity). Reparam for ViT-B/32: approximate (GELU not homogeneous); deviation is logged, not asserted.

## Further Considerations
1. **Approximate reparam on ViT**: The GELU deviation grows with |α-1|. For extreme values (α=0.1, α=10) the network function will change significantly, which may confound the invariance experiment. Report the output deviation alongside reparam variance results to contextualize findings.

---

## Final Report: Next Steps (Post-Milestone)

### Current State (as of 2026-05-24)
- ✅ Experiment 1 (Baseline): 12 checkpoints on ResNet-18/CIFAR-10, seed 42 only
- ✅ Experiment 3 (Flatness): Hutchinson tr(H)/d and 3D loss landscapes for all 12 checkpoints
- ✅ Experiment 2 (Reparam): 20 runs (4 optimizers × 5 alpha values), ResNet-18, seed 42 only
- ⚠️  ViT-B/32 experiments paused (GELU approximation deviation ~0.06 per unit at α=2)
- ❌  Multi-seed evaluation not yet done (SEM = 0 throughout; single seed 42 only)
- ❌  Pearson/Spearman sharpness–generalization correlation not computed

### Surprising Observation: Reparam Variance Results
Single-seed variance of test accuracy across α ∈ {0.1, 0.5, 1.0, 2.0, 10.0}:
- SGD:  0.000010
- SAM:  0.000005  ← **lowest** (counterintuitive — expected most sensitive)
- ASAM: 0.000009
- MSAM: 0.000009
SAM appears more invariant than ASAM/MSAM to initialization scaling in these single-seed runs.
This likely reflects noise from single-seed evaluation and must be revisited with multi-seed data.

---

### Action Plan

#### Priority 1 — Multi-seed evaluation (unblocks all statistical claims)
- Re-run Experiment 1 (baseline) for best-ρ configs: SAM ρ=0.02, ASAM ρ=0.5, M-SAM ρ=0.05, SGD
  with seeds {0, 1, 2} in addition to seed 42 → 4 seeds total
- Re-run Experiment 2 (reparam) for all 4 optimizers × 5 alpha values × seeds {0, 1, 2}
- Compute SEM and 95% CIs across seeds; update `baseline_results.json` and `reparam_results.json`
- **File to update**: `experiments/run_baseline.py`, `experiments/run_reparam.py` (add `--seeds` multi-arg)

#### Priority 2 — Sharpness–generalization correlation
- Using multi-seed flatness data: compute Pearson (and Spearman) correlation between
  tr(H)/d and divergence_rate Δ = L_test − L_train across all optimizer–ρ pairs
- Add `compute_correlation(sharpness_vals, gap_vals)` in `src/analysis/metrics.py`
- Report r and p-value in the final paper; scatter plot in the notebook
- **New file**: `experiments/run_correlation.py` or extend `experiments/plot_baseline.py`

#### Priority 3 — Reparam invariance analysis
- With multi-seed data: compute per-optimizer variance (and SEM) of test accuracy across α values
- Test hypothesis: SAM most sensitive, ASAM partially invariant, M-SAM most invariant
- If single-seed trend (SAM appears most invariant) holds across seeds, this is a noteworthy
  negative result that challenges the hypothesis — explain via the SAM α=0.1 case being
  an outlier or the Monge correction not dominating at this scale

#### Priority 4 — ViT-B/32 (stretch goal)
- Option A: Use approximate reparam and report output deviation alongside results; acknowledge limitation
- Option B: Replace GELU with ReLU in ViT MLP blocks (fine-tune only) to make reparam exact
- Option C: Drop ViT from reparam experiment; keep ViT only for baseline accuracy comparison
- Recommended: Option A for reparam, Option C if compute is tight

#### Priority 5 — Final report writing
- Tables: update Tables 1 & 2 with multi-seed means ± SEM
- Add Table 3: reparam variance (σ²) per optimizer across α values, with SEM
- Add Table 4 / Figure: Pearson correlation matrix (sharpness vs. divergence rate)
- Narrative: address the counterintuitive reparam variance finding

---

### Questions for TA/Mentor

1. **Reparam variance result interpretation**: Our single-seed results show SAM has the *lowest*
   variance across α values (0.000005 vs 0.000009–0.000010 for ASAM/M-SAM). This contradicts
   the hypothesis. Is this likely a single-seed artifact, or could there be a theoretical
   reason SAM is more robust to initialization scaling on ResNet-18 with BN?

2. **Correlation metric choice**: For sharpness–generalization analysis, should we use Pearson
   correlation (assumes linearity) or Spearman rank correlation? The relationship is expected
   to be monotone but possibly non-linear given the ρ sweep.

3. **Number of seeds**: Is 4 seeds (0, 1, 2, 42) sufficient for significance claims (t-test /
   95% CI) given our ~0.5% accuracy differences, or do we need more runs?

4. **ViT reparametrization**: Given GELU is not 1-homogeneous, is there an accepted formulation
   in the literature for approximate invariance experiments on transformers? Alternatively,
   is comparing ViT baseline performance (without reparam) still valuable for the paper?

5. **Scope of final report**: Should we aim for a theoretical framing (e.g., sketch a PAC-Bayes
   bound connecting Hutchinson trace to generalization) or stay purely empirical?

6. **Loss landscape normalization**: We use filter normalization (Li et al., 2018) for the 3D
   plots. For the 1D sharpness plots used in the correlation analysis, should we use the same
   normalization or raw ℓ₂ directions? Does the choice affect the tr(H)/d estimates?

7. **ASAM η parameter**: We use the ASAM default η=0.01. Should we sweep η or is the result
   robust to this choice? The original paper's ablation is on ImageNet; CIFAR-10 may differ.

