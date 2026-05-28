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
11. ✅ `reparam.py` — implemented in `src/analysis/reparam.py` as a unified facade:
    - `apply_relu_reparam(model, alpha)` for ResNet-18 lives in `src/models/resnet18.py` (exact; ReLU homogeneity). Re-exported via the facade.
    - `apply_mlp_reparam_taylor(model, alpha) → float` for ViT-B/32 lives in `src/models/vit.py`. Scales linear1×α and linear2×(1/α) in-place; returns analytic Taylor bound `0.4255·|α−1|` from GELU quadratic term. `_apply_mlp_reparam` is kept private.
    - `measure_reparam_deviation(model, x, alpha)` for empirical output-diff confirmation.
    - `apply_reparam(model, model_name, alpha)` dispatcher in `src/analysis/reparam.py`: returns `None` for ResNet-18, Taylor bound `float` for ViT-B/32.
    - `experiments/run_reparam.py` updated to call `apply_reparam` from the facade.
    - Tests: `tests/test_reparam.py` (ResNet-18, 5 tests) and `tests/test_reparam_vit.py` (ViT, 12 tests). Run with `uv run pytest`.
12. `flatness.py` — `hutchinson_trace(model, loss_fn, loader, n_samples)`: stochastic trace estimation tr(H)/d via Rademacher vectors (z ~ ±1, estimate = z·Hz / d). `loss_landscape_1d(model, loss_fn, loader, direction, steps, range)`: perturb model along direction, record losses.
13. `metrics.py` — `compute_sem(values)`: standard error of mean. Shared helpers for logging.

## Phase 6: Configs (configs/)
14. YAML per experiment:
    - `resnet18_baseline.yaml`, `vit_baseline.yaml` — seeds, epochs, lr, momentum, wd. `rho_sweep` is
      specified **per-optimizer** under each optimizer block. SGD has no `rho_sweep` and runs once
      (rho=0.0). SAM, ASAM, M-SAM each have `rho_sweep: [0.005, 0.01, 0.02, 0.03, 0.05]`.
    - `resnet18_reparam.yaml`, `vit_reparam.yaml` — same per-optimizer `rho_sweep` structure;
      adds `alpha_values: [0.1, 0.5, 1.0, 2.0, 10.0]`. Optimizers stored as a dict (same format
      as baseline), not a flat list.

## Phase 7: Experiment Scripts (experiments/)
15. `run_baseline.py` — for each (optimizer, rho, seed): reads `rho_sweep` from the optimizer's own
    config block (defaults to `[0.0]` if absent). Trains, logs metrics to `results/baseline/`.
    Saves checkpoint per run. Writes `baseline_results.json` after every (opt, rho) group.
16. `run_reparam.py` — for each (optimizer, rho, alpha, seed): reads optimizer dict (same format as
    baseline). SGD runs once at rho=0.0; SAM/ASAM/M-SAM sweep their own rho values. Applies
    `apply_reparam` from the facade, trains from scratch, records final test accuracy and
    reparam_variance (var of mean acc across alpha, per rho). Saves `reparam_results.json`.
17. `run_flatness.py` — loads saved checkpoints, computes Hutchinson trace (n_samples=20,
    max_batch=64) and 31×31 2D loss landscape (filter-normalised, Gram-Schmidt), saves to
    `results/flatness/`.

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
- ✅ `src/analysis/reparam.py` — `apply_reparam` dispatcher + re-exports of `apply_relu_reparam`, `apply_mlp_reparam_taylor`, `measure_reparam_deviation`
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
2. **GELU reparametrization (TA-confirmed)**: Use Taylor approximation to make ViT reparam tractable:
   - **Taylor approximation**: GELU(x) ≈ x·σ(1.702x); near x=0 this is approximately linear, so the scale error is O((αx)² − (x)²) for the quadratic term. Implement as `apply_mlp_reparam_taylor(model, alpha)` and measure output deviation as a function of α.
   - Piecewise linear approximation is out of scope.

---

## Final Report: Next Steps (Post-Milestone)

### Current State (as of 2026-05-28)
- ✅ Config structure: per-optimizer `rho_sweep` in all four configs (baseline + reparam, both models)
- ✅ `run_baseline.py`: reads `rho_sweep` per-optimizer; injects `asam_eta` from optimizer block
- ✅ `run_reparam.py`: iterates optimizer dict; per-optimizer rho_sweep; SGD runs once at rho=0.0
- ✅ `run_smoke.py`: uses per-optimizer `rho_sweep`; runs baseline + flatness + reparam per model
- ✅ Experiment 1 (Baseline): 12 checkpoints on ResNet-18/CIFAR-10, seed 42 only (pre-rho_sweep run)
- ✅ Experiment 3 (Flatness): Hutchinson tr(H)/d and 3D loss landscapes for all 12 checkpoints
- ✅ Experiment 2 (Reparam): 20 runs (4 opts × 5 alphas), ResNet-18, seed 42 only (pre-rho_sweep run)
- ⏳ Full rho_sweep runs (16 baseline configs × seed, 80 reparam configs × seed) not yet executed
- ✅ Reparam code: `apply_mlp_reparam_taylor` + `src/analysis/reparam.py` facade + 17 passing tests
- ✅ Convergence metrics: `elapsed_sec` in trainer, three threshold functions in metrics.py, `plot_convergence.py`
- ⚠️  ViT-B/32 reparam experiments not yet run (Taylor bound implemented; sweep still needed)
- ❌  Multi-seed evaluation not yet done (SEM = 0 throughout; single seed 42 only)
- ❌  Pearson/Spearman sharpness–generalization correlation not computed
- ❌  Convergence rate comparison (train loss / accuracy per epoch) not plotted
- ❌  Geometry-aware generalization hypothesis test not designed

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

#### ✅ Priority 1 — ViT-B/32 GELU reparam (TA-confirmed viable)
- ~~**Step 1**: Implement `apply_mlp_reparam_taylor(model, alpha)` in `src/models/vit.py`~~ **DONE**
  - `apply_mlp_reparam_taylor` in `src/models/vit.py`: scales linear1×α / linear2×(1/α), returns `0.4255·|α−1|` analytic bound
  - `_apply_mlp_reparam` kept private; `measure_reparam_deviation` for empirical confirmation
  - `src/analysis/reparam.py` facade with `apply_reparam(model, model_name, alpha)` dispatcher
  - `experiments/run_reparam.py` updated; 17 tests passing (`uv run pytest`)
- **Step 2** *(pending)*: Run reparam sweep on ViT (same 4 optimizers × 5 alpha × seeds) using `apply_reparam`
  - Log `taylor_bound` alongside test accuracy in results JSON
  - Config: `configs/vit_reparam.yaml` (resize=224 already set)

#### ✅ Priority 2 — Convergence rate comparison (TA-suggested)
- Log train loss and train/test accuracy at *every epoch* (already in trainer, need to persist)
- Generate per-optimizer learning curves (loss vs. epoch, accuracy vs. epoch) with SEM bands
  across seeds for the best-ρ configuration of each optimizer
- Three-tier convergence metrics (applied at accuracy thresholds τ ∈ {90%, 94%, 95%}):

  **Tier 1 — Algorithmic: Epochs to Target** ✅
  - `epochs_to_threshold(acc_curve, tau)` added to `src/analysis/metrics.py`

  **Tier 2 — Computational: Total FLOPs to Target** ✅
  - `flops_to_threshold(flops_per_epoch, epoch)` added to `src/analysis/metrics.py`
  - `compute_flops_per_epoch(opt_name, macs, cfg)` in plot script; uses fvcore `FlopCountAnalysis` (557 M MACs measured for ResNet-18 on 32×32). SAM-family = 2× SGD.

  **Tier 3 — Real-World: Wall-Clock Time to Target** ✅
  - `train_one_epoch` in `src/training/trainer.py` now returns `elapsed_sec` via `time.perf_counter()`
  - `wallclock_to_threshold(time_curve, epoch)` added to `src/analysis/metrics.py`

- **Files modified/created** ✅:
  - `src/training/trainer.py` — `train_one_epoch` returns `elapsed_sec`; flows into history rows
  - `src/analysis/metrics.py` — added `epochs_to_threshold`, `flops_to_threshold`, `wallclock_to_threshold`
  - `experiments/plot_convergence.py` — **new**: three-panel grouped bar chart; gracefully warns when per-epoch history is absent (pre-existing results)

- ⚠️ **Pending**: re-run `experiments/run_baseline.py` to generate results with per-epoch history + `elapsed_sec`. Existing `baseline_results.json` pre-dates per-epoch logging.

#### Priority 3 — Multi-seed evaluation (unblocks all statistical claims)
- Re-run Experiment 1 (baseline) for best-ρ configs: SAM ρ=0.02, ASAM ρ=0.5, M-SAM ρ=0.05, SGD
  with seeds {0, 1, 2} in addition to seed 42 → 4 seeds total
- Re-run Experiment 2 (reparam) for all 4 optimizers × 5 alpha values × seeds {0, 1, 2}
- Compute SEM and 95% CIs across seeds; update `baseline_results.json` and `reparam_results.json`
- **File to update**: `experiments/run_baseline.py`, `experiments/run_reparam.py` (add `--seeds` multi-arg)

#### Priority 4 — Sharpness–generalization correlation
- Using multi-seed flatness data: compute Pearson (and Spearman) correlation between
  tr(H)/d and divergence_rate Δ = L_test − L_train across all optimizer–ρ pairs
- Add `compute_correlation(sharpness_vals, gap_vals)` in `src/analysis/metrics.py`
- Report r and p-value in the final paper; scatter plot in the notebook
- **New file**: `experiments/run_correlation.py` or extend `experiments/plot_baseline.py`

#### Priority 5 — Reparam invariance analysis
- With multi-seed data: compute per-optimizer variance (and SEM) of test accuracy across α values
- Test hypothesis: SAM most sensitive, ASAM partially invariant, M-SAM most invariant
- If single-seed trend (SAM appears most invariant) holds across seeds, this is a noteworthy
  negative result that challenges the hypothesis — explain via the SAM α=0.1 case being
  an outlier or the Monge correction not dominating at this scale

#### Priority 6 — Geometry-aware generalization hypothesis test (TA-suggested)
- **Hypothesis**: Geometry-aware optimizers (ASAM, M-SAM) find flatter minima that generalize better
  *specifically* under label noise — they should degrade less than SAM/SGD when trained on noisy labels.
- **Test design — Label-noise robustness**:
  - Re-train best-ρ configs (SAM ρ=0.02, ASAM ρ=0.5, M-SAM ρ=0.05, SGD) on CIFAR-10 with
    symmetric label noise at levels {10%, 20%, 30%} (random flip to any of 10 classes).
  - Evaluate all checkpoints on the *clean* test set.
  - Primary metric: test accuracy vs. noise level; secondary: divergence rate Δ at each noise level.
  - Prediction: ASAM and M-SAM show shallower accuracy degradation curves than SAM and SGD.
- **Implementation**:
  - `src/data/cifar10.py`: add `label_noise_frac: float = 0.0` param to `get_cifar10_loaders`;
    apply symmetric noise to training labels only (test set untouched).
  - `experiments/run_baseline.py`: add `--label-noise` CLI flag; results saved under `results/noise/`.
  - `experiments/plot_noise.py`: new script — accuracy-vs-noise-level line chart per optimizer,
    with SEM bands across seeds.
- **Expected result**: If M-SAM is geometry-aware and finds flat minima, it should show the
  smallest accuracy drop from 0% → 30% noise, relative to SAM and SGD.

#### Priority 7 — Final report writing
- Tables: update Tables 1 & 2 with multi-seed means ± SEM
- Add Table 3: reparam variance (σ²) per optimizer across α values, with SEM
- Add Table 4: sharpness–generalization regression (R², β₁, p-value)
- Add Figure: convergence curves (loss vs. epoch) with SEM bands per optimizer
- Add Figure: label-noise robustness line chart (accuracy vs. noise level per optimizer, with SEM bands)
- Narrative: address counterintuitive reparam variance; discuss geometry-aware hypothesis result



---

### Questions for TA/Mentor

#### Resolved (2026-05-27 TA discussion)
- ✅ **GELU reparam**: Taylor approximation confirmed as the approach; piecewise approximation skipped.
- ✅ **Convergence rate**: Compare convergence curves (loss/accuracy vs. epoch) across optimizers.
- ✅ **Generalization hypothesis**: Label-noise robustness selected as the test.

#### Open questions

1. **Label-noise signal strength**: We plan to sweep noise levels {10%, 20%, 30%} with symmetric
   label noise. Is 30% enough to create a detectable accuracy separation between optimizers, or
   should we go higher (e.g., 40–50%)? Are there CIFAR-10 benchmarks we should compare against?

3. **Convergence metric preference**: For comparing convergence rates, should we emphasize
   (a) epochs-to-threshold, (b) AUC of the accuracy curve, or (c) final-epoch loss slope?
   We plan to report all three but want to know which the report should foreground.

4. **Reparam variance result interpretation**: Our single-seed results show SAM has the *lowest*
   variance across α values (0.000005 vs 0.000009–0.000010 for ASAM/M-SAM). This contradicts
   the hypothesis. Is this likely a single-seed artifact, or could there be a theoretical
   reason SAM is more robust to initialization scaling on ResNet-18 with BN?

5. **Scope of final report**: Should we include a theoretical framing (e.g., sketch a PAC-Bayes
   bound connecting Hutchinson trace to generalization) or stay purely empirical?

6. **ASAM η parameter**: We use the ASAM default η=0.01. Should we sweep η or is the result
   robust to this choice? The original paper's ablation is on ImageNet; CIFAR-10 may differ.

Discussions with the mentor:
1. https://arxiv.org/pdf/2410.21265
2. https://arxiv.org/pdf/2211.17192

