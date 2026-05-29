# Bug Tracker

Bugs found during codebase review (2026-05-27). Each entry records severity, location, description, status, and the fix applied.

---

## B-01 · `main.py` — `ImportError` on flatness experiment

**Severity:** Critical (runtime crash)  
**File:** `main.py`  
**Status:** Fixed

### Description
`main.py` dispatches the flatness experiment with:
```python
from experiments.run_flatness import main as run
```
but `run_flatness.py` only defines `main_single` and `main_batch` — no `main` symbol exists. Any call to `python main.py flatness ...` crashes immediately with `ImportError`.

### Fix
Added a `main(config_path, checkpoint, seed)` dispatcher to `run_flatness.py` that routes to `main_single` when a checkpoint path is provided, and to `main_batch` (using the checkpoints directory written by `run_baseline.py`) otherwise.

---

## B-02 · `trainer.py` — SAM train metrics logged at perturbed weights

**Severity:** Critical (corrupted metrics)  
**File:** `src/training/trainer.py`  
**Status:** Fixed

### Description
In the SAM training branch, the second forward pass overwrote `outputs` and `loss` with values computed at the perturbed point θ+ε:
```python
outputs = model(inputs)          # first pass  — original θ
...
outputs = model(inputs)          # second pass — perturbed θ+ε  ← overwrites
loss    = loss_fn(outputs, targets)
...
total_loss    += loss.item() * targets.size(0)   # loss at θ+ε, not θ
total_correct += outputs.argmax(dim=1).eq(...)   # acc  at θ+ε, not θ
```
Logged `train_loss` and `train_acc` were therefore measured at the perturbed point, not the actual model parameters.

### Fix
Renamed second-pass variables to `outputs_perturbed` / `loss_perturbed` so the metric accumulation below continues to use the first-pass values (at original θ).

---

## B-03 · `trainer.py` — BatchNorm running stats corrupted during SAM second pass

**Severity:** High (silent inference-time error)  
**File:** `src/training/trainer.py`  
**Status:** Fixed

### Description
PyTorch BatchNorm layers in `model.train()` mode update `running_mean` and `running_var` via EMA on every forward pass. The second SAM forward pass is performed at perturbed weights θ+ε, so the activation distribution seen by BN differs from the true model distribution. Those corrupt statistics are baked into `running_mean`/`running_var` and are used at inference time, silently degrading evaluation accuracy.

Note: `model.train()` **enables** running-stat updates; `model.eval()` **freezes** them.

### Fix
Momentum is set to 0 for the second forward pass, so the running stats are not updated with the perturbed activations:
```python
def _bn_disable_running_stats(model: nn.Module) -> None:
    """Suppress EMA updates to BN running_mean/running_var during the SAM second
    forward pass by setting momentum=0.  The model stays in train mode so batch
    statistics are still used for normalisation — only the running-stat write-back
    is skipped.  Call _bn_restore_running_stats() afterwards.
    """
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m._bak_momentum = m.momentum
            m.momentum = 0.0
```
This preserves the original running stats computed at θ, ensuring correct behaviour at inference time.

---

## B-04 · `run_flatness.py` — `_plot_landscape_comparison` missing matrix transpose

**Severity:** Medium (incorrect visualisation in batch mode)  
**File:** `experiments/run_flatness.py`  
**Status:** Fixed

### Description
Plotly `Surface` interprets `z[i][j]` as the value at `(x[j], y[i])`, so the loss matrix must be transposed before passing it in. `plot_loss_landscape_2d` (single-checkpoint path) correctly applies `.T`; `_plot_landscape_comparison` (batch mode) did not, causing the α and β axes to be swapped in every subplot. `_plot_landscape_best` already applied `.T` correctly, making the two batch-mode plots inconsistent with each other.

### Fix
Added `.T` to the `np.clip(...)` call inside `_plot_landscape_comparison`.

---

## B-05 · `landscape.py` — floating-point drift in `loss_landscape_1d`

**Severity:** Minor (subtle numerical error)  
**File:** `src/analysis/landscape.py`  
**Status:** Fixed

### Description
The 1D landscape scan restored model weights by adding `−alpha × direction` after each evaluation:
```python
_perturb_model(model_copy, dir_device,  float(alpha))   # perturb
...
_perturb_model(model_copy, dir_device, -float(alpha))   # restore
```
Because floating-point addition is not perfectly reversible (`p + α·d − α·d ≠ p` exactly), rounding errors accumulate across 51 steps, particularly for large `alpha` values. The 2D function already used the correct pattern (save originals, restore by copy).

### Fix
`loss_landscape_1d` now saves `originals = [p.data.clone() for p in model_copy.parameters()]` before the loop and restores each parameter with `p.data.copy_(orig + d * alpha)`, matching the 2D implementation.

---

## B-06 · `landscape.py` — Gram-Schmidt orthogonalisation discards filter normalisation of `dir2`

**Severity:** Minor (asymmetric 2D landscape grid)  
**File:** `src/analysis/landscape.py`  
**Status:** Fixed

### Description
`loss_landscape_2d` samples two filter-normalised random directions and orthogonalises them via Gram-Schmidt. After the subtraction step:
```python
dir2 = [d2 - coeff * d1 for d2, d1 in zip(dir2, dir1)]
```
the per-filter norms of `dir2` are no longer matched to the model's weight norms, breaking the filter-normalisation invariant established by `_filter_normalized_direction`. The two grid axes therefore have different effective scales, making the landscape surface asymmetric and distances along α incomparable to distances along β.

### Fix
After orthogonalisation, `dir2` is re-normalised filter-by-filter (same logic as `_filter_normalized_direction`) so both directions have comparable per-filter scales.

---

## B-07 · `run_flatness.py` — `torch.load` without `weights_only=True`

**Severity:** Minor (security / forward-compatibility)  
**File:** `experiments/run_flatness.py`  
**Status:** Fixed

### Description
```python
state = torch.load(path, map_location=device)
```
Without `weights_only=True`, PyTorch deserialises the file using `pickle`, which can execute arbitrary code in a malicious checkpoint. PyTorch ≥ 2.0 emits a `FutureWarning`; PyTorch ≥ 2.6 raises an error by default.

### Fix
Changed to `torch.load(path, map_location=device, weights_only=True)`.
