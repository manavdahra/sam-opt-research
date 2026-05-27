"""One-time migration from the old flat layout to the new runs/ + experiments/ structure.

Old layout:
    results/baseline/baseline/resnet18/
        baseline_results.json  *.html  checkpoints/*.pt
    results/baseline/flatness/resnet18/
        *.json  *.html
    results/reparam/resnet18/
        reparam_results.json

New layout:
    results/
    ├── index.json
    ├── runs/
    │   └── <run-id>/
    │       ├── config.yaml
    │       ├── metrics.json   (stub — no per-epoch data for pre-migration runs)
    │       └── checkpoint.pt
    └── experiments/
        ├── baseline/resnet18/
        │   ├── baseline_results.json
        │   ├── checkpoints/   (convenience symlinks for run_flatness.py batch mode)
        │   └── *.html
        ├── flatness/resnet18/
        │   └── *.json  *.html
        └── reparam/resnet18/
            └── reparam_results.json

Usage:
    python experiments/migrate_results.py          # dry run — shows what would happen
    python experiments/migrate_results.py --apply  # actually moves files
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = os.path.join(_ROOT, "results")

# ── Source paths (old layout) ─────────────────────────────────────────────────
OLD_BASELINE_DIR  = os.path.join(RESULTS, "baseline", "baseline", "resnet18")
OLD_FLATNESS_DIR  = os.path.join(RESULTS, "baseline", "flatness", "resnet18")
OLD_REPARAM_DIR   = os.path.join(RESULTS, "reparam", "resnet18")

# ── Destination paths (new layout) ───────────────────────────────────────────
NEW_RUNS_DIR          = os.path.join(RESULTS, "runs")
NEW_BASELINE_DIR      = os.path.join(RESULTS, "experiments", "baseline", "resnet18")
NEW_FLATNESS_DIR      = os.path.join(RESULTS, "experiments", "flatness", "resnet18")
NEW_REPARAM_DIR       = os.path.join(RESULTS, "experiments", "reparam", "resnet18")
NEW_CKPT_DIR          = os.path.join(NEW_BASELINE_DIR, "checkpoints")
INDEX_PATH            = os.path.join(RESULTS, "index.json")


def _move(src: str, dst: str, apply: bool) -> None:
    if not os.path.exists(src):
        print(f"  SKIP (missing): {src}")
        return
    print(f"  mv {src}")
    print(f"     → {dst}")
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def _move_dir_contents(src_dir: str, dst_dir: str, apply: bool) -> None:
    """Move every file in src_dir to dst_dir (non-recursive)."""
    if not os.path.isdir(src_dir):
        print(f"  SKIP (dir missing): {src_dir}")
        return
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            _move(src, dst, apply)


def _build_index(apply: bool) -> None:
    """Populate index.json from baseline_results.json (best-effort)."""
    results_json = os.path.join(NEW_BASELINE_DIR, "baseline_results.json")
    if not os.path.exists(results_json):
        print("  SKIP index.json — baseline_results.json not found yet")
        return

    with open(results_json) as f:
        data = json.load(f)

    index: dict = {"runs": {}}
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            index = json.load(f)

    for entry in data:
        for seed_row in entry.get("per_seed", []):
            opt  = seed_row.get("optimizer", "unknown")
            rho  = seed_row.get("rho", 0.0)
            seed = seed_row.get("seed", 0)
            acc  = seed_row.get("test_acc", 0.0)
            ts   = "migrated"

            # The old checkpoint is already in the new experiments/checkpoints dir.
            ckpt = os.path.join(NEW_CKPT_DIR, f"{opt}_rho{rho}_seed{seed}.pt")
            run_id = f"migrated-resnet18-{opt}-rho{rho}-seed{seed}"

            # Create a stub run directory containing just the checkpoint.
            run_dir = os.path.join(NEW_RUNS_DIR, run_id)
            print(f"  stub run dir: {run_dir}")
            if apply:
                os.makedirs(run_dir, exist_ok=True)
                # Copy (not move) — the canonical copy stays in experiments/checkpoints.
                if os.path.exists(ckpt) and not os.path.exists(os.path.join(run_dir, "checkpoint.pt")):
                    shutil.copy2(ckpt, os.path.join(run_dir, "checkpoint.pt"))
                stub_metrics = seed_row.get("history", [])
                with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                    json.dump(stub_metrics, f, indent=2)

            index["runs"][run_id] = {
                "model": "resnet18",
                "optimizer": opt,
                "rho": rho,
                "seed": seed,
                "test_acc": acc,
                "timestamp": ts,
                "experiment": "baseline",
                "run_dir": run_dir,
                "checkpoint": ckpt,
            }

    print(f"  index.json → {INDEX_PATH}  ({len(index['runs'])} entries)")
    if apply:
        tmp = INDEX_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, INDEX_PATH)


def _remove_if_empty(path: str, apply: bool) -> None:
    if not os.path.exists(path):
        return
    try:
        if not os.listdir(path):
            print(f"  rmdir (empty): {path}")
            if apply:
                os.rmdir(path)
    except NotADirectoryError:
        pass


def main(apply: bool) -> None:
    mode = "APPLY" if apply else "DRY RUN"
    print(f"\n{'='*60}")
    print(f"  migrate_results.py  [{mode}]")
    print(f"{'='*60}\n")

    # 1. Move baseline HTML + JSON (excluding checkpoints subdir)
    print("── Baseline artefacts ─────────────────────────────────────")
    for fname in os.listdir(OLD_BASELINE_DIR) if os.path.isdir(OLD_BASELINE_DIR) else []:
        if fname == "checkpoints":
            continue
        _move(
            os.path.join(OLD_BASELINE_DIR, fname),
            os.path.join(NEW_BASELINE_DIR, fname),
            apply,
        )

    # 2. Move checkpoints
    print("\n── Checkpoints ─────────────────────────────────────────────")
    old_ckpt_dir = os.path.join(OLD_BASELINE_DIR, "checkpoints")
    _move_dir_contents(old_ckpt_dir, NEW_CKPT_DIR, apply)

    # 3. Move flatness artefacts
    print("\n── Flatness artefacts ──────────────────────────────────────")
    _move_dir_contents(OLD_FLATNESS_DIR, NEW_FLATNESS_DIR, apply)

    # 4. Move reparam artefacts
    print("\n── Reparam artefacts ───────────────────────────────────────")
    _move_dir_contents(OLD_REPARAM_DIR, NEW_REPARAM_DIR, apply)

    # 5. Build index.json and stub run directories
    print("\n── index.json + stub run dirs ──────────────────────────────")
    _build_index(apply)

    # 6. Remove now-empty old directories
    print("\n── Cleanup empty directories ───────────────────────────────")
    for d in [
        old_ckpt_dir,
        OLD_BASELINE_DIR,
        os.path.join(RESULTS, "baseline", "baseline"),
        OLD_FLATNESS_DIR,
        os.path.join(RESULTS, "baseline", "flatness"),
        os.path.join(RESULTS, "baseline"),
        OLD_REPARAM_DIR,
        os.path.join(RESULTS, "reparam"),
    ]:
        _remove_if_empty(d, apply)

    print(f"\n{'='*60}")
    print(f"  Done.  {'Changes applied.' if apply else 'Re-run with --apply to execute.'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate results to new directory layout.")
    parser.add_argument("--apply", action="store_true", help="Actually move files (default: dry run)")
    args = parser.parse_args()
    main(apply=args.apply)
