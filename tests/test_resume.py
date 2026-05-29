"""Tests for checkpoint-based resume logic in run_baseline and run_reparam."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# run_baseline resume
# ---------------------------------------------------------------------------

def _make_baseline_cfg(tmp_path, model_name="resnet18"):
    runs_dir = str(tmp_path / "runs")
    experiments_dir = str(tmp_path / "experiments")
    return {
        "model": model_name,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "seeds": [0, 1],
        "data_dir": "./data",
        "runs_dir": runs_dir,
        "experiments_dir": experiments_dir,
        "num_workers": 0,
        "resize": None,
        "optimizers": {
            "sgd": {"type": "sgd"},
            "sam": {"type": "sam", "rho_sweep": [0.05]},
        },
    }


def test_baseline_skips_existing_checkpoint(tmp_path):
    """run_baseline.main() skips a (opt, rho, seed) whose checkpoint already exists."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.run_baseline import main

    cfg = _make_baseline_cfg(tmp_path)

    # Pre-create a checkpoint for sgd rho=0.0 seed=0
    ckpt_dir = tmp_path / "experiments" / "baseline" / "resnet18" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    existing_ckpt = ckpt_dir / "sgd_rho0.0_seed0.pt"
    existing_ckpt.touch()

    # Write config to a temp yaml
    import yaml
    config_path = str(tmp_path / "cfg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    run_single_calls = []

    def fake_run_single(cfg, opt_type, rho, seed, runs_dir, experiments_dir, results_root):
        run_single_calls.append((opt_type, rho, seed))
        return {
            "train_loss": 0.1, "train_acc": 0.9, "test_loss": 0.2, "test_acc": 0.8,
            "elapsed_sec": 1.0, "epoch": 1, "divergence_rate": 0.1,
            "seed": seed, "optimizer": opt_type, "rho": rho,
            "run_id": f"{opt_type}_{rho}_{seed}", "checkpoint": "",
            "history": [],
        }

    with patch("experiments.run_baseline.run_single", side_effect=fake_run_single):
        main(config_path)

    # sgd seed=0 should be skipped; sgd seed=1 and all sam runs should proceed
    assert ("sgd", 0.0, 0) not in run_single_calls, "sgd rho=0.0 seed=0 should have been skipped"
    assert ("sgd", 0.0, 1) in run_single_calls
    assert ("sam", 0.05, 0) in run_single_calls
    assert ("sam", 0.05, 1) in run_single_calls


def test_baseline_runs_all_when_no_checkpoints(tmp_path):
    """run_baseline.main() runs all (opt, rho, seed) combos when no checkpoints exist."""
    import sys, yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.run_baseline import main

    cfg = _make_baseline_cfg(tmp_path)
    config_path = str(tmp_path / "cfg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    run_single_calls = []

    def fake_run_single(cfg, opt_type, rho, seed, runs_dir, experiments_dir, results_root):
        run_single_calls.append((opt_type, rho, seed))
        return {
            "train_loss": 0.1, "train_acc": 0.9, "test_loss": 0.2, "test_acc": 0.8,
            "elapsed_sec": 1.0, "epoch": 1, "divergence_rate": 0.1,
            "seed": seed, "optimizer": opt_type, "rho": rho,
            "run_id": f"{opt_type}_{rho}_{seed}", "checkpoint": "",
            "history": [],
        }

    with patch("experiments.run_baseline.run_single", side_effect=fake_run_single):
        main(config_path)

    # 2 opts (sgd×2seeds + sam×1rho×2seeds) = 4 total calls
    assert len(run_single_calls) == 4


# ---------------------------------------------------------------------------
# run_reparam resume
# ---------------------------------------------------------------------------

def _make_reparam_cfg(tmp_path, model_name="resnet18"):
    return {
        "model": model_name,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "seeds": [0, 1],
        "data_dir": "./data",
        "runs_dir": str(tmp_path / "runs"),
        "experiments_dir": str(tmp_path / "experiments"),
        "num_workers": 0,
        "resize": None,
        "alpha_values": [1.0, 2.0],
        "optimizers": {
            "sgd": {"type": "sgd"},
        },
    }


def test_reparam_skips_done_combinations(tmp_path):
    """run_reparam.main() skips (opt, rho, alpha, seed) tuples present in existing JSON."""
    import sys, yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.run_reparam import main

    cfg = _make_reparam_cfg(tmp_path)
    config_path = str(tmp_path / "cfg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    # Pre-populate output JSON with sgd rho=0.0 alpha=1.0 seed=0 already done
    out_dir = tmp_path / "experiments" / "reparam" / "resnet18"
    out_dir.mkdir(parents=True)
    existing = [{
        "optimizer": "sgd", "rho": 0.0, "alpha": 1.0,
        "test_acc_mean": 0.8, "test_acc_sem": 0.0,
        "per_seed": [{"seed": 0, "test_acc": 0.8, "optimizer": "sgd", "alpha": 1.0, "rho": 0.0}],
    }]
    with open(out_dir / "reparam_results.json", "w") as f:
        json.dump(existing, f)

    run_single_calls = []

    def fake_run_single(cfg, opt_type, rho, alpha, seed):
        run_single_calls.append((opt_type, rho, alpha, seed))
        return {"test_acc": 0.8, "seed": seed, "optimizer": opt_type, "alpha": alpha, "rho": rho}

    with patch("experiments.run_reparam.run_single", side_effect=fake_run_single):
        main(config_path)

    assert ("sgd", 0.0, 1.0, 0) not in run_single_calls, "Already-done combo should be skipped"
    assert ("sgd", 0.0, 1.0, 1) in run_single_calls
    assert ("sgd", 0.0, 2.0, 0) in run_single_calls
    assert ("sgd", 0.0, 2.0, 1) in run_single_calls


def test_baseline_all_seeds_done_no_crash(tmp_path):
    """run_baseline.main() does not crash when all seeds for a group are already done."""
    import sys, yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.run_baseline import main

    cfg = _make_baseline_cfg(tmp_path)
    config_path = str(tmp_path / "cfg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    # Pre-create checkpoints for ALL sgd seeds
    ckpt_dir = tmp_path / "experiments" / "baseline" / "resnet18" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for seed in cfg["seeds"]:
        (ckpt_dir / f"sgd_rho0.0_seed{seed}.pt").touch()

    # Pre-populate JSON so combo_key is in _done_results
    out_dir = tmp_path / "experiments" / "baseline" / "resnet18"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [{
        "summary": {"optimizer": "sgd", "rho": 0.0, "model": "resnet18",
                    "test_acc_mean": 0.8, "test_acc_sem": 0.01},
        "per_seed": [
            {"seed": s, "test_acc": 0.8, "train_loss": 0.1, "test_loss": 0.2,
             "train_acc": 0.9, "elapsed_sec": 1.0, "epoch": 1,
             "divergence_rate": 0.1, "optimizer": "sgd", "rho": 0.0,
             "run_id": f"sgd_0.0_{s}", "checkpoint": ""}
            for s in cfg["seeds"]
        ],
    }]
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(existing, f)

    run_single_calls = []

    def fake_run_single(cfg, opt_type, rho, seed, runs_dir, experiments_dir, results_root):
        run_single_calls.append((opt_type, rho, seed))
        return {
            "train_loss": 0.1, "train_acc": 0.9, "test_loss": 0.2, "test_acc": 0.8,
            "elapsed_sec": 1.0, "epoch": 1, "divergence_rate": 0.1,
            "seed": seed, "optimizer": opt_type, "rho": rho,
            "run_id": f"{opt_type}_{rho}_{seed}", "checkpoint": "", "history": [],
        }

    with patch("experiments.run_baseline.run_single", side_effect=fake_run_single):
        main(config_path)  # must not raise

    # sgd fully done — skipped entirely; sam should still run
    assert ("sgd", 0.0, 0) not in run_single_calls
    assert ("sgd", 0.0, 1) not in run_single_calls
    assert ("sam", 0.05, 0) in run_single_calls
    assert ("sam", 0.05, 1) in run_single_calls


def test_reparam_all_seeds_done_no_crash(tmp_path):
    """run_reparam.main() does not crash when all seeds for every (opt, rho, alpha) are done."""
    import sys, yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.run_reparam import main

    cfg = _make_reparam_cfg(tmp_path)
    config_path = str(tmp_path / "cfg.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    out_dir = tmp_path / "experiments" / "reparam" / "resnet18"
    out_dir.mkdir(parents=True)
    existing = [
        {
            "optimizer": "sgd", "rho": 0.0, "alpha": alpha,
            "test_acc_mean": 0.8, "test_acc_sem": 0.0,
            "per_seed": [
                {"seed": s, "test_acc": 0.8, "optimizer": "sgd", "alpha": alpha, "rho": 0.0}
                for s in cfg["seeds"]
            ],
        }
        for alpha in cfg["alpha_values"]
    ]
    with open(out_dir / "reparam_results.json", "w") as f:
        json.dump(existing, f)

    run_single_calls = []

    def fake_run_single(cfg, opt_type, rho, alpha, seed):
        run_single_calls.append((opt_type, rho, alpha, seed))
        return {"test_acc": 0.8, "seed": seed, "optimizer": opt_type, "alpha": alpha, "rho": rho}

    with patch("experiments.run_reparam.run_single", side_effect=fake_run_single):
        main(config_path)  # must not raise

    assert run_single_calls == [], "All combos done — nothing should run"
