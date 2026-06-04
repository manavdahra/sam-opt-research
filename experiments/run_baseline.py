import argparse
import os
import sys
import yaml
import datetime
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from src.training.trainer import train
from src.analysis.metrics import aggregate_seeds, divergence_rate
from experiments.utils import (
    get_device, set_seed, build_model, build_optimizer, save_results,
    make_run_id, write_run_dir, update_index, build_data_loaders,
)


def run_single(
    cfg: dict,
    opt_type: str,
    rho: float,
    seed: int,
    runs_dir: str,
    experiments_dir: str,
    results_root: str,
) -> dict:
    """Run a single training session for the given optimizer type, rho, and seed, then save results and return a summary dict."""
    device = get_device()
    set_seed(seed)

    train_loader, test_loader = build_data_loaders(cfg)

    model = build_model(cfg, device)
    optimizer, scheduler = build_optimizer(opt_type, model.parameters(), cfg, rho)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model and collect training history
    history = train(
        model,
        optimizer,
        train_loader,
        test_loader,
        loss_fn,
        device,
        epochs=cfg["epochs"],
        scheduler=scheduler,
        verbose=True,
    )

    # Save model checkpoint, training history, and summary metrics for this run
    run_id = make_run_id(cfg["model"], opt_type, rho, seed)
    run_dir = write_run_dir(runs_dir, run_id, cfg, history, model.state_dict())
    print(f"Run artefacts saved at: {run_dir}")

    # Also save a checkpoint in the experiments directory for easy access during landscape analysis and plotting
    ckpt_dir = os.path.join(experiments_dir, "baseline", cfg["model"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{opt_type}_rho{rho}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)

    # Update the central index with metadata and summary metrics for this run
    # We use this to track completed runs and support resuming after interruption without re-running completed (opt, rho, seed) combos
    update_index(results_root, run_id, {
        "model": cfg["model"],
        "optimizer": opt_type,
        "rho": rho,
        "seed": seed,
        "test_acc": history[-1]["test_acc"],
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment": "baseline",
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
    })

    final = history[-1].copy()
    final["divergence_rate"] = divergence_rate(final["train_loss"], final["test_loss"])
    final["seed"] = seed
    final["optimizer"] = opt_type
    final["rho"] = rho
    final["run_id"] = run_id
    final["checkpoint"] = ckpt_path
    final["history"] = history
    return final


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runs_dir = cfg["runs_dir"]
    experiments_dir = cfg["experiments_dir"]
    results_root = os.path.dirname(runs_dir.rstrip("/\\"))
    model_name = cfg["model"]
    seeds = cfg["seeds"]
    opt_cfgs = cfg["optimizers"]

    all_results = []
    out_path = os.path.join(experiments_dir, "baseline", model_name, "baseline_results.json")

    # Build lookup of already-completed (opt, rho) groups for resume support.
    done_results = {}
    done_seeds = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        
        for _entry in all_results:
            agg = _entry.get("summary", {})
            opt = agg.get("optimizer")
            rho = agg.get("rho")
            for _ps in _entry.get("per_seed", []):
                done_seeds.add((opt, rho, _ps["seed"]))
            done_results[(opt, rho)] = _entry
        
        print(f"Resuming: {len(done_seeds)} (opt, rho, seed) combos already done.")

    for opt_name, opt_cfg in opt_cfgs.items():
        opt_type = opt_cfg["type"]
        rho_sweep = opt_cfg.get("rho_sweep", [0.0])
        
        # Inject per-optimizer eta into cfg so build_optimizer can read it
        if opt_type == "asam":
            cfg["asam_eta"] = opt_cfg.get("eta", 0.01)

        for rho in rho_sweep:
            combo_key = (opt_name, rho)
            ckpt_dir_check = os.path.join(experiments_dir, "baseline", model_name, "checkpoints")
            
            # Check if all seeds for this (opt, rho) combo are already done (checkpoint exists) before deciding to skip or run
            all_seeds_done = (
                combo_key in done_results
                and all(
                    os.path.exists(os.path.join(ckpt_dir_check, f"{opt_type}_rho{rho}_seed{seed}.pt"))
                    for seed in seeds
                )
            )
            if all_seeds_done:
                print(f"\n[{model_name}] opt={opt_name} rho={rho} — fully done, skipping")
                continue

            # Seed results already saved from a previous partial run
            per_seed = []
            if combo_key in done_results:
                per_seed = list(done_results[combo_key]["per_seed"])

            for seed in seeds:
                # Skip if checkpoint already exists (supports resuming after restart)
                ckpt_path_check = os.path.join(ckpt_dir_check, f"{opt_type}_rho{rho}_seed{seed}.pt")
                if os.path.exists(ckpt_path_check):
                    print(f"\n[{model_name}] opt={opt_name} rho={rho} seed={seed} — skipping (checkpoint exists)")
                    continue

                # If we reach here, this (opt, rho, seed) combo has not been completed yet, so we run it
                print(f"\n[{model_name}] opt={opt_name} rho={rho} seed={seed}")
                result = run_single(cfg, opt_type, rho, seed, runs_dir, experiments_dir, results_root)
                per_seed.append(result)

    
            _non_numeric_fields = {"history", "seed", "optimizer", "checkpoint", "run_id", "model"}

            # Aggregate only the numeric fields across seeds for this (opt, rho) combo, and save the aggregated summary along with per-seed results
            agg = aggregate_seeds(
                [{k: v for k, v in r.items() if k not in _non_numeric_fields} for r in per_seed]
            )
            agg["optimizer"] = opt_name
            agg["rho"] = rho
            agg["model"] = model_name
            entry = {"summary": agg, "per_seed": per_seed}
            
            # TODO: simplify this loop later.
            if combo_key in done_results:
                all_results = [r for r in all_results
                               if not (r.get("summary", {}).get("optimizer") == opt_name
                                       and r.get("summary", {}).get("rho") == rho)]
            all_results.append(entry)
            done_results[combo_key] = entry
            
            # Write after every (opt, rho) group so results survive preemption.
            save_results(out_path, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
