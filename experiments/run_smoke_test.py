import argparse
import tempfile
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments import run_baseline, run_flatness, run_reparam


def shared_config(work_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "seeds": [0],
        "data_dir": str(repo_root / "data"),
        "runs_dir": str(work_dir / "runs"),
        "experiments_dir": str(work_dir / "experiments"),
        "num_workers": 0,
        "optimizers": {
            "sgd": {"type": "sgd"},
            "sam": {"type": "sam", "rho_sweep": [0.05]},
            "asam": {"type": "asam", "eta": 0.01, "rho_sweep": [0.05]},
            "msam": {"type": "msam", "rho_sweep": [0.05]},
        },
    }


def reparam_overrides() -> dict:
    """Extra keys required by run_reparam (on top of shared config)."""
    return {
        "alpha_values": [2.0],
        # reparam uses the same optimizers dict; rho_sweep is already per-optimizer
    }


def build_smoke_configs(work_dir: Path) -> list[dict]:
    shared = shared_config(work_dir)
    reparam_extra = reparam_overrides()
    resnet_base = {
        **shared,
        "model": "resnet18",
        "epochs": 1,
        "batch_size": 1,
        "max_samples": 1,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5.0e-4,
        "resize": None,
    }
    vit_base = {
        **shared,
        "model": "vit_b_32",
        "epochs": 1,
        "batch_size": 1,
        "max_samples": 1,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 5.0e-4,
        "resize": 224,
        "pretrained": True,
    }
    return [
        {"baseline": resnet_base, "reparam": {**resnet_base, **reparam_extra}},
        {"baseline": vit_base,   "reparam": {**vit_base,   **reparam_extra}},
    ]


def write_config(config: dict, work_dir: Path, suffix: str = "") -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    name = f"smoke_{config['model']}{suffix}.yaml"
    config_path = work_dir / name
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResNet-18 and ViT-B/32 smoke tests.")
    parser.add_argument(
        "--work-dir",
        default=Path(tempfile.gettempdir()) / "sam-opt-smoke",
        type=Path,
        help="Directory where smoke config and outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for the smoke run.",
    )
    args = parser.parse_args()

    for configs in build_smoke_configs(args.work_dir):
        baseline_cfg = configs["baseline"]
        reparam_cfg = configs["reparam"]
        model = baseline_cfg["model"]

        baseline_path = write_config(baseline_cfg, args.work_dir, "_baseline")
        print(f"[{model}] Smoke baseline config written at {baseline_path}")
        print(f"[{model}] Running baseline smoke sweep...")
        run_baseline.main(str(baseline_path))

        ckpt_dir = Path(baseline_cfg["experiments_dir"]) / "baseline" / model / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found for flatness run in: {ckpt_dir}")

        print(f"[{model}] Running flatness smoke check on {len(ckpt_files)} checkpoints...")
        run_flatness.main(str(baseline_path), None, args.seed)

        reparam_path = write_config(reparam_cfg, args.work_dir, "_reparam")
        print(f"[{model}] Smoke reparam config written at {reparam_path}")
        print(f"[{model}] Running reparam smoke sweep...")
        run_reparam.main(str(reparam_path))

        reparam_ckpt_dir = Path(reparam_cfg["experiments_dir"]) / "reparam" / model / "checkpoints"
        reparam_ckpt_files = sorted(reparam_ckpt_dir.glob("*.pt"))
        if not reparam_ckpt_files:
            raise FileNotFoundError(f"No reparam checkpoints found in: {reparam_ckpt_dir}")
        print(f"[{model}] Reparam checkpoints verified ({len(reparam_ckpt_files)} found).")

        print(f"[{model}] Running flatness smoke check on reparam checkpoints...")
        run_flatness.main(str(reparam_path), None, args.seed, experiment="reparam")

    print(f"Smoke outputs saved under {args.work_dir}")


if __name__ == "__main__":
    main()