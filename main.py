"""SAM Optimizer Research — CLI entry point.

Usage:
    python main.py baseline --config configs/resnet18_baseline.yaml
    python main.py baseline --config configs/vit_baseline.yaml
    python main.py reparam  --config configs/resnet18_reparam.yaml
    python main.py baseline --config configs/vit_reparam.yaml
    python main.py flatness --config configs/resnet18_baseline.yaml [--checkpoint path/to/model.pt]
    python main.py baseline --config configs/vit_baseline.yaml [--checkpoint path/to/model.pt]
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM / ASAM / M-SAM benchmark on CIFAR-10"
    )
    parser.add_argument(
        "experiment",
        choices=["baseline", "reparam", "flatness"],
        help="Which experiment to run.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="(flatness only) Path to a saved .pt model checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="(flatness only) Random seed.",
    )
    args = parser.parse_args()

    if args.experiment == "baseline":
        from experiments.run_baseline import main as run
        run(args.config)
    elif args.experiment == "reparam":
        from experiments.run_reparam import main as run
        run(args.config)
    elif args.experiment == "flatness":
        from experiments.run_flatness import main as run
        run(args.config, args.checkpoint, args.seed)
    else:
        print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

