"""Runner for all experiments (A through E).
Run on a target high-spec machine or cloud instance.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from baseline.train import run_experiment_a
from hurdle.train import run_experiment_b
from ltr.train import run_experiment_c
from cross_device.train import run_experiment_d
from survival.train import run_experiment_e


def main():
    parser = argparse.ArgumentParser(
        description="Run GNSS signal optimization experiments."
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "a", "b", "c", "d", "e"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Subsample rows for fast testing (e.g. 50000)",
    )
    args = parser.parse_args()

    experiments = {
        "a": ("Experiment A (Baseline Ridge vs Absolute Regret)", run_experiment_a),
        "b": ("Experiment B (Two-Stage Hurdle Model)", run_experiment_b),
        "c": ("Experiment C (Learning-to-Rank Selection)", run_experiment_c),
        "d": ("Experiment D (Leave-One-Device-Out Generalization)", run_experiment_d),
        "e": ("Experiment E (Survival-Weighted Expected Utility)", run_experiment_e),
    }

    selected = experiments.keys() if args.exp == "all" else [args.exp]

    for key in selected:
        name, fn = experiments[key]
        print(f"\n{'=' * 70}\nRunning {name}\n{'=' * 70}")
        fn(sample_n=args.sample_n)


if __name__ == "__main__":
    main()
