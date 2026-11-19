"""Experiment E: Survival-Weighted Expected-Value Selection.
Models expected C/N0 factoring in track survival probability:
E[C/N0] = P(survive) * predicted_C/N0 + (1 - P(survive)) * floor.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_and_preprocess_data


def run_experiment_e(sample_n: int | None = None, sensitivity_floor: float = 12.0):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(sample_n=sample_n)

    # Parametric survival probability from elevation and margin above sensitivity floor
    # Calibrated against Table 21 empirical survival curves
    el = data["ElevationDegrees"].to_numpy(dtype=float)
    cn0 = data["Cn0DbHz"].to_numpy(dtype=float)
    p_survive = 1.0 / (1.0 + np.exp(-(0.06 * el + 0.12 * (cn0 - 15.0))))
    data["p_survive"] = p_survive

    # Expected utility under persistence
    data["expected_utility"] = (
        data["p_survive"] * data["Cn0DbHz"]
        + (1.0 - data["p_survive"]) * sensitivity_floor
    )

    devices = data["device_id"]
    records = []

    for dev in devices.unique():
        test_data = data[devices == dev]
        reg_pure, reg_survival = [], []

        for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
            if len(g) < 2:
                continue
            best = g["target_cn0_raw"].max()
            pure_choice = g.loc[g["Cn0DbHz"].idxmax(), "target_cn0_raw"]
            surv_choice = g.loc[g["expected_utility"].idxmax(), "target_cn0_raw"]
            reg_pure.append(best - pure_choice)
            reg_survival.append(best - surv_choice)

        records.append(
            {
                "held_out_device": dev,
                "pure_persistence_regret": np.mean(reg_pure),
                "survival_weighted_regret": np.mean(reg_survival),
                "regret_change": np.mean(reg_survival) - np.mean(reg_pure),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(results_dir / "survival_results.csv", index=False)
    print("Experiment E completed.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_experiment_e()
