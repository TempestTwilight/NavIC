"""Experiment C: Learning-to-Rank via epoch-relative target.
Normalizes target within each epoch to directly train selection ranking.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_and_preprocess_data, FEATURE_COLS


def run_experiment_c(sample_n: int | None = None):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(sample_n=sample_n)
    data["epoch_mean"] = data.groupby(["session_id", "UnixTimeMillis"])[
        "target_cn0_raw"
    ].transform("mean")
    data["target_relative"] = data["target_cn0_raw"] - data["epoch_mean"]

    X = data[FEATURE_COLS].astype(np.float32)
    y_rel = data["target_relative"].astype(np.float32)
    devices = data["device_id"]

    records = []
    for dev in devices.unique():
        train_mask = devices != dev
        test_mask = devices == dev
        test_data = data[test_mask].copy()

        model = HistGradientBoostingRegressor(
            max_iter=60, max_depth=8, min_samples_leaf=30, random_state=42
        )
        model.fit(X[train_mask], y_rel[train_mask])
        test_data["pred_rel"] = model.predict(X[test_mask])

        reg_ltr, reg_persist = [], []
        top1_ltr, top1_persist = [], []

        for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
            if len(g) < 2:
                continue
            target_series = g["target_cn0_raw"].to_numpy(dtype=float)
            best_idx = np.argmax(target_series)
            best_val = target_series[best_idx]

            pred_rel_arr = g["pred_rel"].to_numpy(dtype=float)
            cn0_arr = g["Cn0DbHz"].to_numpy(dtype=float)

            ltr_choice = np.argmax(pred_rel_arr)
            persist_choice = np.argmax(cn0_arr)

            reg_ltr.append(best_val - target_series[ltr_choice])
            reg_persist.append(best_val - target_series[persist_choice])
            top1_ltr.append(int(ltr_choice == best_idx))
            top1_persist.append(int(persist_choice == best_idx))

        records.append(
            {
                "held_out_device": dev,
                "persistence_mean_regret": np.mean(reg_persist),
                "ltr_mean_regret": np.mean(reg_ltr),
                "persistence_top1_acc": np.mean(top1_persist),
                "ltr_top1_acc": np.mean(top1_ltr),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(results_dir / "ltr_results.csv", index=False)
    print("Experiment C completed.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_experiment_c()
