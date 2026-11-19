"""Experiment A: Baseline Ridge & HistGB with absolute C/N0 selection regret.
Addresses §5.4: compares realized future C/N0 regret against persistence.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_and_preprocess_data, FEATURE_COLS


def run_experiment_a(sample_n: int | None = None):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(sample_n=sample_n)
    X = data[FEATURE_COLS].astype(np.float32)
    y_delta = data["target_delta_cn0"].astype(np.float32)
    devices = data["device_id"]

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    records = []

    for dev in devices.unique():
        train_mask = devices != dev
        test_mask = devices == dev
        test_data = data[test_mask].copy()

        model.fit(X[train_mask], y_delta[train_mask])
        pred_delta = model.predict(X[test_mask])

        test_data["pred_raw_model"] = test_data["Cn0DbHz"] + pred_delta
        test_data["pred_raw_persist"] = test_data["Cn0DbHz"]

        reg_model, reg_persist = [], []
        for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
            if len(g) < 2:
                continue
            best = g["target_cn0_raw"].max()
            reg_persist.append(
                best - g.loc[g["pred_raw_persist"].idxmax(), "target_cn0_raw"]
            )
            reg_model.append(
                best - g.loc[g["pred_raw_model"].idxmax(), "target_cn0_raw"]
            )

        records.append(
            {
                "held_out_device": dev,
                "n_epochs": len(reg_model),
                "persistence_mean_regret": np.mean(reg_persist),
                "model_mean_regret": np.mean(reg_model),
                "delta_improvement": np.mean(reg_persist) - np.mean(reg_model),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(results_dir / "baseline_regret.csv", index=False)
    print("Experiment A completed.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_experiment_a()
