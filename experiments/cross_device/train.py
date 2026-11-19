"""Experiment D: Leave-One-Device-Out Generalization.
Compares raw C/N0 target vs residual delta target across unseen devices.
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


def run_experiment_d(sample_n: int | None = None):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(sample_n=sample_n)
    X = data[FEATURE_COLS].astype(np.float32)
    y_raw = data["target_cn0_raw"].astype(np.float32)
    y_delta = data["target_delta_cn0"].astype(np.float32)
    devices = data["device_id"]

    records = []
    for dev in devices.unique():
        train_mask = devices != dev
        test_mask = devices == dev
        test_data = data[test_mask].copy()

        m_raw = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        m_raw.fit(X[train_mask], y_raw[train_mask])
        test_data["pred_raw_strategy"] = m_raw.predict(X[test_mask])

        m_delta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        m_delta.fit(X[train_mask], y_delta[train_mask])
        test_data["pred_delta_strategy"] = test_data["Cn0DbHz"] + m_delta.predict(
            X[test_mask]
        )

        reg_raw, reg_delta = [], []
        for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
            if len(g) < 2:
                continue
            best = g["target_cn0_raw"].max()
            reg_raw.append(
                best - g.loc[g["pred_raw_strategy"].idxmax(), "target_cn0_raw"]
            )
            reg_delta.append(
                best - g.loc[g["pred_delta_strategy"].idxmax(), "target_cn0_raw"]
            )

        records.append(
            {
                "held_out_device": dev,
                "raw_strategy_mean_regret": np.mean(reg_raw),
                "delta_strategy_mean_regret": np.mean(reg_delta),
                "delta_improvement": np.mean(reg_raw) - np.mean(reg_delta),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(results_dir / "cross_device_results.csv", index=False)
    print("Experiment D completed.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_experiment_d()
