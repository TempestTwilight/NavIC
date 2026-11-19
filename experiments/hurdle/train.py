"""Experiment B: Two-Stage Hurdle Model.
Stage 1: Predict whether signal will fade (|delta| > 1.5 dB).
Stage 2: Specialized regressor trained only on fading instances.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_and_preprocess_data, FEATURE_COLS


def run_experiment_b(sample_n: int | None = None):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(sample_n=sample_n)
    X = data[FEATURE_COLS].astype(np.float32)
    y_delta = data["target_delta_cn0"].astype(np.float32)
    y_fade = data["is_fading"].astype(np.float32)
    devices = data["device_id"]

    records = []
    for dev in devices.unique():
        train_mask = devices != dev
        test_mask = devices == dev

        clf = HistGradientBoostingClassifier(max_iter=60, random_state=42)
        clf.fit(X[train_mask], y_fade[train_mask])
        fade_prob = clf.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y_fade[test_mask], fade_prob)

        fade_train = train_mask & (y_fade == 1)
        reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        reg.fit(X[fade_train], y_delta[fade_train])

        pred_delta_hurdle = np.where(fade_prob > 0.5, reg.predict(X[test_mask]), 0.0)

        fade_test = y_fade[test_mask] == 1
        rmse_hurdle = np.sqrt(
            mean_squared_error(
                y_delta[test_mask][fade_test], pred_delta_hurdle[fade_test]
            )
        )
        rmse_persist = np.sqrt(
            mean_squared_error(y_delta[test_mask][fade_test], np.zeros(fade_test.sum()))
        )

        records.append(
            {
                "held_out_device": dev,
                "fading_classifier_auc": auc,
                "persistence_fading_rmse": rmse_persist,
                "hurdle_fading_rmse": rmse_hurdle,
                "fading_rmse_reduction": rmse_persist - rmse_hurdle,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(results_dir / "hurdle_results.csv", index=False)
    print("Experiment B completed.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_experiment_b()
