#!/usr/bin/env python3
"""Recompute the Kotlin unit-test expectations with the original joblib models.

The Kotlin test file stores three 46-feature vectors and three 24-feature
vectors plus the expected predictions.  This script loads the scikit-learn
pipelines that produced ModelArtifacts.kt and checks the same vectors, so a
failure isolates either an export error or a Kotlin arithmetic error.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "android/app/src/test/java/org/isro/gnssoptimizer/ModelInferenceTest.kt"
ARTIFACTS = {
    "temporal": ROOT / "model/temporal_artifacts",
    "orientation": ROOT / "model/orientation_artifacts",
}
MODELS = {
    "temporal": ARTIFACTS["temporal"] / "ridge_temporal_1s.joblib",
    "orientation": ARTIFACTS["orientation"] / "ridge_orientation.joblib",
}
NUMBER = re.compile(r"[-+]?\d+\.\d+e[-+]\d+")


def block(source: str, name: str) -> str:
    return source.split(name, 1)[1].split("private val", 1)[0]


def vectors(text: str) -> list[np.ndarray]:
    return [
        np.array([float(v) for v in NUMBER.findall(body)])
        for body in re.findall(r"doubleArrayOf\((.*?)\)", text, re.S)
    ]


def main() -> None:
    source = TEST_FILE.read_text()
    cases = {
        "temporal": (
            vectors(block(source, "temporalTestInputs")),
            np.array([float(v) for v in NUMBER.findall(block(source, "temporalExpectedOutputs"))]),
        ),
        "orientation": (
            vectors(block(source, "orientationTestInputs")),
            np.array([float(v) for v in NUMBER.findall(block(source, "orientationExpectedOutputs"))]),
        ),
    }

    for label, (inputs, expected) in cases.items():
        pipeline = joblib.load(MODELS[label])
        features = json.loads((ARTIFACTS[label] / "feature_cols.json").read_text())
        assert len(features) == pipeline.n_features_in_, f"{label}: feature list mismatch"

        predictions = np.array([pipeline.predict(x.reshape(1, -1))[0] for x in inputs])
        print(f"{label}: {len(inputs)} vectors, {inputs[0].size} features")
        for index, (prediction, wanted) in enumerate(zip(predictions, expected)):
            print(f"  vector {index}: python={prediction:.12e} kotlin={wanted:.12e}")
        print(f"  max abs diff: {np.max(np.abs(predictions - expected)):.3e}")


if __name__ == "__main__":
    main()
