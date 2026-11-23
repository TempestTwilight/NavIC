#!/usr/bin/env python3
"""Entry point executed inside the Kaggle environment.

Sets up environment paths pointing to Kaggle datasets, installs any missing
packages, runs the training pipelines, writes artifacts to /kaggle/working,
and emits an artifact manifest with SHA256 hashes.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"FAILED: {' '.join(cmd)} with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    print("=" * 70)
    print("ISRO GNSS Model Training on Kaggle")
    print("=" * 70)

    # 1. Locate inputs
    kaggle_input = Path("/kaggle/input")
    data_dir = None
    code_dir = None

    # Search for data and code datasets dynamically
    for d in kaggle_input.iterdir():
        if not d.is_dir():
            continue
        if (d / "combined_corpus.parquet").exists():
            data_dir = d
        elif (d / "results" / "merged_data" / "combined_corpus.parquet").exists():
            data_dir = d / "results" / "merged_data"
        if (d / "src" / "isro_project").exists():
            code_dir = d

    # If running in a kernel pushed with code in working directory
    working_dir = Path("/kaggle/working")
    if code_dir is None:
        if (working_dir / "src" / "isro_project").exists():
            code_dir = working_dir
        else:
            code_dir = Path(__file__).resolve().parents[1]

    if data_dir is None:
        # Fallback to standard slug name
        standard_data = kaggle_input / "isro-gnss-data"
        if standard_data.exists():
            data_dir = standard_data
        else:
            print("ERROR: Could not find combined_corpus.parquet under /kaggle/input")
            sys.exit(1)

    corpus_path = (
        data_dir / "combined_corpus.parquet"
        if (data_dir / "combined_corpus.parquet").exists()
        else data_dir / "results" / "merged_data" / "combined_corpus.parquet"
    )
    print(f"Code root:   {code_dir}")
    print(f"Corpus path: {corpus_path}")

    # 2. Configure output paths under /kaggle/working
    out_models = working_dir / "model"
    out_experiments = working_dir / "experiments"
    out_results = working_dir / "results"
    for d in [out_models, out_experiments, out_results]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Setup environment variables
    env = os.environ.copy()
    env["ISRO_PROJECT_ROOT"] = str(working_dir)
    env["ISRO_MERGED_CORPUS"] = str(corpus_path)
    env["ISRO_MODEL_DIR"] = str(out_models)
    env["ISRO_RESULTS_DIR"] = str(out_results)
    env["PYTHONPATH"] = f"{code_dir / 'src'}:{code_dir / 'experiments'}:{env.get('PYTHONPATH', '')}"

    # Also sync experiments folder structure into working dir so train scripts find their results folders
    for exp in ["baseline", "hurdle", "ltr", "cross_device", "survival"]:
        (out_experiments / exp / "results").mkdir(parents=True, exist_ok=True)

    # 4. Train orientation model
    print("\n" + "=" * 50)
    print("Training Orientation Model (src/isro_project/02_orientation_model.py)")
    print("=" * 50)
    run_cmd([sys.executable, "-m", "isro_project.02_orientation_model"], env)

    # 5. Train temporal future model
    print("\n" + "=" * 50)
    print("Training Temporal Model (src/isro_project/02_future_model.py)")
    print("=" * 50)
    run_cmd([sys.executable, "-m", "isro_project.02_future_model"], env)

    # 6. Run all research experiments (A-E)
    print("\n" + "=" * 50)
    print("Running Experiments A-E (experiments/run_all.py)")
    print("=" * 50)
    run_cmd([sys.executable, str(code_dir / "experiments" / "run_all.py"), "--exp", "all"], env)

    # 7. Print manifest of all generated artifacts with sha256
    print("\n" + "=" * 70)
    print("ARTIFACTS:")
    print("=" * 70)
    manifest = []
    for root, _, files in os.walk(working_dir):
        for f in sorted(files):
            fp = Path(root) / f
            rel = fp.relative_to(working_dir)
            if str(rel).startswith("."):
                continue
            h = sha256_file(fp)
            size = fp.stat().st_size
            print(f"{rel} | {size} bytes | sha256:{h}")
            manifest.append(f"{rel},{size},{h}\n")

    manifest_path = working_dir / "artifacts_manifest.csv"
    manifest_path.write_text("path,size_bytes,sha256\n" + "".join(manifest))
    print(f"\nWrote manifest to {manifest_path}")
    print("Training on Kaggle completed successfully.")


if __name__ == "__main__":
    main()
