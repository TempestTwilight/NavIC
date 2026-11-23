#!/usr/bin/env python3
"""Entry point executed inside the Kaggle environment.

Locates the Kaggle datasets, installs requirements, runs the training
pipelines, writes artifacts to /kaggle/working, and emits an artifact
manifest with SHA256 hashes.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
EXPERIMENTS = ["baseline", "hurdle", "ltr", "cross_device", "survival"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], env: dict[str, str], fatal: bool = True) -> bool:
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"FAILED: {' '.join(cmd)} with exit code {res.returncode}")
        if fatal:
            sys.exit(res.returncode)
        return False
    return True


def fail_with_listing(msg: str) -> None:
    """Exit with an error and show what is actually mounted (can't inspect Kaggle's FS otherwise)."""
    print(f"ERROR: {msg}")
    print(f"Contents of {KAGGLE_INPUT} (first 100 entries):")
    if KAGGLE_INPUT.exists():
        for p in sorted(KAGGLE_INPUT.rglob("*"))[:100]:
            print("  ", p)
    sys.exit(1)


def find_corpus() -> Path:
    matches = sorted(KAGGLE_INPUT.rglob("combined_corpus.parquet"))
    if not matches:
        fail_with_listing("combined_corpus.parquet not found under /kaggle/input")
    return matches[0]


def find_code_root() -> Path:
    """Return the repo root, i.e. the directory that contains src/isro_project."""
    pkgs = [p for p in KAGGLE_INPUT.rglob("isro_project") if p.is_dir()]
    in_src = [p for p in pkgs if p.parent.name == "src"]
    chosen = (in_src or pkgs or [None])[0]
    if chosen is None:
        fail_with_listing("isro_project package not found under /kaggle/input")
    # .../<root>/src/isro_project -> <root>
    return chosen.parent.parent if chosen.parent.name == "src" else chosen.parent


def main():
    print("=" * 70)
    print("ISRO GNSS Model Training on Kaggle")
    print("=" * 70)

    # 1. Locate inputs (recursive search; fails loudly instead of guessing)
    corpus_path = find_corpus()
    code_dir = find_code_root()
    print(f"Code root:   {code_dir}")
    print(f"Corpus path: {corpus_path}")

    # 2. Configure output paths under /kaggle/working
    out_models = WORKING_DIR / "model"
    out_experiments = WORKING_DIR / "experiments"
    out_results = WORKING_DIR / "results"
    for d in [out_models, out_experiments, out_results]:
        d.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        (out_experiments / exp / "results").mkdir(parents=True, exist_ok=True)

    # 3. Environment variables
    env = os.environ.copy()
    # Outputs are written relative to this root; code is imported via PYTHONPATH.
    env["ISRO_PROJECT_ROOT"] = str(WORKING_DIR)
    env["ISRO_MERGED_CORPUS"] = str(corpus_path)
    env["ISRO_MODEL_DIR"] = str(out_models)
    env["ISRO_RESULTS_DIR"] = str(out_results)
    py_path = [str(code_dir / "src"), str(code_dir / "experiments")]
    if env.get("PYTHONPATH"):
        py_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path)

    # 4. Install requirements (non-fatal: pins may clash with Kaggle's preinstalled packages)
    req = code_dir / "requirements.txt"
    if req.exists():
        ok = run_cmd(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            env,
            fatal=False,
        )
        if not ok:
            print("WARNING: requirements install failed; continuing with preinstalled packages.")
    else:
        print("No requirements.txt found; using preinstalled packages.")

    # 5. Train orientation model
    print("\n" + "=" * 50)
    print("Training Orientation Model (src/isro_project/02_orientation_model.py)")
    print("=" * 50)
    run_cmd([sys.executable, "-m", "isro_project.02_orientation_model"], env)

    # 6. Train temporal future model
    print("\n" + "=" * 50)
    print("Training Temporal Model (src/isro_project/02_future_model.py)")
    print("=" * 50)
    run_cmd([sys.executable, "-m", "isro_project.02_future_model"], env)

    # 7. Run all research experiments (A-E)
    print("\n" + "=" * 50)
    print("Running Experiments A-E (experiments/run_all.py)")
    print("=" * 50)
    run_cmd(
        [sys.executable, str(code_dir / "experiments" / "run_all.py"), "--exp", "all"],
        env,
    )

    # 8. Manifest of generated artifacts with sha256
    print("\n" + "=" * 70)
    print("ARTIFACTS:")
    print("=" * 70)
    manifest = []
    for root, dirs, files in os.walk(WORKING_DIR):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d != "__pycache__"
        )
        for f in sorted(files):
            fp = Path(root) / f
            rel = fp.relative_to(WORKING_DIR)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            h = sha256_file(fp)
            size = fp.stat().st_size
            print(f"{rel} | {size} bytes | sha256:{h}")
            manifest.append(f"{rel},{size},{h}\n")

    manifest_path = WORKING_DIR / "artifacts_manifest.csv"
    manifest_path.write_text("path,size_bytes,sha256\n" + "".join(manifest))
    print(f"\nWrote manifest to {manifest_path}")
    print("Training on Kaggle completed successfully.")


if __name__ == "__main__":
    main()