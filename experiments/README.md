# GNSS Signal Prediction Experiments

This directory contains experiment scripts for improving temporal and orientation model accuracy for satellite selection.

## Structure

```
experiments/
├── baseline/           # Experiment A: Ridge regression with proper CV
├── hurdle/             # Experiment B: Two-stage hurdle model
├── ltr/                # Experiment C: Learning-to-Rank formulation
├── cross_device/       # Experiment D: Leave-one-device-out generalization
├── survival/           # Experiment E: Survival-weighted selection
└── features/           # Feature engineering experiments
```

## Data Sources (Read-Only)

- **Raw GNSS data**: `/home/tempest/Documents/ISRO_Project/data/Raw_Logs/`
- **Phase 1 typed parquet**: `/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/`
- **Session metadata**: `/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv`

## Resource Constraints

- **RAM limit**: 2 GB (system constraint)
- **DuckDB memory**: 1 GB per connection
- **Threads**: 2
- **All aggregation in SQL**, only small results (<10,000 rows) into pandas

## Key Baseline to Beat

**Persistence baseline**: 0.16 dB mean regret on 922 continuously-tracked epochs (table 26).
- Note: This excludes dropout regret; corpus-wide baseline would be higher.
- Any ML model must demonstrate <0.16 dB mean regret with 95% CI across multiple held-out days.

## Running Experiments

Each experiment directory contains:
- `config.py` — Experiment-specific configuration
- `features.sql` — DuckDB SQL for feature extraction
- `train.py` — Model training script (to be run after verification)
- `evaluate.py` — Evaluation metrics and regret computation
- `results/` — Output directory for artifacts

## CV Strategy

**CRITICAL**: All experiments must use **calendar-day grouped cross-validation** (`GroupKFold` on recording date).
- Random splits cause data leakage (92.4% satellite overlap within 2 hours)
- Device is confounded with date, so device-based splits also leak

## Target Definition

**Primary target**: Residual Δ C/N0 at 15-second horizon
```
target = C/N0(t+15s) - C/N0(t)
```

**Alternative targets** (for specific experiments):
- Binary: |Δ C/N0| > 1.5 dB (fading classifier)
- Ranking: Relative future C/N0 within epoch (for LTR)

## Verification Checklist

Before running any training:
- [ ] Feature SQL runs without memory errors
- [ ] Feature availability verified across all 4 devices
- [ ] CV grouping column (recording date) is present
- [ ] Target distribution sanity-checked (no extreme outliers from zero-masking)
- [ ] Baseline metrics computed on same evaluation set
