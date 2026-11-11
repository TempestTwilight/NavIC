# 02 — Record type `UncalGyro`

**What it is:** uncalibrated gyroscope record (~200 Hz).

**Volume [OBSERVED]:** 16,661,992 rows across **89 sessions** (files containing this type); time range 1,780,543,010,888 → 1,793,074,859,028 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/UncalGyro.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms).

**Why it matters:** Calendar anchor.

**Use / project relevance:** Joins.

**Caveats:** Standard.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 15,808,879  *(approx_count_distinct)*
- **min:** 1,780,543,010,888
- **max:** 1,793,074,859,028
- **mean:** 1,791,364,465,349
- **std:** 3,923,867,133
- **p50:** 1,792,753,393,231
- **p95:** 1,793,072,549,637
- **p99:** 1,793,074,168,564

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns); ~200 Hz sampling, 16.66 M rows.

**Why it matters:** Alignment clock.

**Use / project relevance:** Integration check (integrate gyro → compare with quaternion rotation).

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 16,317,868  *(approx_count_distinct)*
- **min:** 139,220,886,052
- **max:** 1,898,566,083,532,778
- **mean:** 611,029,904,687,096
- **std:** 456,600,176,834,641
- **p50:** 585,964,686,613,361
- **p95:** 1,807,233,048,495,037
- **p99:** 1,897,441,575,169,847

## `UncalGyroXRadPerSec`

**What:** Uncalibrated angular rate about X (rad/s), bias NOT removed.

**Why it matters:** Raw rotation rate of the phone.

**Use / project relevance:** Motion features (is the phone being moved?); gyro integration for orientation deltas.

**Caveats:** Spikes to ±23.6 rad/s observed — beyond any plausible hand motion (~±10 rad/s max); check whether spikes align with sample-time gaps (jitter) — Q for Phase D.

**Status:** OBSERVED (±23.6 spikes) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 39,954  *(approx_count_distinct)*
- **min:** -9.645
- **max:** 12.55
- **mean:** 0.009962
- **std:** 0.1493
- **p50:** 0.001287
- **p95:** 0.0279
- **p99:** 0.1114
- **zeros:** 338,268 (2.0%)
- **negatives:** 4,644,509 (27.9%)

## `UncalGyroYRadPerSec`

**What:** Uncalibrated rate about Y (rad/s).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 42,128  *(approx_count_distinct)*
- **min:** -23.6
- **max:** 20.01
- **mean:** -0.01688
- **std:** 0.1808
- **p50:** -0.0003746
- **p95:** 0.002155
- **p99:** 0.09015
- **zeros:** 1,203,979 (7.2%)
- **negatives:** 9,302,860 (55.8%)

## `UncalGyroZRadPerSec`

**What:** Uncalibrated rate about Z (rad/s).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 40,944  *(approx_count_distinct)*
- **min:** -12.02
- **max:** 16.03
- **mean:** 0.01303
- **std:** 0.1649
- **p50:** 0.001772
- **p95:** 0.03635
- **p99:** 0.09898
- **zeros:** 1,401,930 (8.4%)
- **negatives:** 2,493,087 (15.0%)

## `DriftXRadPerSec`

**What:** Reported gyro bias estimate for X (rad/s).

**Why it matters:** Convert uncal → calibrated: cal = uncal − drift.

**Use / project relevance:** Bias-stability analysis of the gyro.

**Caveats:** Check whether populated (unlike UncalAccel biases which were all zero).

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 772  *(approx_count_distinct)*
- **min:** -0.002362
- **max:** 0.0281
- **mean:** 0.00978
- **std:** 0.01235
- **p50:** 0.0005
- **p95:** 0.02627
- **p99:** 0.02688
- **negatives:** 3,347,612 (20.1%)

## `DriftYRadPerSec`

**What:** Gyro bias estimate for Y (rad/s).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 750  *(approx_count_distinct)*
- **min:** -0.05315
- **max:** 0.001641
- **mean:** -0.01709
- **std:** 0.02244
- **p50:** 0.0005
- **p95:** 0.001021
- **p99:** 0.001295
- **negatives:** 6,452,756 (38.7%)

## `DriftZRadPerSec`

**What:** Gyro bias estimate for Z (rad/s).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 743  *(approx_count_distinct)*
- **min:** -0.0002033
- **max:** 0.0391
- **mean:** 0.01314
- **std:** 0.01589
- **p50:** 0.0006873
- **p95:** 0.03543
- **p99:** 0.03658
- **negatives:** 3,261 (0.0%)

## `CalibrationAccuracy`

**What:** Sensor confidence code 0-3.

**Why it matters:** Trust level.

**Use / project relevance:** Filter.

**Caveats:** Check constancy.

**Status:** OBSERVED IN DATA + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 3
- **max:** 3
- **mean:** 3
- **std:** 0
- **p50:** 3
- **p95:** 3
- **p99:** 3

