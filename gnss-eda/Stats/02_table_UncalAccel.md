# 02 — Record type `UncalAccel`

**What it is:** uncalibrated accelerometer record (60 sessions).

**Volume [OBSERVED]:** 8,365,329 rows across **60 sessions** (files containing this type); time range 1,780,543,010,888 → 1,793,073,942,186 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/UncalAccel.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms).

**Why it matters:** Calendar anchor.

**Use / project relevance:** Joins.

**Caveats:** Standard.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 8,190,057  *(approx_count_distinct)*
- **min:** 1,780,543,010,888
- **max:** 1,793,073,942,186
- **mean:** 1,789,911,666,759
- **std:** 5,137,047,925
- **p50:** 1,792,693,364,346
- **p95:** 1,793,056,055,625
- **p99:** 1,793,073,091,662

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns).

**Why it matters:** Alignment clock.

**Use / project relevance:** Joins; 60 sessions present.

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 8,117,573  *(approx_count_distinct)*
- **min:** 139,220,886,052
- **max:** 1,898,566,083,532,778
- **mean:** 806,181,010,811,230
- **std:** 544,616,366,693,732
- **p50:** 644,332,532,204,916
- **p95:** 1,839,123,118,879,937
- **p99:** 1,898,141,474,695,334

## `UncalAccelXMps2`

**What:** Uncalibrated accelerometer X: raw sensor output WITHOUT the vendor bias removed (m/s²).

**Why it matters:** Raw dunder signal — bias columns let you reproduce the calibrated value yourself.

**Use / project relevance:** Redundant with Accel for the project; document as fallback.

**Caveats:** Bias* columns are 100% zeros observed → the vendor never reports biases, so uncal == cal + (unknown bias); prefer calibrated Accel where present.

**Status:** OBSERVED (Bias*=0) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 18,068  *(approx_count_distinct)*
- **min:** -38.38
- **max:** 51.43
- **mean:** 0.3288
- **std:** 3.158
- **p50:** 0.02169
- **p95:** 7.198
- **p99:** 9.835
- **zeros:** 26,239 (0.3%)
- **negatives:** 3,776,196 (45.1%)

## `UncalAccelYMps2`

**What:** Uncalibrated accelerometer Y (m/s²).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 16,895  *(approx_count_distinct)*
- **min:** -51.62
- **max:** 53.99
- **mean:** 4.164
- **std:** 3.919
- **p50:** 6.127
- **p95:** 9.638
- **p99:** 9.763
- **zeros:** 8,597 (0.1%)
- **negatives:** 2,334,366 (27.9%)

## `UncalAccelZMps2`

**What:** Uncalibrated accelerometer Z (m/s²).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 21,228  *(approx_count_distinct)*
- **min:** -65.55
- **max:** 62.35
- **mean:** 0.6801
- **std:** 7.228
- **p50:** 0.6508
- **p95:** 9.84
- **p99:** 9.864
- **negatives:** 3,700,592 (44.2%)

## `BiasXMps2`

**What:** Reported accelerometer bias compensation for X (m/s²).

**Why it matters:** Should convert uncal → calibrated: cal = uncal − bias.

**Use / project relevance:** Cross-check vs calibrated Accel; here: dead weight (all zeros).

**Caveats:** 100% zeros observed — vendor does not populate.

**Status:** OBSERVED (constant 0) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 8,365,329 (100.0%)

## `BiasYMps2`

**What:** Bias compensation for Y.

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** All zeros.

**Status:** OBSERVED (constant 0) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 8,365,329 (100.0%)

## `BiasZMps2`

**What:** Bias compensation for Z.

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** All zeros.

**Status:** OBSERVED (constant 0) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 8,365,329 (100.0%)

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

