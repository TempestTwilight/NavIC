# 02 — Record type `Accel`

**What it is:** calibrated accelerometer record (~200 Hz, 29 sessions).

**Volume [OBSERVED]:** 6,369,268 rows across **29 sessions** (files containing this type); time range 1,792,380,080,294 → 1,793,074,859,028 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Accel.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms).

**Why it matters:** Calendar anchor.

**Use / project relevance:** Joins.

**Caveats:** Standard.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6,115,053  *(approx_count_distinct)*
- **min:** 1,792,380,080,294
- **max:** 1,793,074,859,028
- **mean:** 1,792,837,801,017
- **std:** 196,848,919
- **p50:** 1,792,757,932,379
- **p95:** 1,793,073,536,553
- **p99:** 1,793,074,607,522

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns) at ~200 Hz sampling.

**Why it matters:** Alignment clock.

**Use / project relevance:** Gravity check (Phase G): mean |z| ≈ 9.8 when phone flat proves the axis convention.

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6,453,182  *(approx_count_distinct)*
- **min:** 249,885,469,277
- **max:** 692,152,599,181,250
- **mean:** 455,260,430,529,263
- **std:** 196,464,539,455,608
- **p50:** 375,230,022,448,093
- **p95:** 690,819,318,527,899
- **p99:** 691,902,750,866,655

## `AccelXMps2`

**What:** Calibrated accelerometer X (m/s²), device frame (screen-up convention: +z out of the screen).

**Why it matters:** Linear acceleration including gravity.

**Use / project relevance:** Feature: motion state (static → gravity only); Phase G gravity check.

**Caveats:** Includes gravity — subtract 9.81 norm for linear motion; 29 sessions only (Samsung), so cross-device coverage is partial.

**Status:** OBSERVED (29 sessions) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 9,770  *(approx_count_distinct)*
- **min:** -61.04
- **max:** 57.03
- **mean:** 0.03697
- **std:** 1.998
- **p50:** 0.04106
- **p95:** 0.6855
- **p99:** 6.922
- **zeros:** 57,256 (0.9%)
- **negatives:** 2,762,305 (43.4%)

## `AccelYMps2`

**What:** Calibrated accelerometer Y (m/s²).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same; median 6.35 observed → phone was tilted, not flat, in the Samsung sessions.

**Status:** OBSERVED (median 6.35) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 10,013  *(approx_count_distinct)*
- **min:** -57.64
- **max:** 57.16
- **mean:** 4.676
- **std:** 3.99
- **p50:** 6.224
- **p95:** 9.921
- **p99:** 9.935
- **zeros:** 41 (0.0%)
- **negatives:** 992,480 (15.6%)

## `AccelZMps2`

**What:** Calibrated accelerometer Z (m/s²).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 12,245  *(approx_count_distinct)*
- **min:** -60.56
- **max:** 74.81
- **mean:** 1.428
- **std:** 7.389
- **p50:** 1.978
- **p95:** 9.938
- **p99:** 9.962
- **zeros:** 18 (0.0%)
- **negatives:** 2,573,898 (40.4%)

## `CalibrationAccuracy`

**What:** Sensor confidence code 0-3.

**Why it matters:** Trust level of calibration.

**Use / project relevance:** Filter.

**Caveats:** Check constancy.

**Status:** OBSERVED IN DATA + REFERENCE (code meaning)

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

