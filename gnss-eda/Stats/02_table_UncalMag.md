# 02 — Record type `UncalMag`

**What it is:** uncalibrated magnetometer record (~55 Hz).

**Volume [OBSERVED]:** 6,723,621 rows across **89 sessions** (files containing this type); time range 1,780,543,010,910 → 1,793,074,859,029 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/UncalMag.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms).

**Why it matters:** Calendar anchor.

**Use / project relevance:** Joins.

**Caveats:** Standard.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6,496,592  *(approx_count_distinct)*
- **min:** 1,780,543,010,910
- **max:** 1,793,074,859,029
- **mean:** 1,791,906,183,435
- **std:** 3,192,948,384
- **p50:** 1,792,754,809,158
- **p95:** 1,793,072,904,555
- **p99:** 1,793,074,321,839

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns).

**Why it matters:** Alignment clock.

**Use / project relevance:** Joins.

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6,714,696  *(approx_count_distinct)*
- **min:** 139,219,826,208
- **max:** 1,898,566,085,870,577
- **mean:** 528,898,415,015,398
- **std:** 398,242,971,679,965
- **p50:** 383,660,404,119,386
- **p95:** 1,722,776,674,363,768
- **p99:** 1,876,899,470,567,461

## `UncalMagXMicroT`

**What:** Uncalibrated magnetometer X (µT) — raw field WITHOUT hard-iron bias removed.

**Why it matters:** Earth field + local magnetic environment in the device frame.

**Use / project relevance:** Absolute-heading source (with GRV); Phase G orientation checks.

**Caveats:** Huge DC offsets observed (X median −67 µT) — hard-iron bias dominates; must use Bias columns or calibrated Mag (which is never present!) to get the true field.

**Status:** OBSERVED (median −67 µT) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 89,814  *(approx_count_distinct)*
- **min:** -172.3
- **max:** 575.6
- **mean:** -70.86
- **std:** 27.63
- **p50:** -67.76
- **p95:** -17.52
- **p99:** -9.49
- **negatives:** 6,723,576 (100.0%)

## `UncalMagYMicroT`

**What:** Uncalibrated magnetometer Y (µT).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same as X.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 86,561  *(approx_count_distinct)*
- **min:** -28.89
- **max:** 254
- **mean:** 33.66
- **std:** 36.09
- **p50:** 26.25
- **p95:** 101
- **p99:** 114.3
- **zeros:** 7 (0.0%)
- **negatives:** 997,436 (14.8%)

## `UncalMagZMicroT`

**What:** Uncalibrated magnetometer Z (µT).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Z median −87.6 µT observed — see X.

**Status:** OBSERVED (−87.6 µT) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 95,149  *(approx_count_distinct)*
- **min:** -199.9
- **max:** 270.3
- **mean:** -44
- **std:** 109.7
- **p50:** -88.49
- **p95:** 159.8
- **p99:** 176.3
- **negatives:** 5,089,853 (75.7%)

## `BiasXMicroT`

**What:** Reported hard-iron bias for X (µT).

**Why it matters:** Convert uncal → calibrated: cal = uncal − bias.

**Use / project relevance:** Recover the true magnetic field; calibrating the magnetometer ourselves.

**Caveats:** BiasX median −69 µT observed — populated here (unlike accel biases); close to the uncal offset, consistent.

**Status:** OBSERVED (median −69) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1,223  *(approx_count_distinct)*
- **min:** -144.7
- **max:** 8.99
- **mean:** -73.32
- **std:** 15.98
- **p50:** -69.1
- **p95:** -55.31
- **p99:** -55.31
- **negatives:** 6,723,615 (100.0%)

## `BiasYMicroT`

**What:** Hard-iron bias for Y (µT).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Check population.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1,170  *(approx_count_distinct)*
- **min:** 19.96
- **max:** 119.5
- **mean:** 44.27
- **std:** 23.46
- **p50:** 36.95
- **p95:** 88.67
- **p99:** 102.2

## `BiasZMicroT`

**What:** Hard-iron bias for Z (µT).

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Check population.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1,184  *(approx_count_distinct)*
- **min:** -152.3
- **max:** 217.5
- **mean:** -39.75
- **std:** 104.3
- **p50:** -86.98
- **p95:** 147.6
- **p99:** 148.2
- **negatives:** 5,089,829 (75.7%)

## `CalibrationAccuracy`

**What:** Magnetic-field sensor confidence code 0-3.

**Why it matters:** Trust level.

**Use / project relevance:** Filter; magnetometers are the least reliable consumer sensor near electronics.

**Caveats:** Check value distribution.

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

