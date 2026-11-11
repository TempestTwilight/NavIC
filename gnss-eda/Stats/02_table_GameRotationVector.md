# 02 — Record type `GameRotationVector`

**What it is:** TYPE_GAME_ROTATION_VECTOR quaternion (accel+gyro fusion, ~125 Hz).

**Volume [OBSERVED]:** 11,550,067 rows across **89 sessions** (files containing this type); time range 1,780,543,010,903 → 1,793,074,859,024 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/GameRotationVector.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms) of the quaternion sample.

**Why it matters:** Calendar-time anchor.

**Use / project relevance:** Joins by nearest time.

**Caveats:** Same as other sensor rows.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 11,109,766  *(approx_count_distinct)*
- **min:** 1,780,543,010,903
- **max:** 1,793,074,859,024
- **mean:** 1,790,718,532,441
- **std:** 4,564,341,530
- **p50:** 1,792,751,313,678
- **p95:** 1,793,071,807,556
- **p99:** 1,793,073,939,249

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns).

**Why it matters:** Stable alignment clock.

**Use / project relevance:** As-of joins; ~125 Hz sampling makes it possible to interpolate orientation to any GNSS epoch.

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 11,564,369  *(approx_count_distinct)*
- **min:** 139,250,540,323
- **max:** 1,898,566,088,609,399
- **mean:** 709,415,170,183,387
- **std:** 500,056,033,057,015
- **p50:** 604,037,185,866,487
- **p95:** 1,838,140,891,983,365
- **p99:** 1,897,963,566,181,072

## `x`

**What:** Quaternion component x (rotation vector axis · sin(θ/2)).

**Why it matters:** With y,z,w encodes the device orientation relative to Earth as a quaternion (game rotation vector = gyro+accel ONLY, no magnetometer → yaw drifts relative to true north).

**Use / project relevance:** Phase G: convert to Euler angles and compare with OrientationDeg; derive the orientation grid without Euler singularity issues.

**Caveats:** Non-normalised per-row? Verify norm ≈ 1; values are relative to a gravity-based frame — yaw is NOT absolute north.

**Status:** OBSERVED IN DATA + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 6,685,046  *(approx_count_distinct)*
- **min:** -0.9999
- **max:** 0.9998
- **mean:** 0.208
- **std:** 0.4241
- **p50:** 0.2497
- **p95:** 0.8521
- **p99:** 0.9199
- **negatives:** 2,436,194 (21.1%)

## `y`

**What:** Quaternion component y.

**Why it matters:** Same as x.

**Use / project relevance:** Same as x.

**Caveats:** Same as x.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 6,923,633  *(approx_count_distinct)*
- **min:** -0.9994
- **max:** 0.9953
- **mean:** -0.01477
- **std:** 0.482
- **p50:** -0.001398
- **p95:** 0.8303
- **p99:** 0.9072
- **negatives:** 6,030,772 (52.2%)

## `z`

**What:** Quaternion component z.

**Why it matters:** Same as x.

**Use / project relevance:** Same as x.

**Caveats:** Same as x.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 6,125,248  *(approx_count_distinct)*
- **min:** -0.9999
- **max:** 1
- **mean:** 0.04957
- **std:** 0.5243
- **p50:** 0.0005516
- **p95:** 0.8772
- **p99:** 0.96
- **zeros:** 2 (0.0%)
- **negatives:** 5,696,442 (49.3%)

## `w`

**What:** Quaternion scalar component w = cos(θ/2).

**Why it matters:** Completes the quaternion; Android reports it LAST (x,y,z,w order in the header).

**Use / project relevance:** Same as x; also canonical-sign check: Android rotation vectors keep w ≥ 0 (observed — unlike FOP where w goes negative).

**Caveats:** Order in the LOGGER HEADER is x,y,z,w — the reference doc's listing must be checked against this.

**Status:** OBSERVED (w ≥ 0 always) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 6,341,280  *(approx_count_distinct)*
- **min:** 1.015e-05
- **max:** 1
- **mean:** 0.42
- **std:** 0.3011
- **p50:** 0.3546
- **p95:** 0.9153
- **p99:** 0.9822

