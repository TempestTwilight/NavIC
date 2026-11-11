# 02 — Record type `FOP`

**What it is:** fused orientation provider quaternion + magnetic heading (~125 Hz).

**Volume [OBSERVED]:** 7,337,813 rows across **75 sessions** (files containing this type); time range 1,792,380,029,896 → 1,793,074,859,012 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/FOP.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms).

**Why it matters:** Calendar anchor.

**Use / project relevance:** Joins.

**Caveats:** Standard.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6,958,866  *(approx_count_distinct)*
- **min:** 1,792,380,029,896
- **max:** 1,793,074,859,012
- **mean:** 1,792,806,199,273
- **std:** 187,644,819
- **p50:** 1,792,755,358,770
- **p95:** 1,793,072,771,550
- **p99:** 1,793,074,271,645

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns).

**Why it matters:** Alignment clock.

**Use / project relevance:** As-of joins at ~125 Hz.

**Caveats:** Pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 7,012,430  *(approx_count_distinct)*
- **min:** 139,230,770,948
- **max:** 905,685,658,147,606
- **mean:** 477,709,124,246,197
- **std:** 234,583,883,239,624
- **p50:** 557,816,298,855,179
- **p95:** 901,773,654,565,783
- **p99:** 904,860,251,692,533

## `x`

**What:** Quaternion component x of the FUSED orientation provider (accel+gyro+mag → absolute-ish heading).

**Why it matters:** Full quaternion of the fused sensor.

**Use / project relevance:** Phase G comparison: FOP vs GameRotationVector vs OrientationDeg.

**Caveats:** w is NOT canonicalised here (14% of rows negative) — normalise sign before comparing quaternions.

**Status:** OBSERVED (w negative 14%) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 5,023,156  *(approx_count_distinct)*
- **min:** -0.9928
- **max:** 0.9989
- **mean:** 0.28
- **std:** 0.379
- **p50:** 0.2414
- **p95:** 0.9116
- **p99:** 0.9976
- **negatives:** 1,202,772 (16.4%)

## `y`

**What:** Quaternion component y.

**Why it matters:** Same as x.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 4,990,422  *(approx_count_distinct)*
- **min:** -0.9995
- **max:** 0.9994
- **mean:** 0.1104
- **std:** 0.4504
- **p50:** 0.003722
- **p95:** 0.9161
- **p99:** 0.9649
- **negatives:** 3,520,300 (48.0%)

## `z`

**What:** Quaternion component z.

**Why it matters:** Same.

**Use / project relevance:** Same.

**Caveats:** Same.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 5,016,932  *(approx_count_distinct)*
- **min:** -0.9989
- **max:** 0.9997
- **mean:** 0.1708
- **std:** 0.5151
- **p50:** 0.007228
- **p95:** 0.9458
- **p99:** 0.9873
- **negatives:** 3,551,018 (48.4%)

## `w`

**What:** Quaternion scalar w.

**Why it matters:** Completes quaternion.

**Use / project relevance:** Sign canonicalisation needed (see x).

**Caveats:** Negative in 14% of rows — q and −q are the same rotation; must be canonicalised before element-wise comparisons.

**Status:** OBSERVED (14% negative) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 5,162,824  *(approx_count_distinct)*
- **min:** -0.9995
- **max:** 1
- **mean:** 0.3326
- **std:** 0.3972
- **p50:** 0.2933
- **p95:** 0.9399
- **p99:** 0.9991
- **negatives:** 1,037,880 (14.1%)

## `headingDegrees`

**What:** Fused heading in degrees (0-360) — the magnetic-azimuth of the device from the fusion (TYPE_GAME_ROTATION_VECTOR + geomagnetic correction).

**Why it matters:** The absolute-yaw equivalent of the fused provider.

**Use / project relevance:** Project: candidate 'recommended yaw' variable; compare against OrientationDeg.yawDeg for consistency.

**Caveats:** 0-360 wraparound at 0/360; missing when magnetometer is unreliable (FOP absent in the June orientation files!).

**Status:** OBSERVED (0-360) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 3,987,897  *(approx_count_distinct)*
- **min:** 0.0002976
- **max:** 360
- **mean:** 148.7
- **std:** 102.1
- **p50:** 188.4
- **p95:** 290.9
- **p99:** 356.4

## `headingErrorDegrees`

**What:** Estimated error of headingDegrees in degrees.

**Why it matters:** Quality of the heading.

**Use / project relevance:** Weight/filter; quantised (304 distinct values, min 20.9°) — treat as categorical-ish.

**Caveats:** Large minimum error (~21°) suggests coarse quantisation; not a smooth quality signal.

**Status:** OBSERVED (quantised, min 20.9°) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 304  *(approx_count_distinct)*
- **min:** 20.93
- **max:** 180
- **mean:** 36.52
- **std:** 21.12
- **p50:** 29.49
- **p95:** 100.8
- **p99:** 112.9

## `conservativeHeadingErrorDegrees`

**What:** A deliberately pessimistic version of headingErrorDegrees (vendor-defined).

**Why it matters:** Safety margin for consumers of the heading.

**Use / project relevance:** Alternative quality bound; same usage as headingErrorDegrees.

**Caveats:** Vendor-defined semantics — do not interpret precisely; compare with headingErrorDegrees (should be ≥).

**Status:** OBSERVED IN DATA + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 276  *(approx_count_distinct)*
- **min:** 20.93
- **max:** 180
- **mean:** 42.46
- **std:** 36.62
- **p50:** 29.49
- **p95:** 133.4
- **p99:** 180

