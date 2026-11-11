# 02 — Record type `OrientationDeg`

**What it is:** Euler-angle orientation record (yaw/roll/pitch, ~125 Hz).

**Volume [OBSERVED]:** 11,550,170 rows across **89 sessions** (files containing this type); time range 1,780,543,010,893 → 1,793,074,859,024 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/OrientationDeg.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms) of the orientation sample.

**Why it matters:** Calendar-time anchor.

**Use / project relevance:** Joins to fix/GNSS epochs by nearest time.

**Caveats:** Wall clock — verify monotonicity; prefer elapsedRealtimeNanos for ordering.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 11,109,766  *(approx_count_distinct)*
- **min:** 1,780,543,010,893
- **max:** 1,793,074,859,024
- **mean:** 1,790,718,533,405
- **std:** 4,564,341,027
- **p50:** 1,792,751,374,979
- **p95:** 1,793,071,803,160
- **p99:** 1,793,073,929,918

## `elapsedRealtimeNanos`

**What:** Monotonic time since boot (ns) of the orientation sample.

**Why it matters:** Stable clock for alignment with Raw/Status/Fix.

**Use / project relevance:** THE alignment clock for Phase G (orientation at epoch time).

**Caveats:** Per-device only; always pair with session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 11,564,369  *(approx_count_distinct)*
- **min:** 139,240,617,041
- **max:** 1,898,566,088,609,399
- **mean:** 709,414,402,354,078
- **std:** 500,056,405,997,724
- **p50:** 604,214,480,630,234
- **p95:** 1,838,174,269,201,169
- **p99:** 1,897,911,229,662,102

## `yawDeg`

**What:** Yaw (azimuth) angle of the phone in degrees: rotation about the vertical axis; 0 = magnetic/east-of-north convention per the TYPE_ROTATION_VECTOR→getOrientation mapping.

**Why it matters:** First angle in the record; THE candidate output of the project (recommend a yaw).

**Use / project relevance:** Phase G: map yaw to the orientation grid (face_up/back, N/E/S/W, landscape) and study C/N0 vs yaw.

**Caveats:** Observed range 0-359 integer-like (quantised to 1°); convention must be verified against GameRotationVector quaternions (Q-006/G); magnetic vs true north distinction depends on declination (~0.5° in Pune — negligible).

**Status:** OBSERVED (0-359, integer-like) + BACKGROUND (convention)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 356  *(approx_count_distinct)*
- **min:** 0
- **max:** 359
- **mean:** 155.5
- **std:** 94.54
- **p50:** 185.9
- **p95:** 286.9
- **p99:** 355.1
- **zeros:** 7,935 (0.1%)

## `rollDeg`

**What:** Roll angle in degrees (rotation about the phone's long axis). SECOND angle in this record per the logger header.

**Why it matters:** Distinguishes phone tilt.

**Use / project relevance:** Orientation-grid construction (flat vs vertical).

**Caveats:** Observed −179..179; the header order (yaw,roll,pitch) CONTRADICTS docs/Data_Obs.md (yaw,pitch,roll) — the pitch/roll ranges settle it: values like ±97 are impossible for pitch on a phone (−90..90) but fine for roll, so header order is right and the doc is wrong (Q-006).

**Status:** OBSERVED (range proves order) + REFERENCE (doc CONTRADICTED)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 353  *(approx_count_distinct)*
- **min:** -179
- **max:** 179
- **mean:** -12.59
- **std:** 110.5
- **p50:** 0
- **p95:** 179
- **p99:** 179
- **zeros:** 2,253,663 (19.5%)
- **negatives:** 5,072,641 (43.9%)

## `pitchDeg`

**What:** Pitch angle in degrees (rotation about the phone's short axis; −90 = camera down... per Android convention). THIRD angle in this record.

**Why it matters:** Completes the Euler triple.

**Use / project relevance:** Orientation-grid construction.

**Caveats:** Observed −89..89 — consistent with pitch; see rollDeg note.

**Status:** OBSERVED (−89..89) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 180  *(approx_count_distinct)*
- **min:** -89
- **max:** 89
- **mean:** -31.31
- **std:** 31.27
- **p50:** -38.37
- **p95:** 1.368
- **p99:** 3
- **zeros:** 2,273,696 (19.7%)
- **negatives:** 7,729,518 (66.9%)

## `CalibrationAccuracy`

**What:** Sensor-manager confidence code: 0 UNRELIABLE, 1 LOW, 2 MEDIUM, 3 HIGH.

**Why it matters:** Tells how much to trust the orientation values.

**Use / project relevance:** Quality filter for orientation features.

**Caveats:** Constant 3 observed — always 'high'; dead weight as a feature but confirms the vendor fusion was confident.

**Status:** OBSERVED (constant 3) + REFERENCE (code meaning)

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

