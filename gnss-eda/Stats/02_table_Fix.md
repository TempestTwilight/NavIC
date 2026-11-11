# 02 — Record type `Fix`

**What it is:** position fix record (Android Location objects from FLP/GPS/NLP).

**Volume [OBSERVED]:** 189,986 rows across **95 sessions** (files containing this type); time range 1,780,543,010,645 → 1,793,074,858,759 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Fix.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `Provider`

**What:** Which location provider produced the fix: 'FLP' (Fused Location Provider), 'GPS' (GNSS engine), 'NLP' (Network Location Provider).

**Why it matters:** A fix's origin determines its nature — GPS fixes are satellite-derived, NLP fixes can be WiFi/cell-tower derived.

**Use / project relevance:** Grouping/filtering variable. Project: CRITICAL — only GPS-provider fixes are true GNSS fixes; NLP rows can leak non-GNSS information (WiFi databases) and must be excluded from GNSS analyses. Counts observed: FLP 93,229 / GPS 86,539 / NLP 8,311.

**Caveats:** FLP fuses sensors+GNSS+network; its coordinates are smoothed, not raw GNSS.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `VARCHAR`; **nulls:** 0 (0.0%)
- **Distinct:** 3  *(exact)*
- **Top values:** `FLP` = 94,176, `GPS` = 87,402, `NLP` = 8,408

## `LatitudeDegrees`

**What:** Fix latitude in degrees (WGS-84).

**Why it matters:** Position of the phone.

**Use / project relevance:** Geographic context; session-site identification. Project: all values concentrate around 18.4603 N / 73.8837 E (Pune campus) — no wide variation, so lat/lon are near-constant features.

**Caveats:** FLP/NLP smoothing makes jitter understated; compare providers before trusting accuracy.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 13,005  *(approx_count_distinct)*
- **min:** 18.46
- **max:** 18.46
- **mean:** 18.46
- **std:** 0.000192
- **p50:** 18.46
- **p95:** 18.46
- **p99:** 18.46

## `LongitudeDegrees`

**What:** Fix longitude in degrees (WGS-84).

**Why it matters:** Position of the phone.

**Use / project relevance:** Same as LatitudeDegrees.

**Caveats:** Same as LatitudeDegrees.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 10,708  *(approx_count_distinct)*
- **min:** 73.88
- **max:** 73.88
- **mean:** 73.88
- **std:** 0.0001154
- **p50:** 73.88
- **p95:** 73.88
- **p99:** 73.88

## `AltitudeMeters`

**What:** Fix altitude in metres (WGS-84 ellipsoidal height, as reported by Android's Location.getAltitude()).

**Why it matters:** Height component of the fix.

**Use / project relevance:** Site context. Project: observed 569-632 m around Pune — variation is mostly noise, not real movement.

**Caveats:** Vertical error is usually 2-3× horizontal; NLP altitudes are coarse.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,334 (0.7%)
- **Distinct:** 16,046  *(approx_count_distinct)*
- **min:** 569
- **max:** 631.7
- **mean:** 594.6
- **std:** 8.718
- **p50:** 594
- **p95:** 607
- **p99:** 619.2

## `SpeedMps`

**What:** Horizontal speed from the fix (m/s).

**Why it matters:** Motion state of the phone.

**Use / project relevance:** Feature (walking vs static changes multipath dynamics). Project: available at prediction time; observed max 3.75 m/s — pedestrian speeds only.

**Caveats:** 0 may mean 'static' or 'not computed'; check provider differences.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 8,515 (4.5%)
- **Distinct:** 92,711  *(approx_count_distinct)*
- **min:** 0
- **max:** 3.75
- **mean:** 0.03863
- **std:** 0.08711
- **p50:** 0.02229
- **p95:** 0.1034
- **p99:** 0.2606
- **zeros:** 88,835 (46.8%)

## `AccuracyMeters`

**What:** 68% horizontal accuracy radius of the fix (metres).

**Why it matters:** Standard Android quality metric of the fix.

**Use / project relevance:** Quality filter / feature. Project: a per-epoch GNSS-quality signal available in real time — useful context for C/N0.

**Caveats:** Provider-dependent meaning; NLP accuracy is network-quality, not GNSS.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 9,752  *(approx_count_distinct)*
- **min:** 0.5
- **max:** 175.8
- **mean:** 5.661
- **std:** 13.9
- **p50:** 3
- **p95:** 10.76
- **p99:** 91.59

## `BearingDegrees`

**What:** Direction of travel of the device in degrees east of true north (0-360).

**Why it matters:** Motion direction.

**Use / project relevance:** Only meaningful while moving. Project: null when static (71.8% null observed); do not impute blindly.

**Caveats:** 71.8% null confirms the doc claim; mostly-static sessions explain it.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 136,411 (71.8%)
- **Distinct:** 4,934  *(approx_count_distinct)*
- **min:** 0
- **max:** 359.9
- **mean:** 174.1
- **std:** 99.04
- **p50:** 149.2
- **p95:** 333.4
- **p99:** 355.3
- **zeros:** 111 (0.1%)

## `UnixTimeMillis`

**What:** UTC timestamp of the fix (ms).

**Why it matters:** Time axis of the fix stream.

**Use / project relevance:** As-of joins of fixes to Raw/sensor epochs; train/test split boundaries.

**Caveats:** Fixes are irregular (~2.7 Hz across providers); never assume regular spacing.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 147,739  *(approx_count_distinct)*
- **min:** 1,780,543,010,645
- **max:** 1,793,074,858,759
- **mean:** 1,791,461,706,379
- **std:** 3,773,554,685
- **p50:** 1,792,738,115,586
- **p95:** 1,793,072,213,852
- **p99:** 1,793,073,999,378

## `SpeedAccuracyMps`

**What:** 1-sigma accuracy of SpeedMps (m/s).

**Why it matters:** Quality of the speed estimate.

**Use / project relevance:** Weight/filter for speed features.

**Caveats:** Reference doc claims 100% null — CONFIRMED in the profile; column is dead weight.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 8,515 (4.5%)
- **Distinct:** 93,737  *(approx_count_distinct)*
- **min:** 0
- **max:** 11.5
- **mean:** 0.2316
- **std:** 0.3308
- **p50:** 0.09787
- **p95:** 0.9003
- **p99:** 1.36
- **zeros:** 4 (0.0%)

## `BearingAccuracyDegrees`

**What:** 1-sigma accuracy of BearingDegrees (degrees).

**Why it matters:** Quality of the bearing estimate.

**Use / project relevance:** Weight/filter for bearing features.

**Caveats:** Expected mostly null when bearing is null; verify in profile.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 136,411 (71.8%)
- **Distinct:** 5,302  *(approx_count_distinct)*
- **min:** 3
- **max:** 180
- **mean:** 164.2
- **std:** 35.03
- **p50:** 179.9
- **p95:** 180
- **p99:** 180

## `elapsedRealtimeNanos`

**What:** Time since boot of the fix (ns) — the monotonic clock of Android.

**Why it matters:** The stable clock for cross-table alignment (wall clock can jump).

**Use / project relevance:** THE join/alignment clock to Raw and sensors (which also stamp elapsedRealtimeNanos). Project: use for all as-of joins; also fixes the 'which came first' ordering.

**Caveats:** Not comparable across devices (reboots reset it); always pair with session id.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 165,025  *(approx_count_distinct)*
- **min:** 138,325,408,851
- **max:** 1,898,565,423,587,145
- **mean:** 570,444,919,572,767
- **std:** 449,607,734,433,884
- **p50:** 420,231,083,222,016
- **p95:** 1,774,284,685,843,809
- **p99:** 1,897,462,743,952,188

## `VerticalAccuracyMeters`

**What:** 1-sigma vertical accuracy of the altitude (metres).

**Why it matters:** Quality of the height estimate.

**Use / project relevance:** Quality filter.

**Caveats:** Null when the provider does not compute it; check null rate.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,334 (0.7%)
- **Distinct:** 58,002  *(approx_count_distinct)*
- **min:** 1
- **max:** 107.3
- **mean:** 8.761
- **std:** 9.593
- **p50:** 6.268
- **p95:** 21.2
- **p99:** 53.68

## `MockLocation`

**What:** Boolean: whether this fix came from a mock-location app.

**Why it matters:** Data-integrity flag — mock fixes would poison the analysis.

**Use / project relevance:** Sanity check. Project: expected constant 0 everywhere (no mock apps used); if truly constant it is dead weight but its constancy is itself a finding.

**Caveats:** Constant 0 observed.

**Status:** OBSERVED IN DATA (constant 0) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 189,986 (100.0%)

## `NumberOfUsedSignals`

**What:** Number of GNSS signals used in the fix (Android 12+ field).

**Why it matters:** Would link the fix to the Status rows flagged UsedInFix.

**Use / project relevance:** Cross-check UsedInFix counts. Project: potential epoch feature IF populated.

**Caveats:** Reference doc claims 100% null — CONFIRMED; column is dead weight.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 189,986 (100.0%)
- **Distinct:** 0  *(exact)*

## `VerticalSpeedAccuracyMps`

**What:** 1-sigma accuracy of vertical speed (m/s).

**Why it matters:** Quality of the vertical speed estimate.

**Use / project relevance:** Dead weight if null.

**Caveats:** Reference doc claims 100% null — CONFIRMED.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 189,986 (100.0%)
- **Distinct:** 0  *(exact)*

## `SolutionType`

**What:** Code describing the fix computation method (e.g. PPP-related solution classes on some chipsets).

**Why it matters:** Would describe the quality class of the solution.

**Use / project relevance:** Potentially useful categorical if ever populated.

**Caveats:** Reference doc claims 100% null — CONFIRMED.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `VARCHAR`; **nulls:** 189,986 (100.0%)
- **Distinct:** 0  *(exact)*

## `ExtraKeyValuePairs`

**What:** Pipe-separated key=value pairs with provider-specific extras (e.g. 'maxCn0=41|meanCn0=28|satellites=37').

**Why it matters:** Carries chipset-specific summary values not in the schema.

**Use / project relevance:** Parsing candidates: maxCn0/meanCn0 are useful epoch-level C/N0 summaries. Project: meanCn0 is a per-epoch aggregate of the target itself — usable as a context feature at prediction time, but never use FUTURE values (leakage).

**Caveats:** 54.4% null observed (mostly when the provider has no extras); the key set may differ per provider — parse defensively.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `VARCHAR`; **nulls:** 102,584 (54.0%)
- **Distinct:** 3,487  *(approx_count_distinct)*
- **Top values:** `maxCn0=49|meanCn0=40|satellites=31` = 743, `maxCn0=39|meanCn0=32|satellites=31` = 693, `maxCn0=51|meanCn0=40|satellites=31` = 669, `maxCn0=50|meanCn0=40|satellites=30` = 594, `maxCn0=50|meanCn0=40|satellites=31` = 583, `maxCn0=44|meanCn0=31|satellites=42` = 563, `dr=-1|eb=-1.0|maxCn0=43|meanCn0=32|satellites=40|type=0` = 561, `maxCn0=53|meanCn0=41|satellites=31` = 556

