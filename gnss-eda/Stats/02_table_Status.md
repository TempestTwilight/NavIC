# 02 — Record type `Status`

**What it is:** per-signal satellite status record (milestone-based, ~1 Hz batches).

**Volume [OBSERVED]:** 6,956,811 rows across **95 sessions** (files containing this type); time range 1,780,543,012,000 → 1,793,074,857,000 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Status.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `UnixTimeMillis`

**What:** UTC wall-clock timestamp (ms) of this Status record, as the app stamped it. Observed to be EMPTY on many early rows (e.g. the first rows of 0_down_facing.txt had it blank).

**Why it matters:** Anchors each satellite-status row to calendar time so it can be joined to Raw/Fix/sensors.

**Use / project relevance:** Join key to other tables (together with Svid/ConstellationType/SignalIndex). Project: time-of-day/session features are legitimate; beware that blank timestamps must be filled from the same-file neighbours before joining.

**Caveats:** Blank on some rows (structural: the logger leaves it empty when the event shares the previous timestamp); NOT row-unique; cross-device collisions expected.

**Status:** OBSERVED IN DATA (blank rows) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 8,652 (0.1%)
- **Distinct:** 47,671  *(approx_count_distinct)*
- **min:** 1,780,543,012,000
- **max:** 1,793,074,857,000
- **mean:** 1,791,215,913,400
- **std:** 4,062,159,775
- **p50:** 1,792,728,215,907
- **p95:** 1,793,072,099,560
- **p99:** 1,793,073,860,833

## `SignalCount`

**What:** Number of GNSS signals the receiver was tracking at that event (count of Status rows in the same batch).

**Why it matters:** Global measure of tracking load — a proxy for sky visibility and receiver capability.

**Use / project relevance:** Epoch-level feature, available in real time. Project: directly relevant — C/N0 depends on how many signals are tracked (front-end channel sharing / scheduling).

**Caveats:** Reference doc says 'typical 20-70'; observed medians are far higher (89) on multi-constellation phones — verify and re-baseline. Must equal per-batch row count — check in Phase D.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 53  *(exact)*
- **min:** 47
- **max:** 105
- **mean:** 82.79
- **std:** 15.37
- **p50:** 89
- **p95:** 101
- **p99:** 103

## `SignalIndex`

**What:** 0-based index of this signal within the batch (0 .. SignalCount-1).

**Why it matters:** Disambiguates rows of one batch; with batch timestamp it forms a row key.

**Use / project relevance:** Row key component; sanity check that indices are dense/unique per batch.

**Caveats:** Carries no signal meaning by itself — never use as a numeric feature.

**Status:** OBSERVED IN DATA (0-104) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 106  *(approx_count_distinct)*
- **min:** 0
- **max:** 104
- **mean:** 40.89
- **std:** 25.49
- **p50:** 39
- **p95:** 85
- **p99:** 93
- **zeros:** 87,511 (1.3%)

## `ConstellationType`

**What:** GNSS constellation code (1 GPS, 2 SBAS, 3 GLONASS, 4 QZSS, 5 BeiDou, 6 Galileo, 7 NavIC/IRNSS).

**Why it matters:** Identifies which system the satellite belongs to.

**Use / project relevance:** Categorical feature + grouping key. Project: C/N0 behaviour differs per constellation (different modulation/bands), so this is a key grouping variable.

**Caveats:** Same numeric value as in Raw — verify identical value sets across tables.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 7  *(exact)*
- **min:** 1
- **max:** 7
- **mean:** 4.119
- **std:** 1.937
- **p50:** 5
- **p95:** 6
- **p99:** 7

## `Svid`

**What:** Satellite ID within the constellation (see Raw.Svid).

**Why it matters:** Identifies the satellite.

**Use / project relevance:** Track persistence / grouping. Project: categorical feature; available at prediction time.

**Caveats:** Only unique together with ConstellationType.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 79  *(exact)*
- **min:** 1
- **max:** 199
- **mean:** 26.85
- **std:** 35.97
- **p50:** 21
- **p95:** 68
- **p99:** 195

## `CarrierFrequencyHz`

**What:** Carrier frequency of the signal in Hz (e.g. GPS L1 1,575,420,000).

**Why it matters:** Identifies the band (L1/L5/B1/E1…) within a constellation.

**Use / project relevance:** Signal identity key; categorical band feature. Project: band is a strong C/N0 predictor (L5 usually better than L1).

**Caveats:** GLONASS frequencies vary per channel; treat as categorical, not continuous.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 15  *(exact)*
- **min:** 1,176,450,000
- **max:** 1,605,375,000
- **mean:** 1,451,684,863
- **std:** 184,525,242
- **p50:** 1,561,098,000
- **p95:** 1,602,000,000
- **p99:** 1,604,099,112

## `Cn0DbHz`

**What:** Carrier-to-noise density of the received signal (dB-Hz), the primary signal-strength measure.

**Why it matters:** THE measurement of signal quality; the prediction target of the project.

**Use / project relevance:** Project: this is the Status-side copy of the C/N0 that Raw also reports — use for cross-checks and for building the target at 1 Hz.

**Caveats:** 0 values observed (18.2% of rows) — likely 'not reported' rather than real 0; must be handled as missing, not as a number. Doc range 0-63, typical 10-50.

**Status:** OBSERVED IN DATA (incl. zeros) + REFERENCE (range)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 519  *(approx_count_distinct)*
- **min:** 0
- **max:** 54.7
- **mean:** 25.88
- **std:** 14.1
- **p50:** 29.79
- **p95:** 43.32
- **p99:** 48.06
- **zeros:** 1,263,343 (18.2%)

## `AzimuthDegrees`

**What:** Satellite azimuth: horizontal direction from the phone to the satellite, degrees clockwise from true north (0-360).

**Why it matters:** Geometry of the satellite relative to the sky.

**Use / project relevance:** Sky-plot analyses; features (azimuth relative to phone yaw for orientation effects). Project: available at prediction time.

**Caveats:** Undefined when the satellite is not in view (check nulls).

**Status:** OBSERVED IN DATA (0-359.98) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 15,163  *(approx_count_distinct)*
- **min:** 0
- **max:** 360
- **mean:** 164.1
- **std:** 102.5
- **p50:** 154.4
- **p95:** 327.7
- **p99:** 352.3
- **zeros:** 95,602 (1.4%)

## `ElevationDegrees`

**What:** Satellite elevation above the local horizon in degrees (0 = horizon, 90 = zenith).

**Why it matters:** Main geometric driver of C/N0: lower elevation → longer atmospheric/obstruction path.

**Use / project relevance:** Feature + binning variable for Phase F elevation analysis.

**Caveats:** Observed min is −5.0 (below horizon) — the chipset can report slightly negative values when the satellite is below the horizon but still tracked; do not blindly clip.

**Status:** OBSERVED IN DATA (min −5) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 13,311  *(approx_count_distinct)*
- **min:** -5
- **max:** 87
- **mean:** 31.42
- **std:** 18.91
- **p50:** 30.26
- **p95:** 66.15
- **p99:** 77.45
- **zeros:** 88,531 (1.3%)
- **negatives:** 37 (0.0%)

## `UsedInFix`

**What:** Boolean: whether this satellite was used in the most recent position fix.

**Why it matters:** Marks the subset of satellites the chipset trusted for positioning.

**Use / project relevance:** Analysis variable (survivorship); could be a weak feature. Project: available at prediction time only if the last fix is known — treat carefully.

**Caveats:** Only meaningful at fix epochs; between fixes it is stale. Reference doc claims ~36% true — verify.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 2  *(exact)*
- **min:** 0
- **max:** 1
- **mean:** 0.6342
- **std:** 0.4816
- **p50:** 1
- **p95:** 1
- **p99:** 1
- **zeros:** 2,544,647 (36.6%)

## `HasAlmanacData`

**What:** Boolean: receiver has the satellite's almanac (coarse orbit info).

**Why it matters:** Indicates how much the receiver knows about the satellite.

**Use / project relevance:** Data-quality descriptor; weak feature.

**Caveats:** 74% false observed — mostly the almanac is not broadcast/kept.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 2  *(exact)*
- **min:** 0
- **max:** 1
- **mean:** 0.253
- **std:** 0.4348
- **p50:** 0
- **p95:** 1
- **p99:** 1
- **zeros:** 5,196,401 (74.7%)

## `HasEphemerisData`

**What:** Boolean: receiver has the satellite's ephemeris (precise orbit) — required to use the satellite in the position solution.

**Why it matters:** Explains why some strong signals are not UsedInFix.

**Use / project relevance:** Feature / analysis variable. Project: available in real time; correlates with tracking age.

**Caveats:** 58% false observed; ephemeris validity has a ~2-4 h lifetime — expect time structure within a session.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 2  *(exact)*
- **min:** 0
- **max:** 1
- **mean:** 0.4235
- **std:** 0.4941
- **p50:** 0
- **p95:** 1
- **p99:** 1
- **zeros:** 4,010,767 (57.7%)

## `BasebandCn0DbHz`

**What:** C/N0 measured in the baseband (post-correlation), typically a few dB below Raw.Cn0DbHz (antenna port).

**Why it matters:** Second, independent signal-quality estimate from inside the chipset.

**Use / project relevance:** Cross-check of Raw Cn0; possibly a cleaner target or feature. Project: real-time available.

**Caveats:** Zeros count exactly equals Cn0DbHz zeros in first pass — the two columns share a reporting gap; verify missingness overlap.

**Status:** OBSERVED IN DATA + REFERENCE (typical offset)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 820  *(approx_count_distinct)*
- **min:** 0
- **max:** 49.7
- **mean:** 23.74
- **std:** 13.05
- **p50:** 27.32
- **p95:** 40.15
- **p99:** 43.61
- **zeros:** 1,263,343 (18.2%)

