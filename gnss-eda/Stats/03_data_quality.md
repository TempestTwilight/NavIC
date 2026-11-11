# 03 — Data Quality and Integrity (Phase D)

This document provides a systematic data quality assessment across all 12 tables in the GNSS dataset (95 recording sessions, 5,482,734 `Raw` rows, 6,956,811 `Status` rows, 189,986 `Fix` rows, and ~49.9M inertial/orientation sensor rows).

The primary goal of this phase is to evaluate missingness, uninformative constants, duplicate records, numerical outliers, and timing fidelity to ensure robust model training and feature extraction for predicting carrier-to-noise density ratio ($C/N_0$) 15 seconds ahead.

All statistics are computed directly from the typed Parquet caches in `scratch/typed/` via DuckDB 2.0 under strict resource constraints (`SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;`). Queries are preserved in `docs/EDA/queries/011_*.sql` through `016_*.sql`.

---

## 1. NULL Patterns and Missingness Analysis

### 1.1 Structural vs. Random Missingness Overview
Across the 12 tables, missing data falls cleanly into two distinct categories:
1. **Structural Missingness (By Design / API / Vendor Tier)**: Fields intentionally left blank or unpopulated because of device hardware capabilities, chipset vendor firmware choices, Android API level differences, or vehicle/stationary test states. These are not data loss incidents and should not be imputed with random or continuous estimators.
2. **Random Missingness (Data Loss / Sensor Dropouts)**: Intermittent transmission drops or unacquired signals. In this corpus, random missingness is remarkably low ($< 0.15\%$).

### 1.2 Table-by-Table Missingness Summary

| Table Name | Total Rows | Columns with Non-Zero Nulls | Null Count | Null % | Nature of Missingness |
|---|---:|---|---:|---:|---|
| **Raw** | 5,482,734 | `CarrierCycles` | 5,482,734 | 100.0% | Structural (Deprecated Android API field) |
| | | `CarrierPhase` | 5,482,734 | 100.0% | Structural (Deprecated Android API field) |
| | | `CarrierPhaseUncertainty` | 5,482,734 | 100.0% | Structural (Deprecated Android API field) |
| | | `IsFullTracking` | 3,215,027 | 58.64% | Structural (Reported only on CPH2401) |
| | | `SvPositionEcef*` (X,Y,Z) | 3,117,297 | 56.86% | Structural / Conditional (Device + satellite ephemeris availability) |
| | | `ChipsetElapsedRealtimeNanos` | 2,595,460 | 47.34% | Structural (Only reported by Qualcomm CPH2487 & Samsung SM-M146B) |
| | | `Klobuchar*` (Alpha0-3, Beta0-3) | 2,267,707 | 41.36% | Structural (CPH2401 MTK firmware omits iono broadcast coefficients) |
| | | `TimeUncertaintyNanos` | 1,555,526 | 28.37% | Structural (Samsung SM-M146B does not export clock uncertainty) |
| | | `DriftNanosPerSecond` | 1,555,526 | 28.37% | Structural (Samsung SM-M146B does not export clock drift) |
| | | `DriftUncertaintyNanosPerSecond` | 1,555,526 | 28.37% | Structural (Samsung SM-M146B does not export clock drift uncertainty) |
| | | `SnrInDb` | 1,555,526 | 28.37% | Structural (Samsung SM-M146B omits SNR; superseded by $C/N_0$) |
| | | `SatelliteInterSignalBias*` | 1,331,748 | 24.29% | Structural (Qualcomm CPH2487 never populates satellite ISB) |
| **Status** | 6,956,811 | `UnixTimeMillis` | 8,652 | 0.12% | Random / Initialization quirk (First epoch in some sessions) |
| | | *All other 12 columns* | 0 | 0.00% | Zero missingness |
| **Fix** | 189,986 | `NumberOfUsedSignals` | 189,986 | 100.0% | Structural (Field unpopulated in Android Location) |
| | | `VerticalSpeedAccuracyMps` | 189,986 | 100.0% | Structural (Field unpopulated in Android Location) |
| | | `SolutionType` | 189,986 | 100.0% | Structural (Field unpopulated in Android Location) |
| | | `BearingDegrees` | 136,411 | 71.80% | Structural (Static / walking device has no meaningful velocity bearing) |
| | | `BearingAccuracyDegrees` | 136,411 | 71.80% | Structural (Coupled to BearingDegrees nullity) |
| | | `ExtraKeyValuePairs` | 102,584 | 54.00% | Structural (Present on Samsung SM-M146B; absent on others) |
| | | `SpeedMps` | 8,515 | 4.48% | Structural (Device stationary; speed uncomputed) |
| | | `SpeedAccuracyMps` | 8,515 | 4.48% | Structural (Coupled to SpeedMps) |
| | | `AltitudeMeters` | 1,334 | 0.70% | Random (2D fix prior to 3D altitude resolution) |
| | | `VerticalAccuracyMeters` | 1,334 | 0.70% | Random (Coupled to AltitudeMeters) |
| **Agc** | 1,555,526 | `TimeUncertaintyNanos` | 1,555,526 | 100.0% | Structural (Samsung SM-M146B clock block does not populate) |
| | | `DriftNanosPerSecond` | 1,555,526 | 100.0% | Structural (Samsung SM-M146B clock block does not populate) |
| | | `DriftUncertaintyNanosPerSecond` | 1,555,526 | 100.0% | Structural (Samsung SM-M146B clock block does not populate) |
| | | `utcTimeMillis` | 55,217 | 3.55% | Structural / Parser artifact in Samsung clock header lines |
| **Nav** | 783,399 | *All columns* | 0 | 0.00% | Complete (Payload bytes stored as string) |
| **Accel** | 6,369,268 | *All columns* | 0 | 0.00% | Complete |
| **UncalAccel** | 8,365,329 | *All columns* | 0 | 0.00% | Complete |
| **UncalGyro** | 16,661,992 | *All columns* | 0 | 0.00% | Complete |
| **UncalMag** | 6,723,621 | *All columns* | 0 | 0.00% | Complete |
| **OrientationDeg**| 11,550,170 | *All columns* | 0 | 0.00% | Complete |
| **GameRotationVector**| 11,550,067 | *All columns* | 0 | 0.00% | Complete |
| **FOP** | 7,337,813 | *All columns* | 0 | 0.00% | Complete |

### 1.3 Per-Device Null Breakdown in `Raw`
- **Goal**: Characterize device-specific reporting behaviors in `Raw.parquet`.
- **Why**: Features must not be engineered on columns that are completely absent for entire device families.
- **SQL / Source**: `scratch/profiles/raw_null_by_device.csv`
- **Result** [OBSERVED]:

| Device Model | Chipset Vendor | Sessions | Raw Rows | `TimeUncertainty` | `Drift*` | `SnrInDb` | `Klobuchar*` | `SvPosition*` | `SatISB*` | `ChipsetElapsed` | `IsFullTracking` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CPH2401** | MediaTek Dimensity 9000 | 36 | 2,267,707 | 0.0% | 0.0% | 0.0% | **100.0%** | **100.0%** | 0.0% | **100.0%** | 0.0% (const 0) |
| **SM-M146B** | Samsung Exynos 1330 | 29 | 1,555,526 | **100.0%** | **100.0%** | **100.0%** | 0.0% | 46.8% | 0.0% | 0.0% | **100.0%** |
| **CPH2487** | Qualcomm Snapdragon 782G| 22 | 1,331,748 | 0.0% | 0.0% | 0.0% | 0.0% | 6.7% | **100.0%** | 0.0% | **100.0%** |
| **CPH2325** | MediaTek Dimensity 900 | 6 | 327,753 | 0.0% | 0.0% | 0.0% | 0.0% | 10.0% | 0.0% | **100.0%** | **100.0%** |

- **Insight**:
  - `IsFullTracking` is only populated by OnePlus CPH2401 (where it is constantly `0`). The remaining devices report `NULL`.
  - `Klobuchar` ionospheric coefficients are completely missing on CPH2401, but available on Samsung and Qualcomm.
  - Qualcomm CPH2487 never populates `SatelliteInterSignalBiasNanos`.
  - Samsung SM-M146B completely suppresses clock variance estimates (`TimeUncertaintyNanos`, `DriftNanosPerSecond`, `DriftUncertaintyNanosPerSecond`).
- **ML Relevance**:
  - **Feature Pruning**: `CarrierCycles`, `CarrierPhase`, `CarrierPhaseUncertainty`, `IsFullTracking`, `NumberOfUsedSignals`, and `SolutionType` must be dropped unconditionally.
  - **Device Grouping**: Tree models trained across all devices must not rely on `Klobuchar*`, `SvPositionEcef*`, or `TimeUncertaintyNanos` as mandatory split features, or must include explicit missingness indicators.

### 1.4 Missingness by Constellation in `Raw`
- **Goal**: Evaluate whether satellite positioning or carrier properties are constellation-dependent.
- **SQL**: `docs/EDA/queries/011_null_patterns_all_tables.sql`
- **Result** [OBSERVED]:

| Constellation ID | Constellation Name | Total Rows | Null `Cn0DbHz` | Null `SvPositionEcef*` | % Null `SvPosition` |
|---|---|---:|---:|---:|---:|
| **1** | GPS | 1,320,562 | 0 | 709,615 | 53.74% |
| **3** | GLONASS | 595,419 | 0 | 310,187 | 52.09% |
| **4** | QZSS | 125,076 | 0 | 125,076 | **100.00%** |
| **5** | BeiDou | 2,036,887 | 0 | 881,260 | 43.26% |
| **6** | Galileo | 1,219,218 | 0 | 905,587 | 74.28% |
| **7** | NavIC (IRNSS) | 185,572 | 0 | 185,572 | **100.00%** |

- **Insight**:
  - `Cn0DbHz` has **0 nulls across all constellations** (the non-tracked channels are populated with numeric `0.0`, discussed in §4).
  - Android GNSS chipset hal never computes `SvPositionEcef*` for QZSS (4) or NavIC (7) across any session (100% null).

---

## 2. Constant and Near-Constant Columns

- **Goal**: Identify features with zero or near-zero variance ($N_{\text{distinct}} \le 2$ or $\sigma \approx 0$).
- **Why**: Constant columns contribute zero predictive entropy, waste memory, and can cause singular covariance matrices in linear baselines.
- **SQL**: `docs/EDA/queries/012_constant_columns_scan.sql`
- **Result** [OBSERVED]:

### 2.1 Table `Raw` Constant Columns

| Column Name | Exact Distinct Count | Min Value | Max Value | StdDev | Explanation / Status |
|---|---:|---:|---:|---:|---|
| `CarrierCycles` | 0 | `NULL` | `NULL` | — | 100% NULL (Deprecated API) |
| `CarrierPhase` | 0 | `NULL` | `NULL` | — | 100% NULL (Deprecated API) |
| `CarrierPhaseUncertainty`| 0 | `NULL` | `NULL` | — | 100% NULL (Deprecated API) |
| `TimeOffsetNanos` | 1 | 0.0 | 0.0 | 0.0 | Always 0.0; Android logger uses standard receiver epoch |
| `MultipathIndicator` | 1 | 0 | 0 | 0.0 | Always 0 (`UNKNOWN`); receiver never reports detected multipath bit |
| `IsFullTracking` | 1 | 0 | 0 | 0.0 | Always 0 when non-null (duty-cycling remains active) |
| `_extra_field_1` | 1 | `""` | `""` | 0.0 | Artifact of trailing CSV delimiter |
| `LeapSecond` | 2 | 18 | 19 | 0.06 | Near-constant: 18 everywhere except 1 anomalous session (`08_15_16`) |

### 2.2 Table `Status` Near-Constant Flags
In `Status`, the binary tracking flags have low distinct cardinality by nature:
- `UsedInFix`: 2 distinct values (`0`: 63.6%, `1`: 36.4%).
- `HasAlmanacData`: 2 distinct values (`0`: 74.1%, `1`: 25.9%).
- `HasEphemerisData`: 2 distinct values (`0`: 57.6%, `1`: 42.4%).

### 2.3 Table `Fix` Constant Columns

| Column Name | Exact Distinct Count | Observed Value | Explanation |
|---|---:|---|---|
| `MockLocation` | 1 | 0 | Verified real physical device logs; mock locations disabled |
| `NumberOfUsedSignals` | 0 | `NULL` | Unpopulated by Android Location framework in these builds |
| `VerticalSpeedAccuracyMps`| 0 | `NULL` | Unpopulated |
| `SolutionType` | 0 | `NULL` | Unpopulated |
| `_extra_field_1` | 1 | `""` | Trailing comma artifact |

- **ML Relevance**:
  - All listed constant columns must be purged during feature extraction to reduce schema footprint by ~20%.

---

## 3. Duplicate Checks and Epoch Density

### 3.1 Raw Table Signal Uniqueness Check
- **Goal**: Check whether `(utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz)` or `(..., CodeType)` contains duplicate rows.
- **Why**: GNSS receivers can track multiple PRN code signals on the same carrier frequency (e.g. pilot 'Q' and data 'I' channels on GPS L5 or Galileo E1/E5a). We must determine whether `CodeType` is required for unique primary key definition.
- **SQL**: `docs/EDA/queries/013_duplicate_checks_raw_status.sql` (§1, §2)
- **Result** [OBSERVED]:
  - Test with `CodeType`: **0 duplicate groups** (0 duplicate rows).
  - Test WITHOUT `CodeType`: **0 duplicate groups** (0 duplicate rows).
- **Insight**:
  - In this dataset, the mobile chipsets do **not** log multiple code channels for the same satellite and carrier frequency simultaneously. Even on dual-frequency or pilot-tracking signals, only a single code type is preserved per epoch.
  - Therefore, the composite tuple `(session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz)` is a **strict primary key** across the entire 5.48M rows of `Raw`.

### 3.2 Status Table Duplicate Rows
- **Goal**: Check whether `(session, UnixTimeMillis, ConstellationType, Svid, CarrierFrequencyHz)` is unique in `Status`.
- **SQL**: `docs/EDA/queries/013_duplicate_checks_raw_status.sql` (§3)
- **Result** [OBSERVED]:
  - Duplicate signal keys: **18,974 groups** (totaling **38,358 duplicate rows** across 3,656 distinct epochs).
- **Insight**:
  - In certain sessions (e.g. `135_back_vertical`, `90_back_vertical`), `Status` callback batches were fired twice within the same millisecond or logged duplicate lines from the OS buffer.
- **Filter Recommendation**:
  - When joining or querying `Status`, always deduplicate by `(session, UnixTimeMillis, ConstellationType, Svid, CarrierFrequencyHz)` taking `FIRST()` or `MAX(Cn0DbHz)`.

### 3.3 Epoch Density and Row-Count Distribution
- **Goal**: Evaluate the distribution of visible/tracked signals per epoch. Determine if any epoch exceeds 100 signals.
- **SQL**: `docs/EDA/queries/014_epoch_signal_count_distribution.sql`
- **Result** [OBSERVED]:

| Table | Distinct Epochs | p1 | p25 | p50 (Median) | p75 | p99 | Max Signals / Epoch | Epochs > 100 Signals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Raw** | 87,512 | 26 | 59 | **64** | 72 | 83 | **88** | **0 (0.0%)** |
| **Status** | 87,198 | 53 | 63 | **88** | 93 | 103 | **206** | **3,730 (4.28%)** |

- **Insight**:
  - `Raw` rows per epoch never exceed **88 signals**, representing actively tracked hardware measurement channels.
  - `Status` reports significantly higher satellite counts (median 88 vs 64 in `Raw`). Status includes satellites in view from the broadcast almanac even if the receiver hardware is not actively tracking them.
  - The maximum in `Status` reaches 206 signals per epoch; all epochs with $> 100$ signals in `Status` are caused by the duplicate callback event batches identified in §3.2.

### 3.4 Raw and Status Overlap and Alignment
- **Goal**: Test whether `Raw` and `Status` share identical epoch timestamps.
- **SQL**: `docs/EDA/queries/010_cross_table_alignment.sql` & `021_raw_status_join_alignment.sql`
- **Result** [OBSERVED]:
  - Exact millisecond timestamp matches `(Raw.utcTimeMillis == Status.UnixTimeMillis)`: only **4,948 epochs (5.65%)**.
  - Nearest-time match within $\pm 50 \text{ ms}$: **87,495 epochs (99.98%)** with a median offset $|\Delta t| = 1 \text{ ms}$ (interquartile range $1 \text{ ms} \text{ to } 3 \text{ ms}$).
- **Insight**:
  - `Raw` and `Status` are driven by separate Android callback threads (`GnssMeasurements.Callback` vs `GnssStatus.Callback`). They do not duplicate each other's fields; rather, `Status` provides satellite look angles (elevation, azimuth) which must be merged onto `Raw` measurement tracks via an **ASOF / nearest-timestamp join** within a tight tolerance ($\pm 50 \text{ ms}$).

---

## 4. Outliers and Impossible Values

- **Goal**: Detect non-physical domain values across satellite geometry, signal power, carrier phase, and motion sensors.
- **SQL**: `docs/EDA/queries/015_outliers_and_impossible_values.sql`
- **Result** [OBSERVED]:

### 4.1 $C/N_0 = 0$ Rows and Cross-Reference Q-001
- **Raw table**: **210,539 rows** report `Cn0DbHz = 0.0`. Of these, **209,783 (99.64%)** are GLONASS (Constellation 3).
- **Status table**: **1,263,343 rows** report `Cn0DbHz = 0.0`.
- **Cross-Reference Q-001 Test**:
  - Sampled 1,000 GLONASS rows where `Raw.Cn0DbHz = 0.0`.
  - Joined to `Status` at the same epoch ($\pm 50 \text{ ms}$) on identical satellite PRN/Svid.
  - **999 out of 1,000 rows (99.9%)** in `Status` reported a valid positive signal strength, with an average $C/N_0$ of **26.65 dB-Hz** (range: 9.5 to 33.8 dB-Hz).
- **Domain Interpretation**:
  - In `Raw`, GLONASS tracking channels frequently log a measurement container before the baseband carrier lock loop finishes decoding $C/N_0$, placing an indicator zero. In `Status`, the high-level positioning engine reports the smoothed carrier tracking value.
  - `Raw` $C/N_0 = 0$ represents **missing measurement data**, NOT zero RF signal power.

### 4.2 Elevation and Azimuth Boundaries in `Status`
- **Azimuth**: Min **0.00°**, Max **359.98°**. **0 rows out of bounds** ($[0, 360]$).
- **Elevation**: Max **87.00°**. **0 rows $> 90°$**.
  - Rows $< 0°$ (negative elevation): Exactly **37 rows** (Min **$-5.00°$**).
  - All 37 negative elevation rows occur on rising or setting satellites just below the geometric horizon; 100% of them have `Cn0DbHz = 0.0` or weak acquisition signals.

### 4.3 Upper Bounds on Signal Power (`Cn0DbHz` and `BasebandCn0DbHz`)
- Android GNSS maximum possible value is defined as 63 dB-Hz in AOSP.
- Corpus Maximum `Cn0DbHz`: **54.70 dB-Hz**. Count $> 60 \text{ dB-Hz}$: **0 rows**.
- Corpus Maximum `BasebandCn0DbHz`: **49.70 dB-Hz**. Count $> 60 \text{ dB-Hz}$: **0 rows**.
- Both metrics remain strictly within physical limits for mobile patch antennas.

### 4.4 Accumulated Delta Range (ADR) Extremes
- `AccumulatedDeltaRangeMeters`:
  - Min: **$-27,471,895.97 \text{ m}$**
  - Max: **$+519,705.20 \text{ m}$**
  - Rows with $| \text{ADR} | > 10^9 \text{ m}$: **0 rows**.
- **Insight**:
  - Large negative values reflect uninitialized phase counters prior to cycle slip resolution. When `AccumulatedDeltaRangeState & 1 == 0` (ADR invalid, 60.6% of rows), the ADR meter value is arbitrary and must not be used for phase differencing.

### 4.5 Position Fix Boundaries
- `LatitudeDegrees`: Min **18.4588° N**, Max **18.4618° N**. (0 rows outside $[-90, 90]$).
- `LongitudeDegrees`: Min **73.8816° E**, Max **73.8844° E**. (0 rows outside $[-180, 180]$).
- `SpeedMps`: Min **0.00 m/s**, Max **3.75 m/s**. (0 negative rows).
- **Insight**:
  - All test sessions are physically confined to a tight 300-meter test area in Pune, India. Speeds correspond to stationary mounts or slow pedestrian walking.

### 4.6 Accelerometer Gravity Norm Distribution
- **Metric**: $\|a\| = \sqrt{X^2 + Y^2 + Z^2}$ across 6,369,268 rows of `Accel.parquet`.
- **Distribution** [OBSERVED]:
  - **Min**: 0.211 m/s²
  - **p1**: 9.744 m/s²
  - **p50 (Median)**: **9.927 m/s²**
  - **p99**: 10.087 m/s²
  - **Max**: 80.870 m/s²
  - **Mean**: **9.917 m/s²** (StdDev: 0.268 m/s²)
  - Samples with $\|a\| < 7.0 \text{ m/s²}$: **3,430 (0.054%)**
  - Samples with $\|a\| > 12.0 \text{ m/s²}$: **8,526 (0.134%)**
- **Insight**:
  - Over 99.8% of accelerometer samples fall within $[9.7, 10.1] \text{ m/s²}$, demonstrating that the device was held or mounted in quasi-static conditions with excellent sensor calibration.
  - The few transient outliers ($> 12 \text{ m/s²}$) correspond to physical phone pickup or adjustment shocks.

---

## 5. Timestamp Integrity and Hardware Clock Behavior

- **Goal**: Verify hardware clock monotonicity, sampling continuity, and cross-clock synchronization.
- **SQL**: `docs/EDA/queries/016_timestamp_problems_and_discontinuities.sql`
- **Result** [OBSERVED]:

### 5.1 Monotonicity of `TimeNanos` within Sessions
- Total epoch transitions in `Raw`: **87,417**.
- Non-monotonic `TimeNanos` transitions ($\Delta t_{\text{nanos}} < 0$): **0 (0.0%)**.
- Non-monotonic `utcTimeMillis` transitions ($\Delta t_{\text{utc}} < 0$): **0 (0.0%)**.
- Minimum $\Delta t_{\text{nanos}}$: **+194,577,617 ns** (~194.6 ms).
- **Insight**:
  - Receiver clock counters advance strictly monotonically within every session without a single backward clock step or timestamp corruption.

### 5.2 Large Sampling Gaps ($> 2.0 \text{ seconds}$)
- `Raw`: **0 gaps $> 2.0 \text{ seconds}$** (Maximum gap observed is **1,820 ms**).
- `Fix`: **0 gaps $> 2.0 \text{ seconds}$** (Maximum gap observed is **1,887 ms**).
- `Accel`: **0 gaps $> 2.0 \text{ seconds}$** (Maximum gap observed is **1,553 ms**).
- `Status`: Exactly **4 gaps $> 2.0 \text{ seconds}$** across the entire dataset:
  - `gnss_log_2026_10_22_10_02_30`: one gap of 5.0s, one gap of 10.0s.
  - `gnss_log_2026_10_23_17_00_21`: one gap of 3.0s.
  - `gnss_log_2026_10_19_08_51_20`: one gap of 3.0s.
- **Insight**:
  - The GNSS measurement stream is virtually uninterrupted at 1 Hz. Even in `Status`, 87,194 out of 87,198 steps (99.995%) are unbroken.

### 5.3 HardwareClockDiscontinuityCount Shifts
- Number of distinct values corpus-wide: **23** (ranging from 0 to 36).
- Number of sessions where `HardwareClockDiscontinuityCount` changed mid-stream: **0 sessions**.
- **Insight**:
  - The hardware clock counter remained completely stationary throughout every logging run. No hardware clock resets or oscillator power cuts occurred during active sessions.

### 5.4 Cross-Clock Drift: `utcTimeMillis` vs. `elapsedRealtimeNanos`
- Evaluated boot clock offset: $\Delta = (\text{utcTimeMillis} \times 10^6) - \text{elapsedRealtimeNanos}$.
- Standard deviation of offset within session: **289.6 µs** (0.289 ms).
- **Insight**:
  - The Android system clock and monotonic boot clock track within sub-millisecond precision over session lifetimes, confirming that either timestamp can be safely converted to UTC epoch time.

---

## 6. Summary of ML Recommendations and Quality Gates

| Check Category | Observed Finding | Action / Filter Recommendation | Target / Leakage Guard |
|---|---|---|---|
| **$C/N_0 = 0$ Values** | 210,539 rows (3.84%) in `Raw` are zeros, mostly GLONASS tracking states | **Filter from training loss**: Exclude rows with $C/N_0 = 0$ from continuous regression target | Critical: Prevents predicting non-physical negative drops |
| **Status Duplicates** | 18,974 duplicate signal records in `Status` | **Deduplicate before join**: Group by `(session, UnixTimeMillis, Svid, CarrierFrequencyHz)` taking `FIRST()` | Prevents row explosion in cross-table joins |
| **Cross-Table Alignment** | Exact millisecond match is only 5.7%; 99.98% within $\pm 50 \text{ ms}$ | **Use ASOF Join**: Match `Status` and orientation features to `Raw` on `session` and nearest timestamp $\le 50 \text{ ms}$ | Preserves continuous geometry tracking |
| **Constant Columns** | 6 columns in `Raw` and 3 in `Fix` have zero variance | **Purge from pipeline**: Drop `CarrierCycles`, `CarrierPhase*`, `MultipathIndicator`, `IsFullTracking`, `SolutionType`, `MockLocation` | Eliminates degenerate zero-variance inputs |
| **Device Missingness** | `Klobuchar*`, `SvPosition*`, `Drift*` are structurally missing on specific phones | **Avoid hard dependence**: Do not use these as mandatory single-feature splits across multi-device sets | Prevents test-time out-of-distribution failure |
| **Status Flags** | `UsedInFix` and `SignalCount` reflect downstream PVT solution | **Do NOT use as $t+15$ inputs**: Only historical/lagged flags ($t \le 0$) may enter feature set | Strict leakage prevention |
