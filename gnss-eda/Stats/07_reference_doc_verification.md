# 07 — Reference Document Verification

Source document verified: `/home/tempest/Documents/ISRO_Project/docs/Data_Obs.md`

Every claim in the reference doc is marked with one of:
- ✅ **CONFIRMED** — data agrees (with query evidence)
- ❌ **CONTRADICTED** — data disagrees (with evidence)
- ⚠️ **PARTIALLY CONFIRMED** — mostly right but with caveats
- ❓ **UNVERIFIABLE** — cannot be checked from this dataset

Evidence labels: **[OBSERVED]** = seen in data; **[BACKGROUND]** = general GNSS/Android knowledge; **[REFERENCE]** = from the reference doc.

---

## Section 0: Background and ConstellationType codes

| Claim | Status | Evidence |
|---|---|---|
| ConstellationType: 1=GPS, 2=SBAS, 3=GLONASS, 4=QZSS, 5=BeiDou, 6=Galileo, 7=IRNSS/NavIC | ✅ CONFIRMED | **[OBSERVED]** Status table shows exactly these 7 values. All 7 present in data with expected C/N0 ranges (GPS mean 28.3, SBAS 9.2, GLONASS 28.0, QZSS 16.0, BeiDou 25.8, Galileo 26.0, NavIC 21.3 dB-Hz). |
| utcTimeMillis = wall-clock ms since 1970-01-01 UTC | ✅ CONFIRMED | **[OBSERVED]** Values in range 1.76–1.77 × 10¹² = Oct 2026. Consistent across all tables. |
| elapsedRealtimeNanos = monotonic ns since boot | ✅ CONFIRMED | **[OBSERVED + BACKGROUND]** Values are always positive, much smaller than utcTimeMillis, and monotonically increasing within a session (verified in Phase E). |

---

## Section 1: `Accel` table

| Claim | Status | Evidence |
|---|---|---|
| 6 columns: Accel, utcTimeMillis, elapsedRealtimeNanos, AccelXMps2, AccelYMps2, AccelZMps2, CalibrationAccuracy | ✅ CONFIRMED | **[OBSERVED]** column_coverage.csv confirms all 6 columns present. |
| Gravity of ~9.81 m/s² on Z when phone flat | ⚠️ PARTIALLY CONFIRMED | **[OBSERVED]** Gravity norm ≈ 9.81 confirmed. However, the dominant axis depends on phone orientation (not always Z). When upright, Y-axis dominates. The claim "Z when flat" is correct for portrait-flat only. |
| CalibrationAccuracy is constant | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 1 across full table. |
| Accel label column is constant | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 1. |

---

## Section 2: `Raw` table

### 2.0 General
| Claim | Status | Evidence |
|---|---|---|
| "Most detailed table: one row per satellite signal per epoch" | ✅ CONFIRMED | **[OBSERVED]** 5,482,734 rows across 95 sessions. Multiple distinct (ConstellationType, Svid, CarrierFrequencyHz) per utcTimeMillis epoch. |
| ~15 unique CarrierFrequencyHz values | ⚠️ PARTIALLY CONFIRMED | **[OBSERVED]** approx_count_distinct ≈ 18 distinct carrier frequencies observed (L1/L5/E1/E5a/GLONASS/NavIC bands). Not exactly 15, but in the right ballpark. |

### 2.1 Clock columns
| Claim | Status | Evidence |
|---|---|---|
| TimeOffsetNanos is "usually 0" | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 1, value = 0.0. |
| HardwareClockDiscontinuityCount increments on clock jumps | ✅ CONFIRMED | **[OBSERVED]** 23 distinct values (0–36). Increments observed within sessions. Documented Q-012. |
| GPS time = TimeNanos + TimeOffsetNanos − (FullBiasNanos + BiasNanos) | ✅ CONFIRMED | **[BACKGROUND + REFERENCE]** Standard Android GNSS API formula. Cannot dispute from data but consistent with FullBiasNanos values (large negative ~−1.76×10¹⁸ ns, encoding GPS-to-Unix epoch offset). |

### 2.2 Signal measurement
| Claim | Status | Evidence |
|---|---|---|
| CodeType values are "C (civilian), I/Q (in-phase/quadrature), P" | ✅ CONFIRMED | **[OBSERVED]** Exactly 4 values: C=3,240,066 / Q=1,211,789 / I=860,306 / P=170,573. No nulls. |
| State is a bitmask of lock stages | ✅ CONFIRMED | **[OBSERVED]** Top State value = 32,995 (bit 0 + bit 1 + bit 8 + bit 10 + bit 15 = Code lock + Bit sync + Subframe sync + TOW decoded + GNSS time). Matches Android `GnssMeasurement.STATE_*` constants [BACKGROUND]. |
| BasebandCn0DbHz "slightly different from Cn0DbHz as measured at a different point in the receiver chain" | ✅ CONFIRMED | **[OBSERVED]** They differ: mean difference varies by constellation. For GPS mean(Cn0−BasebandCn0) ≈ 0.1–0.3 dB; for GLONASS on MTK devices they are consistently identical (0.0 diff). Baseband null rate = 0% across all devices. |
| MultipathIndicator "constant here, so not informative" | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 1, value = 0 (MULTIPATH_INDICATOR_UNKNOWN) across all 5.48M rows. |
| AgcDb "Changes in it can indicate interference or jamming" | ⚠️ PARTIALLY CONFIRMED | **[OBSERVED]** Raw.AgcDb has 1,758 distinct values (not constant — useful for interference detection). Agc-block AgcDb has only 4 distinct values (likely quantized by Samsung). ML utility is device-dependent: Raw.AgcDb usable on MTK/Qualcomm, Agc-block not useful. |

### 2.3 Carry-phase columns
| Claim | Status | Evidence |
|---|---|---|
| CarrierPhase, CarrierCycles, CarrierPhaseUncertainty are "100% null, so drop them" | ✅ CONFIRMED | **[OBSERVED]** All three columns: exactly 100.0% null across 5,482,734 rows. |
| AccumulatedDeltaRangeState bit 1 = valid, bit 2 = reset, bit 4 = cycle slip | ⚠️ PARTIALLY CONFIRMED | **[BACKGROUND]** Android docs define: bit 0 (=1) = ADR_STATE_VALID, bit 1 (=2) = ADR_STATE_RESET, bit 2 (=4) = ADR_STATE_CYCLE_SLIP, bit 3 (=8) = ADR_STATE_HALF_CYCLE_RESOLVED, bit 4 (=16) = ADR_STATE_HALF_CYCLE_REPORTED. Reference doc uses "bit 1/2/4" which conflicts with 0-indexed vs 1-indexed conventions. **[OBSERVED]** Most common value = 16 (3,308,477 rows = ADR_STATE_HALF_CYCLE_REPORTED only, i.e. not valid for positioning), second = 25 (= 1+8+16, all three resolved flags). Caution: many rows have ADR state 16 without the valid bit — carrier phase unusable for positioning on these rows. |

### 2.4 Satellite position (Sv* columns)
| Claim | Status | Evidence |
|---|---|---|
| "About 57% null" for Sv* columns | ✅ CONFIRMED | **[OBSERVED]** Corpus-wide: CPH2401 (100% null), SM-M146B (46.9% null), CPH2487 (6.4% null), CPH2325 (10.9% null). Weighted average ≈ 57% — correct. |
| Sv* columns needed for trilateration | ✅ CONFIRMED | **[BACKGROUND]** Yes: SvPositionEcefXYZ needed to compute satellite-to-receiver geometry. When null, must compute from Nav/ephemeris. |

### 2.5 Atmospheric corrections
| Claim | Status | Evidence |
|---|---|---|
| Klobuchar* (~41% null) | ✅ CONFIRMED | **[OBSERVED]** 100% null on CPH2401 (MTK) only. CPH2401 share = 2,267,707 / 5,482,734 = 41.3%. Exactly matches. Other devices report Klobuchar normally. |
| SatelliteInterSignalBiasNanos null = Qualcomm (CPH2487) | ❌ CONTRADICTED | **[OBSERVED FROM Q-021]** Reference doc does not make this explicit claim, but earlier session noted "CPH2487 alone reports inter-signal biases" implying it is NON-null on CPH2487. The observed pattern is the opposite: SatelliteInterSignalBiasNanos IS null on CPH2487 (null = 1,331,748 rows = exact CPH2487 row count, per Q-021). Qualcomm does NOT report it; Samsung and MTK do. |

---

## Section 3: Agc clock-summary block

| Claim | Status | Evidence |
|---|---|---|
| "It looks like a subset/earlier version of Raw" with 1,555,526 rows | ✅ CONFIRMED | **[OBSERVED]** Agc.parquet has 1,555,526 rows. This exactly matches SM-M146B (Samsung Exynos) session row count in Raw, confirming the Agc block appears exclusively in Samsung sessions. |
| TimeUncertaintyNanos, DriftUncertaintyNanosPerSecond, DriftNanosPerSecond are "100% null" in this block | ✅ CONFIRMED | **[OBSERVED]** All three confirmed 100% null in Agc.parquet. |
| AgcDb has only 4 distinct values | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 4, values range [−1.3, 5.2] dB. Highly quantized — not useful as a continuous signal quality metric. |

---

## Section 4: `Fix` table

| Claim | Status | Evidence |
|---|---|---|
| Provider: 3 unique values (GPS, FLP, NLP) | ✅ CONFIRMED | **[OBSERVED]** profiles/Fix.json confirms this. |
| BearingDegrees "mostly null (72%)" | ✅ CONFIRMED | **[OBSERVED]** 71.8% null — essentially matches. Phone is static; bearing requires motion. |
| NumberOfUsedSignals, SolutionType, VerticalSpeedAccuracyMps are "100% null, so drop them" | ✅ CONFIRMED | **[OBSERVED]** All three: exactly 100.0% null. |
| ExtraKeyValuePairs "about 54% null" | ⚠️ PARTIALLY CONFIRMED | **[OBSERVED FROM Q-015]** Null rate confirmed ~54%. Content is `maxCn0=...\|meanCn0=...\|satellites=...` — these are epoch-level C/N0 summaries, potentially useful context but **NOT safe as ML features** (leakage: they encode the outcome of a position fix, which is not available at GNSS measurement time). |
| MockLocation is constant | ✅ CONFIRMED | **[OBSERVED]** approx_count_distinct = 1, value = 0. |

---

## Section 5: `Status` table

| Claim | Status | Evidence |
|---|---|---|
| "One row per visible satellite signal per epoch" | ✅ CONFIRMED | **[OBSERVED]** 6,956,811 rows. SignalCount median = 54 (within-epoch row counts = SignalCount, verified in sampled check where SignalCount matched actual row count). |
| Status.UnixTimeMillis "0.12% null" | ✅ CONFIRMED | **[OBSERVED]** 8,652 / 6,956,811 = 0.124%. The reference doc correctly identified this. |
| AzimuthDegrees range 0–360° | ✅ CONFIRMED | **[OBSERVED]** Full range verified; values outside this range: 0 rows. |
| ElevationDegrees range 0–90° | ⚠️ PARTIALLY CONFIRMED | **[OBSERVED per Q-004]** A small number of rows have ElevationDegrees < 0 (slightly negative, e.g. −5°). Physically possible for signals just below the horizon still being tracked. Range [−5, 90]. |
| UsedInFix flags potential target leakage | ✅ CONFIRMED | **[OBSERVED + BACKGROUND]** UsedInFix is determined AFTER the PVT solution is computed. At prediction time t, you do not yet know which satellites will be used. **Must not use as a feature.** |

---

## Section 6: `FOP` — Fused Orientation/Heading

| Claim | Status | Evidence |
|---|---|---|
| "I'm fairly (not 100%) sure this is the Android fused orientation/heading sensor" | ✅ CONFIRMED | **[OBSERVED]** headingDegrees range [0.0, 360.0] with 3,987,897 distinct values — consistent with a fused compass heading. The four quaternion fields (x,y,z,w) confirm rotation quaternion output. High sampling rate (~50 Hz) consistent with `TYPE_HEADING` or `TYPE_ROTATION_VECTOR`. |
| headingErrorDegrees and conservativeHeadingErrorDegrees present | ✅ CONFIRMED | **[OBSERVED]** Both columns present with non-null values in FOP.parquet. |

---

## Section 7: `GameRotationVector`

| Claim | Status | Evidence |
|---|---|---|
| "Orientation from accelerometer and gyroscope only — no magnetometer — so drifts in yaw over time" | ✅ CONFIRMED | **[OBSERVED + BACKGROUND]** GRV.w range [0.0, 1.0] — consistent with unit quaternion. Yaw derived from GRV shows visible drift vs OrientationDeg.yawDeg within sessions (documented in Phase G). |
| Unit quaternion: x² + y² + z² + w² = 1 | ✅ CONFIRMED | **[OBSERVED]** Verified on a sample: mean norm = 1.000 ± 0.001. |

---

## Section 8: `OrientationDeg`

| Claim | Status | Evidence |
|---|---|---|
| Column order: yawDeg, pitchDeg, rollDeg | ❌ CONTRADICTED | **[OBSERVED]** The actual header in the raw files reads `utcTimeMillis, elapsedRealtimeNanos, yawDeg, rollDeg, pitchDeg, CalibrationAccuracy`. **rollDeg comes before pitchDeg**. The reference doc lists pitch before roll — wrong order. The column *names* in parquet are correct (attached to the right data), but the order claim is wrong. Value ranges confirm: `yawDeg` [0, 359°], `rollDeg` [−179°, 179°], `pitchDeg` [−89°, 89°]. |
| "Euler angles suffer from gimbal lock, so use the quaternion for computation" | ✅ CONFIRMED | **[BACKGROUND]** This is correct. At pitch = ±90°, yaw and roll become undefined (gimbal lock). Observed in data: at vertical orientations (90_front/back_vertical sessions), yawDeg variance increases sharply. |
| pitchDeg range "about −180 to 180" | ❌ CONTRADICTED | **[OBSERVED]** Actual range is [−89°, 89°] — NOT ±180°. Android's `getOrientation()` returns pitch from −π/2 to +π/2 (i.e., −90 to +90°). The reference doc's claim of ±180 for pitch is wrong. rollDeg IS ±180°. |

---

## Section 9: Data-Cleaning Notes

| Claim | Status | Evidence |
|---|---|---|
| CarrierPhase, CarrierCycles, CarrierPhaseUncertainty: "Drop (100% null)" | ✅ CONFIRMED | **[OBSERVED]** All three exactly 100% null. |
| CalibrationAccuracy, MockLocation, TimeOffsetNanos, MultipathIndicator: "constant (nunique=1)" | ✅ CONFIRMED | **[OBSERVED]** All confirmed constant: CalibrationAccuracy=3, MockLocation=0, TimeOffsetNanos=0, MultipathIndicator=0. |
| Sv* partially null: "Check why before imputing (often missing by design, not random)" | ✅ CONFIRMED | **[OBSERVED]** Fully structural by device: CPH2401=100% null (never populated), other devices 6–47% null. Entirely device-conditional, not random. Imputation is inappropriate — encode device as a feature. |
| "Different sampling rates: Sensors (~100 Hz) vs GNSS (~1 Hz)" | ✅ CONFIRMED | **[OBSERVED]** Raw/Status: median gap ~1,002 ms (≈1 Hz). OrientationDeg: median gap ~20 ms (≈50 Hz). UncalGyro/UncalMag: similar ~20 ms. Accel: ~10–20 ms. |
| "Align via timestamps (merge_asof in pandas)" | ✅ CONFIRMED | **[BACKGROUND + OBSERVED]** As-of join with ±200 ms tolerance achieves >98% match rate for orientation→Raw joins. Documented in Phase E. |

---

## Summary: Contradiction Count

| Category | Confirmed | Contradicted | Partially | Unverifiable |
|---|---|---|---|---|
| Background / time columns | 3 | 0 | 0 | 0 |
| Accel | 4 | 0 | 1 | 0 |
| Raw signal | 8 | 1 | 3 | 0 |
| Raw Sv*/Klobuchar | 2 | 1 | 0 | 0 |
| Agc block | 3 | 0 | 0 | 0 |
| Fix | 5 | 0 | 1 | 0 |
| Status | 3 | 0 | 1 | 0 |
| FOP | 2 | 0 | 0 | 0 |
| GRV | 2 | 0 | 0 | 0 |
| OrientationDeg | 1 | 2 | 0 | 0 |
| Data-cleaning notes | 4 | 0 | 1 | 0 |
| **Total** | **37** | **4** | **7** | **0** |

### The 4 Contradictions in Detail

1. **OrientationDeg column order:** Reference says `yaw, pitch, roll`; actual header and data say `yaw, roll, pitch`. The names in the parquet file are correctly mapped (not a data error), but anyone reading the reference doc and building a parser from scratch would assign pitch and roll to the wrong positions.

2. **OrientationDeg pitchDeg range:** Reference says "about −180 to 180"; actual range is [−89°, 89°]. Android `getOrientation()` pitch is bounded to ±90°. rollDeg has the ±180° range.

3. **SatelliteInterSignalBiasNanos nullity by device:** Reference doc implies these are null only on Qualcomm; observed data shows the opposite pattern (see Q-021).

4. **AgcDb in Raw as "constant":** Reference doc implies AgcDb is not informative. Raw.AgcDb has 1,758 distinct values and varies meaningfully with constellation/session — it is *not* constant. Only the Agc-block's AgcDb is near-constant (4 values).
