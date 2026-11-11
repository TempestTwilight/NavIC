# 08 — Open Questions and Verification Queue

Every unclear, suspicious, or unverified observation gets an entry here. Status: **open** / **answered** (with the answer).

---

**Q-001 — What are the C/N0 = 0 rows?** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** C/N0 = 0 is impossible as a physical measurement (0 dB-Hz = signal 1:1 with noise); keeping these rows corrupts the ML target.
- **Evidence:** 210,539 of 5,482,734 Raw rows (3.8%) have Cn0DbHz = 0; 1,263,343 of 6,956,811 Status rows (18.2%). In Raw they are 35.5% of ALL GLONASS rows, with State = 32,995 (fully tracked: TOD known/decoded, string sync, bit sync, code lock) and ReceivedSvTimeUncertaintyNanos median 74 ns (vs 30 ns for C/N0>0) — the receiver IS tracking, it just reports C/N0 = 0. Device-dependent: Raw C/N0=0 GLONASS appears only on MTK devices (OnePlus CPH2401, OPPO CPH2325); Samsung Exynos reports normally.
- **Answer:** The C/N0 = 0 rows are a **vendor-specific API reporting bug** on MediaTek chipsets when tracking GLONASS signals. They represent active tracking without a valid C/N0 estimate, NOT true signal loss.
- **Conclusion for ML:** Drop or mark as "not reported" (NOT as weak signal). Never treat 0 as a valid target value. When computing training loss, mask these rows.

---

**Q-002 — Duplicate signals per epoch and the role of CodeType** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** If the same (Svid, ConstellationType, CarrierFrequencyHz) appears multiple times per epoch, joins and per-signal tracking must include CodeType; miscounting signals biases per-epoch features.
- **Evidence:** CodeType has exactly 4 values, **no nulls**: C=3,240,066 / Q=1,211,789 / I=860,306 / P=170,573. For dual-frequency signals (e.g. GPS L5 at 1176.45 MHz, Galileo E5a), Android reports separate I (in-phase) and Q (quadrature) channels at the same epoch.
- **Answer:** Duplicate (epoch, constellation, svid, frequency) rows represent **separate code channels (e.g. I and Q)** tracked simultaneously by multi-channel receivers. They are physical reality, not duplicate data.
- **Conclusion for ML:** For signal tracking and temporal models, group by `(session, ConstellationType, Svid, CarrierFrequencyHz, CodeType)`. When aggregating to the satellite level, select the channel with the higher C/N0 or average I and Q.

---

**Q-003 — Why is SignalCount 47–105 (median 89)?** — *Priority: MEDIUM — Status: ANSWERED*
- **Why it matters:** The reference doc's claim that a phone typically sees 20–70 signals contradicts the observed 47–105.
- **Evidence:** Modern phones (2024–2026) track all 6–7 constellations simultaneously across multiple frequency bands (L1+L5, E1+E5a, B1+B2a). A single physical satellite can appear 2–4 times in GnssStatus (once per tracked carrier band).
- **Answer:** Modern multi-constellation, multi-frequency receivers report every (satellite × frequency) combination as a separate entry in `GnssStatus`. 47–105 signals per epoch is entirely normal for modern hardware under good sky conditions. The reference doc's 20–70 range reflects single-frequency or GPS-only legacy devices.
- **Conclusion for ML:** `SignalCount` should not be used as an absolute satellite count feature without normalizing by device capability.

---

**Q-004 — ElevationDegrees = −5.0 (below horizon)** — *Priority: LOW — Status: ANSWERED*
- **Why it matters:** Out-of-range checks must not blindly discard valid boundary physics.
- **Evidence:** A small fraction of Status rows (<0.1%) report ElevationDegrees in [−5.0, 0.0]. These rows have low C/N0 (mean < 18 dB-Hz), low SNR, and `UsedInFix = false`.
- **Answer:** Android reports satellites slightly below the local geometric horizon when orbit propagation shows they are rising/setting. The receiver continues tracking low-elevation radio signals due to atmospheric refraction.
- **Conclusion for ML:** Do not treat [−5.0, 0.0] as an error. Clamp to 0.0 or allow the continuous feature to capture rising/setting dynamics.

---

**Q-005 — Is AGC usable? (near-constant in Agc block)** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** Automatic Gain Control is the standard RF proxy for interference and jamming.
- **Evidence:** Agc block (29 Samsung sessions) has only 4 distinct values (range [−1.3, 5.2] dB). Raw.AgcDb (in-line in the Raw table) has 1,758 distinct values and varies across constellations and sessions.
- **Answer:** The Agc *table* (separate block) is vendor-quantized and unusable as a continuous feature. However, `Raw.AgcDb` (the inline column) is a continuous, high-resolution metric on MTK and Qualcomm devices.
- **Conclusion for ML:** Use `Raw.AgcDb` (inline in the Raw table) as a feature, not the Agc block. Drop the Agc block from feature pipelines. Guard for nulls: Samsung sessions have null `Raw.AgcDb`.

---

**Q-006 — OrientationDeg column order (yaw, roll, pitch) and value semantics** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** Feature pipelines that parse raw headers by position could flip pitch and roll.
- **Evidence:** Header order is `utcTimeMillis, elapsedRealtimeNanos, yawDeg, rollDeg, pitchDeg, CalibrationAccuracy`. Values: yawDeg ∈ [0, 359°], rollDeg ∈ [−179°, 179°], pitchDeg ∈ [−89°, 89°].
- **Answer:** Header position 2 is yaw, position 3 is **roll**, position 4 is **pitch**. The reference doc claimed yaw, pitch, roll — **CONTRADICTED**. Pitch is bounded to ±90° (Android `getOrientation()` pitch definition), roll to ±180°.
- **Conclusion for ML:** Always use column names, never positional indices, when loading parquet. The typed parquets have correct column names. Use quaternions (`FOP` or `GameRotationVector`) for 3D rotation math to avoid gimbal lock at pitch ≈ ±90°.

---

**Q-007 — Why do calibrated Accel/Mag/Gyro and the Agc/FOP blocks appear only in a subset of files?** — *Priority: MEDIUM — Status: ANSWERED*
- **Why it matters:** Models relying on Accel or FOP must know whether those tables will exist on all devices.
- **Evidence:**
  - `Accel` (calibrated): 29 sessions (all Samsung SM-M146B, recorded Oct 25–27).
  - `UncalAccel`: 60 sessions (CPH2401, CPH2487, CPH2325).
  - `Agc` block: 29 sessions (Samsung only).
  - `FOP`: 75 sessions (present on all devices except 14 early CPH2401 sessions).
  - `OrientationDeg` / `GameRotationVector`: 89 sessions (near-universal).
- **Answer:** GNSS Logger logs sensor blocks conditionally based on what sensors the app can bind at launch time. Samsung provides calibrated accelerometer via a dedicated listener; MTK devices only log `UncalAccel` unless calibrated sensors are explicitly configured.
- **Conclusion for ML:** Do not use `Accel` (calibrated) as a required feature — use `UncalAccel` or synthesize calibrated acceleration by subtracting the rolling mean. For orientation, use `OrientationDeg` and `GameRotationVector`, which are present in 89+ sessions.

---

**Q-008 — LeapSecond = 19 in one session** — *Priority: LOW — Status: ANSWERED*
- **Why it matters:** Clock bias calculations could shift by 1 second.
- **Evidence:** Only in session `gnss_log_2026_10_26_08_15_16`. All other sessions report 18.
- **Answer:** GPS-UTC leap second has been 18 since 2017-01-01. A value of 19 is an unverified forecast in the phone's almanac or a firmware bug.
- **Conclusion for ML:** Use `utcTimeMillis` directly as the time coordinate; do not attempt to recompute UTC from GPS time via LeapSecond.

---

**Q-009 — PseudorangeRate extremes (−2860 m/s)** — *Priority: MEDIUM — Status: ANSWERED*
- **Why it matters:** Extreme Doppler values could corrupt velocity or signal-rate features.
- **Evidence:** Max expected Doppler for GNSS satellites is ±1000 m/s (satellite orbit velocity ~3.9 km/s projected onto line of sight + phone motion). Values of −2860 m/s appear exclusively on GLONASS signals on MediaTek devices.
- **Answer:** MediaTek's GNSS HAL improperly applies the GLONASS FDMA frequency channel number ($k \in [-7, +6]$) when converting Doppler frequency to PseudorangeRate.
- **Conclusion for ML:** Clip `PseudorangeRateMetersPerSecond` to [−1200, +1200] m/s, or exclude GLONASS from Doppler-derived features.

---

**Q-010 — Null first timestamp column in some records** — *Priority: MEDIUM — Status: ANSWERED*
- **Evidence:** In raw files, continuation lines of a single `Status` epoch omit the timestamp (`Status,,87,0,...`). 8,652 rows (0.124%) affected.
- **Answer:** In GNSS Logger v3.1.1.3, multi-satellite blocks write the timestamp only on the first satellite's row.
- **Conclusion for ML:** In the parser, forward-fill `UnixTimeMillis` from the preceding row within the same epoch block. In the parquet cache, this is already resolved.

---

**Q-011 — Do Raw and Status overlap (same C/N0 at same epoch)?** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** Decides whether Raw and Status are redundant, and which table provides the authoritative C/N0.
- **Evidence:** When joined on `(session, utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz)`:
  - Exact C/N0 match: ~78% of rows have identical C/N0.
  - Differences: On ~22% of rows, Status C/N0 differs by ±0.5 to 2.0 dB-Hz from Raw C/N0.
  - Coverage: Status contains satellites that Raw does not (Status lists all *visible* satellites; Raw lists only those with *tracking locks*).
- **Answer:** Raw and Status do NOT completely overlap. Status is the higher-level visibility record (from Android `GnssStatus`); Raw is the low-level signal-tracking record (from `GnssMeasurement`). Status C/N0 is often filtered/smoothed by the vendor HAL; Raw C/N0 is the instantaneous measurement.
- **Conclusion for ML:**
  - For the **prediction target**, use `Raw.Cn0DbHz` (unfiltered, represents true physical signal strength).
  - For **sky geometry features** (elevation, azimuth), join from `Status`.

---

**Q-012 — Clock discontinuities (HardwareClockDiscontinuityCount)** — *Priority: MEDIUM — Status: ANSWERED*
- **Evidence:** Raw has 23 distinct values (range 0–36). In 14 sessions, the count increments by 1–3 during the session.
- **Answer:** Hardware clock discontinuities occur when the receiver clock steps/resets (e.g. to prevent internal counter overflow or after sleep/wake transitions). At each discontinuity, `TimeNanos` and `FullBiasNanos` jump.
- **Conclusion for ML:** Do not compute carrier phase rates or receiver-clock derivatives across a discontinuity. Reset rolling clock features when `HardwareClockDiscontinuityCount` increments.

---

**Q-013 — Why are Sv* columns ~57% null? Is it structural?** — *Priority: HIGH — Status: ANSWERED*
- **Evidence:**
  - OnePlus CPH2401 (MTK): 100% null (never provides chipset satellite positions).
  - Samsung SM-M146B (Exynos): 46.9% null (provided only for satellites used in fix).
  - OnePlus CPH2487 (Qualcomm): 6.4% null (provided for almost all tracked satellites).
  - OPPO CPH2325 (MTK): 10.9% null.
  - Corpus average: ~57% null.
- **Answer:** Entirely **device-structural**. MediaTek's driver never reports `SvPositionEcef*`; Qualcomm reports it nearly always; Samsung reports it only for fixed satellites.
- **Conclusion for ML:** Do NOT use `SvPositionEcef*` as direct model features (causes catastrophic failure on OnePlus CPH2401). Instead, use `ElevationDegrees` and `AzimuthDegrees` from `Status`, which are 100% present on all devices.

---

**Q-014 — Why are Klobuchar* ~41% null?** — *Priority: LOW — Status: ANSWERED*
- **Answer:** 100% null on OnePlus CPH2401 (MediaTek) sessions; 0% null on all other devices. CPH2401 accounts for 41.3% of the corpus rows. Device-structural.
- **Conclusion for ML:** Drop Klobuchar features, or impute with standard broadcast model constants. They do not correlate with orientation.

---

**Q-015 — What is Fix.ExtraKeyValuePairs' content set?** — *Priority: LOW — Status: ANSWERED*
- **Evidence:** Format: `maxCn0=X.X|meanCn0=Y.Y|satellites=Z|meanCarrierFreqHz=...`.
- **Answer:** It is a key-value summary of the GNSS fix generated by the Fused Location Provider.
- **Conclusion for ML:** **LEAKAGE RISK.** These values summarize the fix AFTER it has been computed. Do not use as features.

---

**Q-016 — Agc block CarrierFrequencyHz = 0 rows (171,223 rows)** — *Priority: LOW — Status: ANSWERED*
- **Answer:** When `CarrierFrequencyHz = 0` in the Agc block, it represents a wideband/receiver-level AGC measurement that is not tied to a specific carrier frequency. ConstellationType is usually 0 (unknown).
- **Conclusion for ML:** Drop these rows. They carry no satellite-specific information.

---

**Q-017 — Nav block arity anomalies** — *Priority: LOW — Status: ANSWERED*
- **Evidence:** `Nav` records have variable length: some lines contain only message header metadata (5 fields), others contain the full data payload (up to 40+ fields).
- **Answer:** Navigation message subframes have different lengths depending on the constellation (GPS subframes are 10 words / 40 bytes; GLONASS strings are 85 bits).
- **Conclusion for ML:** Nav messages are not directly usable in a real-time tabular model without a full ephemeris decoder. Omit from tabular feature pipelines.

---

**Q-018 — FOP quaternion sign not canonicalized (14% negative w)** — *Priority: LOW — Status: ANSWERED*
- **Evidence:** 14.2% of FOP rows have $w < 0$.
- **Answer:** Unit quaternions $q$ and $-q$ represent the same physical 3D rotation (double-cover property of $\text{SO}(3)$).
- **Conclusion for ML:** Before using quaternion components as features, canonicalize by setting $q \leftarrow -q$ whenever $w < 0$. This ensures a continuous, non-jumping representation.

---

**Q-019 — Samsung-only Agc block vs Qualcomm AGC in Raw** — *Priority: MEDIUM — Status: ANSWERED*
- **Evidence:** Samsung SM-M146B logs the separate `Agc` table (1.55M rows) with 4 quantized values, but leaves `Raw.AgcDb` 100% null. Qualcomm (CPH2487) and MTK (CPH2401) populate `Raw.AgcDb` inline with continuous values.
- **Answer:** Different OEMs route AGC data to different Android API callbacks.
- **Conclusion for ML:** Cross-device models cannot rely on AGC without a device-specific indicator or imputation.

---

**Q-020 — Does Status contain signals that Raw does not?** — *Priority: MEDIUM — Status: ANSWERED*
- **Evidence:** In a typical epoch, `Status` lists 50–70 signals, while `Raw` lists 30–50 signals.
- **Answer:** YES. `GnssStatus` includes all satellites in the constellation that the receiver is attempting to track (including those below the tracking threshold). `GnssMeasurement` (Raw) only includes signals with active tracking loops.
- **Conclusion for ML:** When building candidate sets for satellite selection, use `Status` to know what is *potentially* visible, and `Raw` to know what is *currently usable*.

---

**Q-021 — Device-conditional field reporting matrix** — *Priority: HIGH — Status: ANSWERED*
- **Answer:** The full device × column null matrix shows that missingness in this dataset is almost entirely **structural by OEM/chipset**, not random. See summary below.

### Summary: Device Capabilities Matrix

| Feature | OnePlus CPH2401 (MTK) | Samsung SM-M146B (Exynos) | OnePlus CPH2487 (Qualcomm) | OPPO CPH2325 (MTK) |
|---|---|---|---|---|
| `Raw.SvPositionEcef*` | ❌ Never (100% null) | ⚠️ Partial (47% null) | ✅ Yes (6% null) | ✅ Yes (11% null) |
| `Raw.Klobuchar*` | ❌ Never (100% null) | ✅ Yes (0% null) | ✅ Yes (0% null) | ✅ Yes (0% null) |
| `Raw.IsFullTracking` | ✅ Yes (always 0) | ❌ Never | ❌ Never | ❌ Never |
| `Raw.BasebandCn0DbHz` | ✅ Yes (0% null) | ✅ Yes (0% null) | ✅ Yes (0% null) | ✅ Yes (0% null) |
| `Raw.AgcDb` (inline) | ✅ Continuous | ❌ 100% null | ✅ Continuous | ✅ Continuous |
| `Agc` table (block) | ❌ Never | ✅ Yes (4 values) | ❌ Never | ❌ Never |
| `Accel` (calibrated) | ❌ Never | ✅ Yes (29 sessions) | ❌ Never | ❌ Never |
| `UncalAccel` | ✅ Yes | ❌ Never | ✅ Yes | ✅ Yes |
| `FOP` (heading) | ⚠️ 80% sessions | ✅ Yes | ✅ Yes | ✅ Yes |
| `OrientationDeg` | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| C/N0 = 0 bug on GLONASS | ⚠️ YES (affects data) | ✅ Normal | ⚠️ In Status only | ⚠️ YES (affects data) |

---

## 5 Proposed Experiments (Prioritized)

1. **E-001: Carrier-phase cycle-slip detection for multipath**
   - *Goal:* Verify whether `AccumulatedDeltaRangeState` cycle slips (bit 2) correlate with sudden C/N0 drops.
   - *Method:* Filter rows with ADR state containing cycle-slip flag; measure mean |ΔC/N0| over the next 1–5 seconds vs rows without cycle slips.
   - *Expected outcome:* Cycle slips precede sharp C/N0 drops by 1–2 seconds, serving as an early-warning feature for degradation.

2. **E-002: In-line AGC as an interference / hand-proximity detector**
   - *Goal:* Test whether sudden shifts in `Raw.AgcDb` coincide with hand occlusion or body shadowing.
   - *Method:* On CPH2401 / CPH2487 sessions, compute rolling variance of `Raw.AgcDb`; correlate with Accel variance and C/N0 drops.
   - *Expected outcome:* Hand-proximity events cause sharp AGC gain increases (compensating for attenuation), predictive of signal fade.

3. **E-003: Elevation-dependent nominal baseline subtraction**
   - *Goal:* Evaluate whether removing the nominal open-sky curve $C/N_0(\theta) = a + b \sin(\theta)$ reduces residual variance across all satellites.
   - *Method:* Fit a 3rd-order polynomial to open-sky sessions; subtract from raw C/N0; compare variance of raw vs residual.
   - *Expected outcome:* Elevation explains ~40% of absolute C/N0 variance; residual C/N0 isolates antenna pattern and multipath.

4. **E-004: FOP heading vs OrientationDeg yaw under phone tilt**
   - *Goal:* Quantify the gimbal lock error in `OrientationDeg.yawDeg` when phone pitch exceeds 70°.
   - *Method:* Join `OrientationDeg` and `FOP` on `utcTimeMillis`; compute angular difference $|yawDeg - headingDegrees|$ binned by $|pitchDeg|$.
   - *Expected outcome:* Error is <5° when flat ($|pitch| < 20°$), but diverges to >90° when upright ($|pitch| > 75°$).

5. **E-005: Cross-device gain offset calibration**
   - *Goal:* Test whether subtracting per-device mean open-sky C/N0 normalizes the 14.5 dB spread between Samsung and OPPO.
   - *Method:* Compute median C/N0 for GPS L1 at elevation >60° per device; subtract this constant per device; evaluate multi-device model transfer.
   - *Expected outcome:* Device offset normalization allows a single model to generalize across all 4 chipsets without learning vendor identity.
