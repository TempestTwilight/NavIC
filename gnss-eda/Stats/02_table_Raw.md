# 02 — Record type `Raw`

**What it is:** per-signal GNSS measurement record (one row per tracked signal per ~1 Hz epoch).

**Volume [OBSERVED]:** 5,482,734 rows across **95 sessions** (files containing this type); time range 1,780,543,012,000 → 1,793,074,857,996 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Raw.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp of the whole measurement epoch (ms since 1970-01-01), as the Android app stamped it when the GnssMeasurementsEvent arrived.

**Why it matters:** Anchors every measurement batch to a common calendar time so Raw rows, Status rows, fixes and sensors can be aligned.

**Use / project relevance:** Joining tables; epoch definition; train/test splitting by time. Project: available at prediction time (it is the current time), so it is a legitimate feature (e.g. time-of-day, session). Never use it to 'cheat' by joining to the future.

**Caveats:** Is wall clock: can jump (NTP/user change); monotonicity must be checked. All rows of one epoch share it, so it is NOT a row-unique key.

**Status:** OBSERVED IN DATA (format) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 80,903  *(approx_count_distinct)*
- **min:** 1,780,543,012,000
- **max:** 1,793,074,857,996
- **mean:** 1,791,190,318,136
- **std:** 4,095,122,899
- **p50:** 1,792,745,707,581
- **p95:** 1,793,072,389,251
- **p99:** 1,793,073,994,833

## `TimeNanos`

**What:** The GNSS receiver's internal hardware clock value at the epoch (ns). Not UTC — it is a free-running counter with a large offset from true time.

**Why it matters:** The base of the pseudorange equation: GPS receive time = TimeNanos + TimeOffsetNanos − (FullBiasNanos + BiasNanos).

**Use / project relevance:** Positioning math (compute true receive time per epoch); deriving inter-epoch clock behaviour. Project: needed only for geometry-independent features; C/N0 prediction does not need it, but it can give a stable per-epoch index.

**Caveats:** Must be combined with FullBiasNanos/BiasNanos to become meaningful; do NOT use raw as a time axis. Constant within an epoch.

**Status:** OBSERVED IN DATA (present, constant within epoch) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 90,616  *(approx_count_distinct)*
- **min:** 6,304,594,111
- **max:** 375,929,404,000,000
- **mean:** 53,342,287,005,962
- **std:** 114,663,050,263,185
- **p50:** 1,787,079,805,218
- **p95:** 371,448,295,276,701
- **p99:** 375,236,877,171,820

## `LeapSecond`

**What:** Current GPS−UTC leap second offset (18 s since 2017).

**Why it matters:** Needed to convert GPS system time to UTC precisely.

**Use / project relevance:** Time conversion in positioning; constant in practice → near-zero information for ML. Project: not usable as a feature (constant).

**Caveats:** Constant → the doc says drop; verify.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 2  *(exact)*
- **min:** 18
- **max:** 19
- **mean:** 18
- **std:** 0.04438
- **p50:** 18
- **p95:** 18
- **p99:** 18

## `TimeUncertaintyNanos`

**What:** 1-sigma estimated uncertainty of the receiver clock value TimeNanos (ns).

**Why it matters:** Weights the clock in positioning filters; large values flag bad epochs.

**Use / project relevance:** Quality filter for epochs; weighting in WLS/KF. Project: could be an epoch-level quality feature IF provided in real time — but check null rate first (reference doc says ~57% null).

**Caveats:** Reference doc claims it is 100% null in the Agc block and ~57% null in Raw — verify; nulls may be structural (chipset did not report).

**Status:** FROM REFERENCE DOC ONLY (until profiled)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (28.4%)
- **Distinct:** 40,669  *(approx_count_distinct)*
- **min:** 0
- **max:** 7.041e+05
- **mean:** 107.4
- **std:** 8129
- **p50:** 12.12
- **p95:** 29.95
- **p99:** 38.47
- **zeros:** 1,331,748 (24.3%)

## `FullBiasNanos`

**What:** Offset between the receiver hardware clock and true GPS time at the epoch (ns). Huge negative number (~ −1.46e18 in samples).

**Why it matters:** Converts hardware clock time to GPS time; the core clock-bias term of positioning.

**Use / project relevance:** Positioning math; clock-stability analysis; HardwareClockDiscontinuityCount changes explain jumps. Project: as a feature it encodes the phone clock state, available at prediction time, but it is an epoch-level value — be careful with per-signal leakage of nothing (target is C/N0, unrelated), so it is safe.

**Caveats:** Can change abruptly when the chipset resets its clock (see HardwareClockDiscontinuityCount); magnitude makes float64 lose ns precision — treat as int64 (DuckDB BIGINT).

**Status:** OBSERVED IN DATA (sample values) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 43,185  *(approx_count_distinct)*
- **min:** -1,477,109,202,919,905,792
- **max:** -1,464,577,865,782,000,128
- **mean:** -1,475,172,193,851,284,480
- **std:** 4,076,908,167,849,578
- **p50:** -1,476,701,688,011,211,008
- **p95:** -1,464,609,639,043,009,280
- **p99:** -1,464,578,593,845,452,288
- **negatives:** 5,482,734 (100.0%)

## `BiasNanos`

**What:** Sub-nanosecond correction to FullBiasNanos (double, ns).

**Why it matters:** Refines the clock bias below the integer-ns resolution of FullBiasNanos.

**Use / project relevance:** Precise pseudorange computation. Project: negligible for C/N0 prediction; useful only for positioning-quality side analyses.

**Caveats:** Small by construction; easily swamped by FullBiasNanos if added in float64.

**Status:** BACKGROUND + OBSERVED IN DATA (format)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 19,048  *(approx_count_distinct)*
- **min:** -1
- **max:** 1
- **mean:** 0.05348
- **std:** 0.2814
- **p50:** 0
- **p95:** 0.7175
- **p99:** 0.9448
- **zeros:** 4,150,986 (75.7%)
- **negatives:** 374,261 (6.8%)

## `BiasUncertaintyNanos`

**What:** 1-sigma uncertainty of the clock bias (ns).

**Why it matters:** Quality indicator of the receiver clock estimate.

**Use / project relevance:** Epoch quality weighting. Project: marginal; epoch-level, available in real time, but low value for C/N0 prediction.

**Caveats:** May equal TimeUncertaintyNanos on some chipsets (both were 22.67143834154307 in the first sample rows) — check for perfect correlation.

**Status:** OBSERVED IN DATA (first rows show equality with TimeUncertaintyNanos)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 62,456  *(approx_count_distinct)*
- **min:** 7.197
- **max:** 647.2
- **mean:** 15.36
- **std:** 6.664
- **p50:** 14.4
- **p95:** 28.05
- **p99:** 36.79

## `DriftNanosPerSecond`

**What:** Rate at which the receiver clock drifts relative to GPS time (ns/s).

**Why it matters:** Clock frequency error; used to predict clock state between epochs.

**Use / project relevance:** Positioning; velocity-ish analyses. Project: epoch-level, real-time-available, but no direct link to C/N0; probably drop.

**Caveats:** Reference doc says 100% null in the Agc block; in Raw verify null rate.

**Status:** FROM REFERENCE DOC ONLY (until profiled)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (28.4%)
- **Distinct:** 64,248  *(approx_count_distinct)*
- **min:** -6946
- **max:** 17.57
- **mean:** -3081
- **std:** 2333
- **p50:** -3948
- **p95:** 0.6407
- **p99:** 5.652
- **negatives:** 3,687,946 (67.3%)

## `DriftUncertaintyNanosPerSecond`

**What:** 1-sigma uncertainty of the drift (ns/s).

**Why it matters:** Quality of the drift estimate.

**Use / project relevance:** Same as DriftNanosPerSecond (weighting). Project: marginal.

**Caveats:** Reference doc says 100% null in the Agc block.

**Status:** FROM REFERENCE DOC ONLY (until profiled)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (28.4%)
- **Distinct:** 17,840  *(approx_count_distinct)*
- **min:** 0
- **max:** 36.7
- **mean:** 3.26
- **std:** 4.563
- **p50:** 0
- **p95:** 9.914
- **p99:** 10.76
- **zeros:** 2,595,460 (47.3%)

## `HardwareClockDiscontinuityCount`

**What:** Counter that increments whenever the receiver hardware clock has a discontinuity (reset/jump).

**Why it matters:** Tells you when the clock-bias history is broken — measurements before/after a change cannot share clock extrapolation.

**Use / project relevance:** Segmenting sessions into clock-continuous runs; sanity check for time alignment. Project: a session/segment feature available at prediction time; useful for detecting the 10/22 08:30 sessions' clock jumps. Check observed values and how often it changes.

**Caveats:** Does not say WHEN the jump happened inside the epoch gap; must be detected by comparing consecutive values.

**Status:** OBSERVED IN DATA (present) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 23  *(exact)*
- **min:** 0
- **max:** 36
- **mean:** 3.547
- **std:** 8.065
- **p50:** 0
- **p95:** 24
- **p99:** 35
- **zeros:** 4,150,986 (75.7%)

## `Svid`

**What:** Satellite Vehicle ID within its constellation (GPS PRN 1–32, GLONASS slot 1–24(+), Galileo 1–36, BeiDou 1–63, QZSS 1–10ish, SBAS 120–158, NavIC 1–14).

**Why it matters:** Identifies the physical satellite; together with ConstellationType (+ carrier band) forms the signal identity.

**Use / project relevance:** Track persistence, per-satellite statistics. Project: categorical feature; available at prediction time (the receiver knows which signal it is measuring). Same Svid in different constellations is a different satellite.

**Caveats:** Not globally unique — must be keyed with ConstellationType and CarrierFrequencyHz; GLONASS Svids are slot numbers (frequency channel = slot+…), not PRNs.

**Status:** BACKGROUND + OBSERVED IN DATA

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 51  *(exact)*
- **min:** 1
- **max:** 199
- **mean:** 22.44
- **std:** 28.92
- **p50:** 19
- **p95:** 41
- **p99:** 195

## `TimeOffsetNanos`

**What:** Per-measurement sub-epoch offset added to TimeNanos (ns): measurement time = TimeNanos + TimeOffsetNanos.

**Why it matters:** Gives each signal its own precise timestamp within the batch; usually ~0 in phone logs.

**Use / project relevance:** Sub-epoch timing in precise positioning. Project: negligible; reference doc says constant 0 — verify and drop if so.

**Caveats:** If it is truly constant the column carries no information.

**Status:** FROM REFERENCE DOC ONLY (until profiled)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 5,482,734 (100.0%)

## `State`

**What:** Bitmask of the receiver's tracking-loop sync state for this signal (code lock, bit sync, subframe sync, TOW decoded, msec ambiguity, etc.).

**Why it matters:** Tells how trustworthy ReceivedSvTimeNanos is: without TOW decoded, transmit time is ambiguous by multiples of 1 ms/20 ms/6 s.

**Use / project relevance:** Filtering measurements (keep only TOW_DECODED or TOW_KNOWN); counting 'fully usable' signals. Project: available at prediction time (the receiver reports it live) — legitimate feature; also the key to interpreting C/N0=0 rows (Q-001).

**Caveats:** Bit meanings are constellation-specific (e.g. BDS D2 bits); see queries/004 for verified bit values; vendors may set bits inconsistently.

**Status:** OBSERVED IN DATA (values) + VERIFIED AOSP (bit meanings)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 40  *(exact)*
- **min:** 1
- **max:** 81,935
- **mean:** 1.718e+04
- **std:** 1.783e+04
- **p50:** 16,431
- **p95:** 49,359
- **p99:** 81,935
- **Bitmask decode:** DECODED [OBSERVED, queries/004]: code_lock 96.1%; bit_sync 80.2%; subframe_sync 77.6%; tow_decoded 80.9%; symbol_sync 54.6%; tow_known 65.1%; msec_ambiguous 1.06%; SBAS_SYNC never set (0.00%, consistent with no SBAS in corpus); State=0 never occurs. Top values: 16431 (29.6%), 16399 (14.7%), 15 (10.9%), 22569 (8.6%), 32995 (5.4%, GLONASS), 81935 (4.6%, L5 with 2nd code lock).

## `ReceivedSvTimeNanos`

**What:** Satellite transmit time decoded from the navigation message (ns since start of GPS week — or GLONASS day).

**Why it matters:** Together with receive time gives the pseudorange: rho = (t_rx − t_tx) × c.

**Use / project relevance:** Pseudorange computation; cycle/time-integrity checks (should advance smoothly per epoch). Project: mostly irrelevant to C/N0 prediction, but its UNCERTAINTY (next column) is a good tracking-quality feature available at prediction time.

**Caveats:** Range depends on State; GLONASS is relative to day-of-year and rolls over daily; msec ambiguity when TOW not decoded.

**Status:** BACKGROUND + OBSERVED IN DATA (format)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 5,168,675  *(approx_count_distinct)*
- **min:** 0
- **max:** 536,866,931,423,268
- **mean:** 268,455,922,249,869
- **std:** 180,830,644,682,208
- **p50:** 357,939,647,777,819
- **p95:** 476,617,956,530,473
- **p99:** 535,704,808,969,901
- **zeros:** 15,515 (0.3%)

## `ReceivedSvTimeUncertaintyNanos`

**What:** 1-sigma uncertainty of the decoded satellite transmit time (ns).

**Why it matters:** Direct measure of tracking quality of the code correlation.

**Use / project relevance:** Quality filter (huge values = weak/unlocked signal); weighting. Project: EXCELLENT real-time feature correlated with C/N0 — but check whether it is too tightly coupled to C/N0 (if it is derived from the same correlator power, including it is fine at prediction time since the receiver provides it live; no leakage of the FUTURE C/N0, only current-state info).

**Caveats:** Vendor-dependent scaling; can be exactly 0 on some chipsets for strong signals.

**Status:** BACKGROUND (semantics) + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 141  *(approx_count_distinct)*
- **min:** 2
- **max:** 5,286,634
- **mean:** 378.3
- **std:** 3.675e+04
- **p50:** 32
- **p95:** 487
- **p99:** 797

## `Cn0DbHz`

**What:** Carrier-to-noise density ratio (dB-Hz) — the project's TARGET. How strong the received signal is relative to the noise floor, measured at the antenna input.

**Why it matters:** Primary signal-strength metric; drives which satellites are usable.

**Use / project relevance:** Quality filtering, NLOS/multipath detection (with elevation), and the 15-s-ahead prediction target. Project: THE target; make sure 'best signal' selection uses predicted values consistently.

**Caveats:** 0 dB-Hz rows exist (see State analysis); typical range 10–50 per Android docs; different vendors report slightly differently (MTK vs Qualcomm vs Exynos).

**Status:** OBSERVED IN DATA + VERIFIED AOSP (definition/range)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 37,611  *(approx_count_distinct)*
- **min:** 0
- **max:** 54.7
- **mean:** 30.84
- **std:** 9.653
- **p50:** 31.94
- **p95:** 44.2
- **p99:** 48.7
- **zeros:** 210,539 (3.8%)

## `PseudorangeRateMetersPerSecond`

**What:** Rate of change of the pseudorange (m/s), from Doppler; uncorrected for clock drifts; positive = satellite receding.

**Why it matters:** Gives receiver velocity information and satellite motion sense.

**Use / project relevance:** Doppler-derived features; velocity estimation; detecting static sessions. Project: available at prediction time; its UNCERTAINTY is a decent quality proxy.

**Caveats:** Sign convention is 'positive = away'; includes receiver clock drift unless corrected.

**Status:** BACKGROUND + OBSERVED IN DATA (format)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 4,763,937  *(approx_count_distinct)*
- **min:** -2860
- **max:** 820.9
- **mean:** -892.7
- **std:** 686.8
- **p50:** -994.8
- **p95:** 268.5
- **p99:** 541.8
- **zeros:** 177,337 (3.2%)
- **negatives:** 4,649,518 (84.8%)

## `PseudorangeRateUncertaintyMetersPerSecond`

**What:** 1-sigma uncertainty of the pseudorange rate (m/s).

**Why it matters:** Quality of the Doppler measurement (tracking-loop noise).

**Use / project relevance:** Quality feature/filter. Project: good real-time feature for C/N0 prediction (weak signal → jittery Doppler).

**Caveats:** Vendor-dependent; correlates with C/N0.

**Status:** BACKGROUND + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 17,512  *(approx_count_distinct)*
- **min:** 0.004822
- **max:** 147
- **mean:** 0.2107
- **std:** 0.8453
- **p50:** 0.15
- **p95:** 0.7116
- **p99:** 2.027

## `AccumulatedDeltaRangeState`

**What:** Bitmask describing the carrier-phase (ADR) measurement state: VALID (bit0), RESET (bit1), CYCLE_SLIP (bit2), HALF_CYCLE_RESOLVED (bit3), HALF_CYCLE_REPORTED (bit4).

**Why it matters:** ADR is only meaningful when valid and without reset/slip.

**Use / project relevance:** Selecting rows for carrier-phase smoothing (Hatch filter); measuring cycle-slip frequency. Project: mostly positioning-side; for C/N0 prediction ADRUncertainty could act as a quality feature, but ADR is often unavailable (check null rate).

**Caveats:** Must check bits, not the raw integer; VALID alone is not enough (doc's §2.4 says 'only use rows where valid' but per AOSP you must ALSO require RESET=0 and CYCLE_SLIP=0 — the reference doc's simplification is wrong for delta-range use).

**Status:** VERIFIED AOSP (bits) + OBSERVED IN DATA (frequencies)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 7  *(exact)*
- **min:** 16
- **max:** 27
- **mean:** 19.06
- **std:** 4.187
- **p50:** 16
- **p95:** 25
- **p99:** 25
- **Bitmask decode:** DECODED [OBSERVED, queries/005]: HALF_CYCLE_REPORTED set on 100% of rows; VALID 39.4%; HALF_CYCLE_RESOLVED 32.9%; RESET 0.78%; CYCLE_SLIP 0.38%; no value >31; ADR-state 0 never occurs (min = 16).

## `AccumulatedDeltaRangeMeters`

**What:** Accumulated carrier-phase change since last reset, in meters (ambiguous by an unknown integer number of wavelengths).

**Why it matters:** cm-precision differential measurement (the raw material of RTK/PPP).

**Use / project relevance:** Carrier smoothing of pseudoranges; high-precision velocity. Project: marginal for C/N0; could indicate tracking continuity.

**Caveats:** Ambiguity resets; sentinel values when invalid (e.g. 0 with state UNKNOWN); huge magnitudes for L5.

**Status:** BACKGROUND + OBSERVED IN DATA (format)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 2,347,218  *(approx_count_distinct)*
- **min:** -27,471,896
- **max:** 5.197e+05
- **mean:** -8.664e+05
- **std:** 1,963,204
- **p50:** 0
- **p95:** 0
- **p99:** 2.874e+05
- **zeros:** 3,053,260 (55.7%)
- **negatives:** 2,285,487 (41.7%)

## `AccumulatedDeltaRangeUncertaintyMeters`

**What:** 1-sigma uncertainty of ADR (m).

**Why it matters:** Tells whether carrier phase is usable (cm-level) or not (m-level).

**Use / project relevance:** Quality gating of carrier-phase use. Project: real-time quality feature if present.

**Caveats:** When ADR invalid, vendors may report a huge number or 0.

**Status:** BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1,265  *(approx_count_distinct)*
- **min:** 0
- **max:** 70.71
- **mean:** 0.02832
- **std:** 0.4944
- **p50:** 0.000225
- **p95:** 0.1158
- **p99:** 0.2271
- **zeros:** 2,605,754 (47.5%)

## `CarrierFrequencyHz`

**What:** Center frequency of the tracked signal in Hz (1575420000 = GPS L1, 1176450000 = L5, 1602000000+ = GLONASS channels, 1561098000 = E1, 1207140000 = E5b, 1176450000 = E5a/B2a, etc.).

**Why it matters:** Identifies the RF band; L5 signals are typically stronger and less prone to multipath than L1.

**Use / project relevance:** Band-level features; C/N0 comparison across bands. Project: KEY categorical feature (L1 vs L5 behaviour differs); available at prediction time. Also a join key with Status.

**Caveats:** Float32 precision in the Android API — values like 1575420000 are exact; GLONASS has many channel values (~15 unique across corpus per doc).

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 15  *(exact)*
- **min:** 1,176,450,000
- **max:** 1,605,375,000
- **mean:** 1,434,450,675
- **std:** 190,779,204
- **p50:** 1,561,098,000
- **p95:** 1,602,500,325
- **p99:** 1,604,689,420

## `CarrierCycles`

**What:** Deprecated legacy field: number of full carrier cycles between satellite and receiver.

**Why it matters:** Predecessor of ADR.

**Use / project relevance:** None in modern data.

**Caveats:** Reference doc says 100% null — verify and drop.

**Status:** FROM REFERENCE DOC ONLY

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 5,482,734 (100.0%)
- **Distinct:** 0  *(exact)*

## `CarrierPhase`

**What:** Deprecated legacy field: fractional RF phase [0,1).

**Why it matters:** Predecessor of ADR.

**Use / project relevance:** None.

**Caveats:** Reference doc says 100% null — verify and drop.

**Status:** FROM REFERENCE DOC ONLY

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 5,482,734 (100.0%)
- **Distinct:** 0  *(exact)*

## `CarrierPhaseUncertainty`

**What:** Deprecated legacy field: phase uncertainty in cycles.

**Why it matters:** Predecessor of ADR uncertainty.

**Use / project relevance:** None.

**Caveats:** Reference doc says 100% null — verify and drop.

**Status:** FROM REFERENCE DOC ONLY

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 5,482,734 (100.0%)
- **Distinct:** 0  *(exact)*

## `MultipathIndicator`

**What:** Vendor's opinion on multipath: 0 = UNKNOWN, 1 = DETECTED, 2 = NOT_DETECTED (AOSP verified).

**Why it matters:** Flags reflected signals directly when the vendor implements detection.

**Use / project relevance:** NLOS filtering. Project: would be a great real-time feature IF populated; reference doc says constant (UNKNOWN) on this corpus — verify; if constant, drop and rely on C/N0+elevation proxy.

**Caveats:** Most chipsets never set anything but UNKNOWN; the doc's claim needs a bit-frequency check.

**Status:** FROM REFERENCE DOC ONLY (until profiled)

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
- **zeros:** 5,482,734 (100.0%)

## `SnrInDb`

**What:** Post-correlation signal-to-noise ratio in dB (needs the HAS_SNR flag to be valid).

**Why it matters:** Alternative/legacy strength measure, related to C/N0 by integration-time-dependent offset.

**Use / project relevance:** Secondary strength metric; cross-check C/N0. Project: redundancy with C/N0 — do not use both naively (collinearity).

**Caveats:** Only valid when the SNR flag bit is set; on many chipsets it mirrors C/N0 minus a constant.

**Status:** BACKGROUND + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (28.4%)
- **Distinct:** 95,325  *(approx_count_distinct)*
- **min:** 0
- **max:** 32.06
- **mean:** 7.277
- **std:** 8.017
- **p50:** 4.499
- **p95:** 22.05
- **p99:** 25.33
- **zeros:** 1,719,493 (31.4%)

## `ConstellationType`

**What:** Which GNSS system the signal belongs to: 1 GPS, 2 SBAS, 3 GLONASS, 4 QZSS, 5 BeiDou, 6 Galileo, 7 IRNSS/NavIC (codes per reference doc; verify against AOSP GnssStatus).

**Why it matters:** Constellations differ in signal plans, geometry, and reporting quirks.

**Use / project relevance:** Grouping/stratifying analyses; categorical feature. Project: important categorical input; available at prediction time.

**Caveats:** Must be combined with Svid; QZSS shares GPS PRNs (Svid 193+ in some conventions on Android QZSS uses 183–200?).

**Status:** FROM REFERENCE DOC ONLY (codes) + OBSERVED IN DATA (values seen)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6  *(exact)*
- **min:** 1
- **max:** 7
- **mean:** 4.087
- **std:** 1.958
- **p50:** 5
- **p95:** 6
- **p99:** 7

## `AgcDb`

**What:** Automatic Gain Control level in dB per measurement (per signal/constellation): how much the RF front-end amplifies the input. More negative = stronger input (or jamming).

**Why it matters:** Indicates interference/jamming and how hot the receiver input is; complements C/N0.

**Use / project relevance:** Interference detection; environment characterization. Project: real-time available; could be a strong feature IF it varies meaningfully; reference doc/data will tell whether this chipset reports it per measurement (only 29 files have the separate Agc block).

**Caveats:** Deprecated in newer Android APIs (moved to GnssAutomaticGainControls); vendor-implementation-specific scaling; only meaningful relatively over time.

**Status:** VERIFIED AOSP (semantics) + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 1,758  *(approx_count_distinct)*
- **min:** -66.49
- **max:** 6
- **mean:** -8.785
- **std:** 25.15
- **p50:** 5.138
- **p95:** 6
- **p99:** 6
- **negatives:** 1,331,810 (24.3%)

## `BasebandCn0DbHz`

**What:** C/N0 measured at the baseband stage rather than the antenna input; per AOSP 'typically a few dB weaker than C/N0 at the antenna port'.

**Why it matters:** Alternative strength estimate; comparing the two exposes receiver-chain behaviour.

**Use / project relevance:** Cross-check C/N0; interference/AGC studies. Project: real-time available; useful as auxiliary target-related feature, but check whether it is identical to Cn0DbHz on this chipset (doc suggests measuring both).

**Caveats:** Only valid when HAS_BASEBAND_CN0 flag is set; may be 0 or NaN otherwise.

**Status:** VERIFIED AOSP (semantics) + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 37,618  *(approx_count_distinct)*
- **min:** 0.01
- **max:** 49.7
- **mean:** 29.44
- **std:** 7.041
- **p50:** 29.77
- **p95:** 40.83
- **p99:** 44.04

## `FullInterSignalBiasNanos`

**What:** Receiver-side hardware bias between signals (ns), full ISB including clock bias contribution.

**Why it matters:** Corrects inter-frequency/inter-signal hardware delays for multi-frequency positioning.

**Use / project relevance:** Precise positioning corrections. Project: negligible for C/N0 prediction.

**Caveats:** Only valid when the HAS_FULL_ISB flag is set; often empty.

**Status:** BACKGROUND + OBSERVED IN DATA (to confirm)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 111,428  *(approx_count_distinct)*
- **min:** -1592
- **max:** 3804
- **mean:** -156.2
- **std:** 822.9
- **p50:** 0
- **p95:** 0.001013
- **p99:** 3781
- **zeros:** 4,324,131 (78.9%)
- **negatives:** 944,729 (17.2%)

## `FullInterSignalBiasUncertaintyNanos`

**What:** 1-sigma uncertainty of FullInterSignalBiasNanos.

**Why it matters:** Quality of the ISB estimate.

**Use / project relevance:** Same as above.

**Caveats:** Flag-dependent availability.

**Status:** BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 68,521 (1.2%)
- **Distinct:** 133,974  *(approx_count_distinct)*
- **min:** 0
- **max:** 946.3
- **mean:** 4.022
- **std:** 10.65
- **p50:** 0
- **p95:** 33.14
- **p99:** 38.72
- **zeros:** 4,255,610 (77.6%)

## `SatelliteInterSignalBiasNanos`

**What:** Satellite-side inter-signal bias (ns) as broadcast (e.g. B2a vs B1 timing offset).

**Why it matters:** Corrects satellite hardware delay differences between signals.

**Use / project relevance:** Multi-frequency positioning. Project: negligible for C/N0.

**Caveats:** Flag-dependent; broadcast-dependent.

**Status:** BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,331,748 (24.3%)
- **Distinct:** 163,394  *(approx_count_distinct)*
- **min:** -104.7
- **max:** 174.4
- **mean:** 2.634
- **std:** 36.58
- **p50:** 0
- **p95:** 89.28
- **p99:** 149.5
- **zeros:** 2,262,466 (41.3%)
- **negatives:** 1,300,892 (23.7%)

## `SatelliteInterSignalBiasUncertaintyNanos`

**What:** 1-sigma uncertainty of SatelliteInterSignalBiasNanos.

**Why it matters:** Quality of the satellite ISB estimate.

**Use / project relevance:** Same as above.

**Caveats:** Flag-dependent.

**Status:** BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,400,269 (25.5%)
- **Distinct:** 2  *(exact)*
- **min:** 0
- **max:** 66.71
- **mean:** 41.29
- **std:** 32.4
- **p50:** 66.71
- **p95:** 66.71
- **p99:** 66.71
- **zeros:** 1,555,526 (28.4%)

## `CodeType`

**What:** String identifying which code is tracked on the frequency: 'C' = C/A (L1 civil), 'I'/'Q' = L5 in-phase/quadrature, 'L' = L1C pilot?, vendor-specific strings possible.

**Why it matters:** Disambiguates multiple signals on the SAME frequency — two rows can share Svid+CarrierFrequencyHz but differ in CodeType.

**Use / project relevance:** Signal-level stratification; join key with Status. Project: categorical feature available at prediction time; crucial for understanding duplicate signals per epoch (Q-002).

**Caveats:** Android does not fully standardize the strings; empty on some chipsets; QZSS/GPS share frequencies.

**Status:** BACKGROUND + OBSERVED IN DATA (values seen: 'C')

**Observed in this corpus:**
- **DuckDB type:** `VARCHAR`; **nulls:** 0 (0.0%)
- **Distinct:** 4  *(exact)*
- **Top values:** `C` = 3,240,066, `Q` = 1,211,789, `I` = 860,306, `P` = 170,573

## `ChipsetElapsedRealtimeNanos`

**What:** Time-since-boot as reported by the GNSS chipset itself (ns), independent from the Android elapsedRealtimeNanos.

**Why it matters:** Lets you check clock continuity inside the chipset and align with Android time.

**Use / project relevance:** Clock-jump detection; aligning chipset events. Project: minor.

**Caveats:** Often empty (verify); if present, drift vs elapsedRealtimeNanos can be checked.

**Status:** BACKGROUND + OBSERVED IN DATA (first rows empty)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 2,595,460 (47.3%)
- **Distinct:** 44,762  *(approx_count_distinct)*
- **min:** 138,751,079,737
- **max:** 692,153,325,027,042
- **mean:** 366,010,800,974,665
- **std:** 207,299,428,561,193
- **p50:** 372,438,340,457,166
- **p95:** 689,796,270,079,778
- **p99:** 691,701,627,799,976

## `IsFullTracking`

**What:** Boolean: receiver tracked this signal in full-tracking mode (continuous) vs duty-cycled to save power.

**Why it matters:** Duty-cycled measurements have degraded quality (fewer correlator samples).

**Use / project relevance:** Quality filter. Project: real-time available; likely constant per device — check; if constant, drop.

**Caveats:** Vendor-dependent; may always be 0 or 1 on some chipsets.

**Status:** BACKGROUND + OBSERVED IN DATA (first rows 0)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 3,215,027 (58.6%)
- **Distinct:** 1  *(exact)*
- **min:** 0
- **max:** 0
- **mean:** 0
- **std:** 0
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 2,267,707 (41.4%)

## `SvPositionEcefXMeters`

**What:** Satellite position X in ECEF (m) at transmit time, computed by the chipset from its ephemeris.

**Why it matters:** Needed for trilateration without your own ephemeris math.

**Use / project relevance:** Positioning; computing satellite elevation/azimuth from user position. Project: could let you derive geometric elevation per signal WITHOUT Status — useful for the orientation-vs-C/N0 analysis when Status is missing; real-time available (chipset provides it).

**Caveats:** Reference doc says ~57% null (structural: chipsets only report it sometimes); coordinates are at transmit time (Sagnac etc. apply); accuracy is chipset-dependent.

**Status:** FROM REFERENCE DOC ONLY (null claim) + OBSERVED IN DATA (empty in first rows)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,293,536  *(approx_count_distinct)*
- **min:** -32,327,879
- **max:** 28,916,487
- **mean:** 2,509,265
- **std:** 14,199,810
- **p50:** 2,373,388
- **p95:** 23,665,834
- **p99:** 26,375,676
- **negatives:** 1,077,415 (19.7%)

## `SvPositionEcefYMeters`

**What:** Satellite position Y in ECEF (m). See X.

**Why it matters:** See X.

**Use / project relevance:** See X.

**Caveats:** See X.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,232,419  *(approx_count_distinct)*
- **min:** -4,291,963
- **max:** 41,503,319
- **mean:** 18,798,859
- **std:** 8,948,362
- **p50:** 18,356,955
- **p95:** 35,463,005
- **p99:** 39,465,959
- **negatives:** 30,856 (0.6%)

## `SvPositionEcefZMeters`

**What:** Satellite position Z in ECEF (m). See X.

**Why it matters:** See X.

**Use / project relevance:** See X.

**Caveats:** See X.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,443,178  *(approx_count_distinct)*
- **min:** -24,472,946
- **max:** 31,858,536
- **mean:** 8,072,575
- **std:** 13,238,221
- **p50:** 11,431,922
- **p95:** 23,410,042
- **p99:** 27,166,200
- **negatives:** 662,518 (12.1%)

## `SvVelocityEcefXMetersPerSecond`

**What:** Satellite velocity X in ECEF (m/s).

**Why it matters:** Doppler prediction and coordinate propagation.

**Use / project relevance:** Velocity-related computations. Project: marginal.

**Caveats:** Same null structure as position columns.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,575,271  *(approx_count_distinct)*
- **min:** -3145
- **max:** 2514
- **mean:** -568.5
- **std:** 1156
- **p50:** -321.2
- **p95:** 1089
- **p99:** 1675
- **negatives:** 1,522,547 (27.8%)

## `SvVelocityEcefYMetersPerSecond`

**What:** Satellite velocity Y in ECEF (m/s).

**Why it matters:** See X-velocity.

**Use / project relevance:** See X-velocity.

**Caveats:** See X-velocity.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,350,231  *(approx_count_distinct)*
- **min:** -3036
- **max:** 2741
- **mean:** 221.3
- **std:** 1161
- **p50:** 321
- **p95:** 1991
- **p99:** 2439
- **negatives:** 955,319 (17.4%)

## `SvVelocityEcefZMetersPerSecond`

**What:** Satellite velocity Z in ECEF (m/s).

**Why it matters:** See X-velocity.

**Use / project relevance:** See X-velocity.

**Caveats:** See X-velocity.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,493,038  *(approx_count_distinct)*
- **min:** -3604
- **max:** 3591
- **mean:** -215.7
- **std:** 2328
- **p50:** -458
- **p95:** 3056
- **p99:** 3306
- **negatives:** 1,291,186 (23.6%)

## `SvClockBiasMeters`

**What:** Satellite clock error expressed in metres (c × Δt_sv).

**Why it matters:** Must be removed from the pseudorange for a true geometric range.

**Use / project relevance:** Positioning corrections. Project: marginal.

**Caveats:** Same null structure; sign convention per ICD.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 2,137,163  *(approx_count_distinct)*
- **min:** -4.826e+05
- **max:** 1,760,068
- **mean:** -2.787e+04
- **std:** 2.041e+05
- **p50:** -9652
- **p95:** 2.22e+05
- **p99:** 2.855e+05
- **negatives:** 1,303,270 (23.8%)

## `SvClockDriftMetersPerSecond`

**What:** Rate of change of the satellite clock error (m/s).

**Why it matters:** Propagates satellite clock between epochs.

**Use / project relevance:** Positioning corrections. Project: marginal.

**Caveats:** Same null structure.

**Status:** See SvPositionEcefXMeters.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 3,117,297 (56.9%)
- **Distinct:** 1,577,000  *(approx_count_distinct)*
- **min:** -0.01194
- **max:** 0.02484
- **mean:** -0.0003647
- **std:** 0.003559
- **p50:** -4.79e-05
- **p95:** 0.003605
- **p99:** 0.006552
- **zeros:** 99,209 (1.8%)
- **negatives:** 1,245,737 (22.7%)

## `KlobucharAlpha0`

**What:** First Klobuchar ionospheric model alpha coefficient (broadcast by GPS satellites).

**Why it matters:** Lets the receiver estimate ionospheric delay (the alpha/beta polynomial).

**Use / project relevance:** Ionospheric correction; also a crude 'time of day / season' marker since alpha/beta are uploaded daily. Project: broadcast-level data, available at prediction time, but constant within sessions — near-zero ML value; mostly a data-quality marker.

**Caveats:** Reference doc says ~41% null (only when the chipset reports the nav data); same for beta terms.

**Status:** BACKGROUND (semantics) + FROM REFERENCE DOC ONLY (null claim)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 3  *(exact)*
- **min:** 1.676e-08
- **max:** 2.515e-08
- **mean:** 1.828e-08
- **std:** 2.381e-09
- **p50:** 1.676e-08
- **p95:** 2.515e-08
- **p99:** 2.515e-08

## `KlobucharAlpha1`

**What:** Second Klobuchar alpha coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 2  *(exact)*
- **min:** -7.451e-09
- **max:** 0
- **mean:** -2.48e-09
- **std:** 3.511e-09
- **p50:** 0
- **p95:** 0
- **p99:** 0
- **zeros:** 2,144,800 (39.1%)
- **negatives:** 1,070,227 (19.5%)

## `KlobucharAlpha2`

**What:** Third Klobuchar alpha coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 1  *(exact)*
- **min:** -1.192e-07
- **max:** -1.192e-07
- **mean:** -1.192e-07
- **std:** 0
- **p50:** -1.192e-07
- **p95:** -1.192e-07
- **p99:** -1.192e-07
- **negatives:** 3,215,027 (58.6%)

## `KlobucharAlpha3`

**What:** Fourth Klobuchar alpha coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 2  *(exact)*
- **min:** 5.96e-08
- **max:** 1.192e-07
- **mean:** 6.315e-08
- **std:** 1.409e-08
- **p50:** 5.96e-08
- **p95:** 1.192e-07
- **p99:** 1.192e-07

## `KlobucharBeta0`

**What:** First Klobuchar beta coefficient (period terms of the iono model).

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 3  *(exact)*
- **min:** 114,688
- **max:** 131,072
- **mean:** 1.185e+05
- **std:** 5512
- **p50:** 114,688
- **p95:** 131,072
- **p99:** 131,072

## `KlobucharBeta1`

**What:** Second Klobuchar beta coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 2  *(exact)*
- **min:** -32,768
- **max:** -16,384
- **mean:** -2.086e+04
- **std:** 7303
- **p50:** -16,384
- **p95:** -16,384
- **p99:** -16,384
- **negatives:** 3,215,027 (58.6%)

## `KlobucharBeta2`

**What:** Third Klobuchar beta coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 3  *(exact)*
- **min:** -327,680
- **max:** -196,608
- **mean:** -3.02e+05
- **std:** 3.917e+04
- **p50:** -327,680
- **p95:** -196,608
- **p99:** -196,608
- **negatives:** 3,215,027 (58.6%)

## `KlobucharBeta3`

**What:** Fourth Klobuchar beta coefficient. See Alpha0.

**Why it matters:** See Alpha0.

**Use / project relevance:** See Alpha0.

**Caveats:** See Alpha0.

**Status:** See KlobucharAlpha0.

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 2,267,707 (41.4%)
- **Distinct:** 2  *(exact)*
- **min:** -65,536
- **max:** 131,072
- **mean:** 1.194e+05
- **std:** 4.649e+04
- **p50:** 131,072
- **p95:** 131,072
- **p99:** 131,072
- **negatives:** 191,099 (3.5%)

