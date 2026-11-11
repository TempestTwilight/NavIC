# 02 — Record type `Agc`

**What it is:** clock + automatic gain control block (1 Hz, per constellation/band).

**Volume [OBSERVED]:** 1,555,526 rows across **29 sessions** (files containing this type); time range 1,792,380,080,999 → 1,793,074,857,996 (UTC ms).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Agc.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `utcTimeMillis`

**What:** UTC wall-clock timestamp (ms) of this clock/AGC event.

**Why it matters:** Calendar-time anchor of the block.

**Use / project relevance:** Joining to Raw epochs by nearest time; session metadata.

**Caveats:** 3.5% null observed — fill forward within the file before joining.

**Status:** OBSERVED IN DATA (3.5% null) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 23,880  *(approx_count_distinct)*
- **min:** 1,792,380,080,999
- **max:** 1,793,074,857,996
- **mean:** 1,792,837,589,487
- **std:** 197,864,152
- **p50:** 1,792,757,897,340
- **p95:** 1,793,073,593,066
- **p99:** 1,793,074,608,700

## `TimeNanos`

**What:** Receiver hardware clock value (ns) at the event — same semantics as Raw.TimeNanos.

**Why it matters:** The block is a 1 Hz copy of the epoch clock block, one row per constellation/band.

**Use / project relevance:** Alignment with Raw epochs (Raw rows carry the same fields per epoch).

**Caveats:** Constant within one event row; huge values (int64!) like Raw.FullBiasNanos.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 26,130  *(approx_count_distinct)*
- **min:** 6,304,594,111
- **max:** 1,501,768,677,317
- **mean:** 469,067,305,141
- **std:** 276,778,841,135
- **p50:** 459,786,651,621
- **p95:** 883,657,779,930
- **p99:** 1,247,684,337,311

## `LeapSecond`

**What:** GPS−UTC leap seconds as reported by the chipset (18 normally; 19 seen in ONE session — anomalous, Q-008).

**Why it matters:** Time conversion; also a fingerprint of chipset firmware behaviour.

**Use / project relevance:** Sanity check; drop as a feature (constant).

**Caveats:** One session reports 19 — a firmware/reporting quirk worth documenting, not a real leap-second change.

**Status:** OBSERVED (19 in one session) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 18
- **max:** 18
- **mean:** 18
- **std:** 0
- **p50:** 18
- **p95:** 18
- **p99:** 18

## `TimeUncertaintyNanos`

**What:** 1-sigma uncertainty of TimeNanos (ns).

**Why it matters:** Clock quality.

**Use / project relevance:** Quality filter.

**Caveats:** Reference doc claims 100% null here — CONFIRMED.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (100.0%)
- **Distinct:** 0  *(exact)*

## `FullBiasNanos`

**What:** Hardware-clock-to-GPS-time offset (ns) — same as Raw.FullBiasNanos.

**Why it matters:** Clock state per 1 Hz event.

**Use / project relevance:** Clock-stability analysis (Phase F) as a quality/behaviour study of the receiver.

**Caveats:** int64; can jump at discontinuities (see next column).

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 25,330  *(approx_count_distinct)*
- **min:** -1,477,109,202,919,905,792
- **max:** -1,476,415,270,226,898,176
- **mean:** -1,476,872,338,421,998,848
- **std:** 197,905,823,353,650
- **p50:** -1,476,792,745,476,541,184
- **p95:** -1,476,415,270,230,662,912
- **p99:** -1,476,415,270,227,731,712
- **negatives:** 1,555,526 (100.0%)

## `BiasNanos`

**What:** Sub-ns correction to FullBiasNanos.

**Why it matters:** Clock precision.

**Use / project relevance:** Positioning math only.

**Caveats:** Marginal for the project.

**Status:** OBSERVED IN DATA + BACKGROUND

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
- **zeros:** 1,555,526 (100.0%)

## `BiasUncertaintyNanos`

**What:** 1-sigma uncertainty of the clock bias (ns).

**Why it matters:** Clock quality.

**Use / project relevance:** Quality filter.

**Caveats:** Check equality with TimeUncertaintyNanos as seen in Raw samples.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 20  *(exact)*
- **min:** 10.15
- **max:** 100.1
- **mean:** 10.23
- **std:** 0.5698
- **p50:** 10.21
- **p95:** 10.52
- **p99:** 10.58

## `DriftNanosPerSecond`

**What:** Clock drift rate (ns/s).

**Why it matters:** Clock frequency error.

**Use / project relevance:** Clock analysis.

**Caveats:** Reference doc claims 100% null here — CONFIRMED.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (100.0%)
- **Distinct:** 0  *(exact)*

## `DriftUncertaintyNanosPerSecond`

**What:** 1-sigma uncertainty of drift (ns/s).

**Why it matters:** Drift quality.

**Use / project relevance:** Dead weight if null.

**Caveats:** Reference doc claims 100% null here — CONFIRMED.

**Status:** OBSERVED (100% null) + REFERENCE

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 1,555,526 (100.0%)
- **Distinct:** 0  *(exact)*

## `HardwareClockDiscontinuityCount`

**What:** Counter of hardware-clock resets; increments at each discontinuity.

**Why it matters:** Segments the clock history.

**Use / project relevance:** Detecting clock jumps within sessions.

**Caveats:** Constant 0 observed — the clock never reset during Samsung sessions.

**Status:** OBSERVED (constant 0) + BACKGROUND

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
- **zeros:** 1,555,526 (100.0%)

## `AgcDb`

**What:** Automatic Gain Control level in dB — the front-end gain the chipset applies to the incoming signal.

**Why it matters:** THE block's namesake. AGC level indirectly measures the total in-band RF power: strong environment (many strong satellites + interference) → lower gain.

**Use / project relevance:** Project: AGC is a known GNSS-quality proxy (e.g. interference/jamming detection literature). As an epoch-level real-time feature it can help C/N0 prediction; only 4 distinct values observed (heavily quantised firmware output) — treat as categorical, Q-005.

**Caveats:** Only present in 29 sessions (Samsung device) — cross-device analyses cannot use it; check whether the value set differs by band (CarrierFrequencyHz column exists here).

**Status:** OBSERVED (4 distinct values) + BACKGROUND (semantics)

**Observed in this corpus:**
- **DuckDB type:** `DOUBLE`; **nulls:** 0 (0.0%)
- **Distinct:** 4  *(exact)*
- **min:** -1.3
- **max:** 5.2
- **mean:** 4.072
- **std:** 0.4941
- **p50:** 3.9
- **p95:** 5.2
- **p99:** 5.2
- **negatives:** 62 (0.0%)

## `CarrierFrequencyHz`

**What:** Carrier frequency (Hz) of the AGC measurement — the block is per band, not per satellite.

**Why it matters:** Ties the AGC value to a band (L1/L5...).

**Use / project relevance:** Per-band AGC feature. Project: real-time available (on the Samsung device).

**Caveats:** 171,223 zero values observed — zeros are 'not reported'; treat as missing.

**Status:** OBSERVED (zeros present) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 16  *(exact)*
- **min:** 0
- **max:** 1,605,374,976
- **mean:** 1,254,594,773
- **std:** 486,756,963
- **p50:** 1,561,097,984
- **p95:** 1,602,199,683
- **p99:** 1,604,560,254
- **zeros:** 177,284 (11.4%)

## `ConstellationType`

**What:** Constellation code (1 GPS, 2 SBAS, ...) of the AGC row.

**Why it matters:** Per-constellation AGC reporting.

**Use / project relevance:** Grouping for AGC analysis.

**Caveats:** Verify value set vs Raw/Status.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 6  *(exact)*
- **min:** 1
- **max:** 7
- **mean:** 3.983
- **std:** 1.996
- **p50:** 5
- **p95:** 6
- **p99:** 7

