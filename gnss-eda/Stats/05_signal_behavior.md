# 05 — Signal Behavior and Temporal Dynamics (Phase F)

This document analyzes GNSS signal behavior across the 95 Android GNSS Logger sessions (5,482,734 `Raw` rows, 6,956,811 `Status` rows). The primary objective is to ground machine learning feature engineering and model evaluation for the task of **forecasting carrier-to-noise density ratio ($C/N_0$) 15 seconds ahead**.

All statistics are derived from DuckDB 2.0 aggregate queries executing under strict resource boundaries (`memory_limit='1GB'`, `threads=2`, `preserve_insertion_order=false`). Reproducible SQL scripts reside in `docs/EDA/queries/031_*.sql` through `037_*.sql`.

---

## 1. $C/N_0$ Distribution Analysis

### 1.1 Overall $C/N_0$ Distribution (Excluding $C/N_0 = 0$)

- **Goal**: Characterize the statistical distribution of carrier-to-noise density ratio across all valid tracked satellite signals.
- **Why**: $C/N_0$ is the direct regression target. We must establish valid range boundaries, central tendencies, and tail behavior, while filtering unmeasured indicator zeros.
- **SQL**: `docs/EDA/queries/031_cn0_distribution_and_constellations.sql` (§1)
```sql
SELECT
  COUNT(*) AS n,
  MIN(Cn0DbHz) AS min,
  PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY Cn0DbHz) AS p1,
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY Cn0DbHz) AS p5,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Cn0DbHz) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY Cn0DbHz) AS p99,
  MAX(Cn0DbHz) AS max,
  AVG(Cn0DbHz) AS mean,
  STDDEV(Cn0DbHz) AS std
FROM read_parquet('scratch/typed/Raw.parquet')
WHERE Cn0DbHz > 0;
```
- **Result** [OBSERVED]:
  - Total valid rows ($C/N_0 > 0$): **5,272,195** (out of 5,482,734 total `Raw` rows; 210,539 rows have $C/N_0 = 0$).
  - **Min**: 1.80 dB-Hz
  - **p1**: 14.28 dB-Hz
  - **p5**: 18.98 dB-Hz
  - **p25**: 27.00 dB-Hz
  - **p50 (Median)**: **32.32 dB-Hz**
  - **p75**: 37.34 dB-Hz
  - **p95**: 44.30 dB-Hz
  - **p99**: 48.80 dB-Hz
  - **Max**: 54.70 dB-Hz
  - **Mean**: **32.07 dB-Hz**
  - **StdDev**: **7.58 dB-Hz**
  - Interquartile Range (IQR): 10.34 dB-Hz (27.00 to 37.34 dB-Hz)
- **Insight**: 
  - Valid smartphone GNSS signals span roughly 12 to 50 dB-Hz. The median of 32.3 dB-Hz reflects typical mobile receiver performance under mixed outdoor/semi-obstructed operating conditions.
  - The 210,539 rows with $C/N_0 = 0$ (3.84% of `Raw`) are concentrated almost entirely in GLONASS (209,783 rows; 99.6% of all zeros). These indicate channel state tracking without an acquired $C/N_0$ lock, rather than true zero signal power.
- **ML Relevance**:
  - **Target Filtering**: Rows with $C/N_0 = 0$ must be excluded from continuous regression loss (MSE/MAE) to prevent predicting non-physical negative or near-zero power drops.
  - **Loss Function Scaling**: The standard deviation of 7.58 dB-Hz indicates that an unnormalized MSE loss will have scale $\approx 57 \text{ dB}^2$. Using MAE or Huber loss ($\delta = 1.5 \text{ dB}$) provides robustness to tracking slips.

---

### 1.2 $C/N_0$ by Constellation

- **Goal**: Quantify signal strength discrepancies across GNSS constellations.
- **Why**: Different satellite constellations operate at different orbital altitudes (MEO vs IGSO/GEO), transmitter powers, signal structures, and elevation geometries over India.
- **SQL**: `docs/EDA/queries/031_cn0_distribution_and_constellations.sql` (§2)
- **Result** [OBSERVED]:

| Constellation ID | Constellation Name | Valid Rows ($N$) | Min | p5 | p25 | p50 (Median) | p75 | p95 | Max | Mean | Std | % Zero in Raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | GPS | 1,320,527 | 5.01 | 19.30 | 27.30 | **32.99** | 38.20 | 45.70 | 54.30 | 32.69 | 7.86 | 0.003% |
| **3** | GLONASS | 385,636 | 9.50 | 20.49 | 28.80 | **34.60** | 40.60 | 49.60 | 54.70 | 34.73 | 8.59 | **35.23%** |
| **4** | QZSS | 125,071 | 5.01 | 17.00 | 23.70 | **29.00** | 34.00 | 42.80 | 51.20 | 29.12 | 7.70 | 0.004% |
| **5** | BeiDou | 2,036,176 | 5.01 | 19.10 | 27.30 | **32.30** | 37.10 | 43.20 | 51.80 | 31.95 | 7.18 | 0.000% |
| **6** | Galileo | 1,219,213 | 1.80 | 18.00 | 25.70 | **31.20** | 36.01 | 43.00 | 51.30 | 30.83 | 7.49 | 0.000% |
| **7** | NavIC (IRNSS) | 185,572 | 10.60 | 24.40 | 30.00 | **33.98** | 37.50 | 40.90 | 49.30 | 33.52 | **5.32** | 0.000% |
| **2** | SBAS | 0 | — | — | — | — | — | — | — | — | — | — |

- **Insight**:
  - **BeiDou** dominates sample count (38.6% of valid `Raw` rows), followed by GPS (25.0%) and Galileo (23.1%).
  - **NavIC (Constellation 7)** displays the lowest standard deviation (5.32 dB-Hz) and highest 5th-percentile (24.40 dB-Hz). Because NavIC satellites are GEO/GSO positioned over the Indian Ocean/subcontinent, their elevation angles remain stable with fewer multipath-induced fades.
  - **GLONASS** valid measurements have the highest median (34.60 dB-Hz), but 35.2% of GLONASS channel rows report $C/N_0 = 0$.
  - **QZSS** has the lowest median (29.00 dB-Hz), consistent with regional orbital geometries at the edge of high-gain coverage in Pune.
  - **SBAS (2)** has 0 rows in `Raw` (SBAS signals appear only in `Status` as unaugmented tracking channels).
- **ML Relevance**:
  - `ConstellationType` is an essential categorical embedding feature.
  - Separate loss weighting or residual baseline calibration per constellation is beneficial because baseline signal strength shifts by up to 5.6 dB-Hz (GLONASS vs QZSS).

---

### 1.3 $C/N_0$ by Carrier Frequency Band

- **Goal**: Group carrier frequencies into designated standard GNSS frequency bands and inspect signal reception quality.
- **Why**: Dual-frequency smartphones record primary bands (L1/E1/B1 at ~1575 MHz) and secondary wideband/civil aviation signals (L5/E5a/B2a at ~1176 MHz). L5 signals have higher chipping rates and greater multipath rejection.
- **SQL**: `docs/EDA/queries/031_cn0_distribution_and_constellations.sql` (§3)
- **Result** [OBSERVED]:

| Band Name | Frequency Range (Hz) | Rows ($N$) | Min | p25 | p50 (Median) | p75 | p95 | Max | Mean | Std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **L1 / E1 / B1** | 1.559 – 1.591 GHz (~1575 MHz) | 2,952,717 | 1.80 | 28.07 | **33.37** | 38.20 | 45.20 | 54.30 | 32.96 | 7.59 |
| **L5 / E5a / B2a** | 1.164 – 1.190 GHz (~1176 MHz) | 1,933,842 | 5.01 | 25.30 | **30.47** | 35.31 | 40.90 | 50.10 | 30.17 | 6.90 |
| **GLONASS L1** | 1.593 – 1.610 GHz (~1600 MHz) | 385,636 | 9.50 | 28.80 | **34.60** | 40.60 | 49.60 | 54.70 | 34.73 | 8.59 |

- **Insight**:
  - The corpus contains 1.93 million secondary band (L5/E5a/B2a) observations (~36.7% of all valid measurements).
  - L5-band signals exhibit a **2.90 dB-Hz lower median** than L1-band signals (30.47 vs 33.37 dB-Hz) and a tighter standard deviation (6.90 vs 7.59 dB-Hz).
  - This 2.9 dB offset is physically expected: mobile internal antenna design and lower transmitted power on certain L5 channels contribute to reduced power at the low-noise amplifier (LNA) port.
- **ML Relevance**:
  - `CarrierFrequencyHz` must be binned into a categorical band feature or normalized against nominal band carrier frequency. Treating frequency as an unscaled continuous variable (~1.575e9) causes gradient instability in neural networks.
  - When forecasting dual-frequency satellite pairs (e.g., GPS PRN 01 L1 and L5 observed simultaneously), the cross-band difference $\Delta C/N_{0, \text{L1-L5}}$ serves as an indicator of multipath decorrelation.

---

### 1.4 $C/N_0$ vs. Baseband $C/N_0$

- **Goal**: Evaluate whether `Cn0DbHz` and `BasebandCn0DbHz` are identical or systematically offset across constellations.
- **Why**: According to Android API documentation [BACKGROUND], `BasebandCn0DbHz` measures carrier-to-noise ratio at the baseband correlator output, whereas `Cn0DbHz` reflects the antenna/RF front-end port before AGC and filtering adjustments.
- **SQL**: `docs/EDA/queries/031_cn0_distribution_and_constellations.sql` (§4)
- **Result** [OBSERVED]:

| Constellation | Total Pairs | Identical ($|\Delta| < 0.01$) | Differ ($|\Delta| \ge 0.01$) | % Differing | Mean Offset ($\text{Cn0} - \text{Baseband}$) | Std Offset | Median Offset | Max Offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GPS (1)** | 1,320,527 | 638,384 | 682,143 | 51.66% | **+2.40 dB** | 2.36 dB | +3.60 dB | +5.00 dB |
| **GLONASS (3)** | 385,636 | 89,673 | 295,963 | **76.75%** | **+3.50 dB** | 1.98 dB | +4.00 dB | +5.00 dB |
| **QZSS (4)** | 125,071 | 68,516 | 56,555 | 45.22% | **+2.13 dB** | 2.38 dB | 0.00 dB | +5.00 dB |
| **BeiDou (5)** | 2,036,176 | 939,429 | 1,096,747 | 53.86% | **+2.74 dB** | 2.61 dB | +4.30 dB | +7.00 dB |
| **Galileo (6)** | 1,219,213 | 521,815 | 697,398 | 57.20% | **+2.81 dB** | 2.45 dB | +4.30 dB | +5.30 dB |
| **NavIC (7)** | 185,572 | 127,104 | 58,468 | 31.51% | **+1.43 dB** | 2.11 dB | 0.00 dB | +5.00 dB |

- **Insight**:
  - `Cn0DbHz` and `BasebandCn0DbHz` are **NOT identical**. In 54.7% of all valid observations across the corpus, they differ.
  - `Cn0DbHz` is **always greater than or equal to `BasebandCn0DbHz`** ($\text{Min Diff} = 0.00 \text{ dB}$, Max Diff up to $+7.00 \text{ dB}$).
  - The difference is quantized by chipsets into distinct steps (e.g. 0.0 dB, 3.6 dB, 4.0 dB, 4.3 dB, 5.0 dB).
  - GLONASS exhibits the highest proportion of differing measurements (76.75%, mean difference +3.50 dB), while NavIC has the lowest (31.51%, mean difference +1.43 dB).
- **ML Relevance**:
  - `Cn0DbHz - BasebandCn0DbHz` captures receiver baseband correlation loss and front-end noise figure adjustments.
  - Including this differential as an input feature provides a direct proxy for correlator tracking stress, serving as a leading indicator of tracking degradation.

---

### 1.5 $C/N_0$ vs. Satellite Elevation Bin

- **Goal**: Measure the relationship between satellite elevation angle ($0^\circ$ to $90^\circ$) and received $C/N_0$.
- **Why**: Satellite signals at lower elevation traverse longer atmospheric path lengths through tropospheric and ionospheric layers and suffer higher local multipath/diffraction from terrain and clutter.
- **SQL**: `docs/EDA/queries/032_cn0_vs_elevation.sql` (Cross-table join between `Raw` and `Status` on timestamp $\pm 500\text{ ms}$, Constellation, and Svid on session `gnss_log_2026_10_23_08_49_01`, cross-verified corpus-wide).
- **Result** [OBSERVED]:

| Elevation Bin | Valid Observations ($N$) | Mean $C/N_0$ (dB-Hz) | p25 (dB-Hz) | Median $C/N_0$ (dB-Hz) | p75 (dB-Hz) | Std (dB-Hz) | % Zero in Status |
|---|---:|---:|---:|---:|---:|---:|---:|
| **$0^\circ – 10^\circ$** | 24,734 | 20.74 | 16.40 | **19.70** | 24.23 | 6.06 | 46.4% |
| **$10^\circ – 20^\circ$** | 57,604 | 25.06 | 19.64 | **24.80** | 30.12 | 6.89 | 13.7% |
| **$20^\circ – 30^\circ$** | 83,364 | 29.34 | 24.80 | **30.00** | 33.83 | 6.71 | 10.7% |
| **$30^\circ – 40^\circ$** | 61,754 | 31.04 | 25.64 | **31.10** | 36.87 | 6.80 | 7.3% |
| **$40^\circ – 50^\circ$** | 62,078 | 30.84 | 26.85 | **31.41** | 35.70 | 6.26 | 8.8% |
| **$50^\circ – 60^\circ$** | 35,850 | 30.40 | 27.01 | **31.14** | 35.20 | 5.66 | 9.2% |
| **$60^\circ – 70^\circ$** | 19,056 | 31.17 | 28.39 | **31.91** | 34.70 | 4.90 | 14.3% |
| **$70^\circ – 80^\circ$** | 13,960 | 29.82 | 27.00 | **30.29** | 33.30 | 5.55 | 12.0% |
| **$80^\circ – 90^\circ$** | 2,827 | 33.84 | 27.90 | **36.52** | 38.06 | 6.06 | 1.8% |

- **Insight**:
  - **Steep rise below $30^\circ$**: Median $C/N_0$ rises sharply from **19.7 dB-Hz** at $0^\circ–10^\circ$ to **30.0 dB-Hz** at $20^\circ–30^\circ$ (+10.3 dB-Hz gain over $20^\circ$ elevation).
  - **Plateau between $30^\circ$ and $75^\circ$**: Median $C/N_0$ plateaus around 30.5 – 31.9 dB-Hz. The mobile antenna gain pattern and satellite transmitter isoflux designs compensate for geometric path loss differences across these intermediate angles.
  - **Zenith Peak**: Above $80^\circ$, median $C/N_0$ jumps to **36.52 dB-Hz**, where line-of-sight obstruction is minimal.
  - **Zero-signal dropouts**: Low-elevation channels exhibit severe tracking outages, with 46.4% reporting $C/N_0 = 0$ in the $0^\circ–10^\circ$ bin.
- **ML Relevance**:
  - Satellite elevation is a strong nonlinear physical predictor of baseline $C/N_0$.
  - Elevation angle alone accounts for $\approx 15 \text{ dB-Hz}$ of deterministic dynamic range between horizon and zenith. A polynomial or spline encoding of $\sin(\text{Elevation})$ provides high feature importance in tree and neural models.

---

### 1.6 $C/N_0$ by CodeType

- **Goal**: Group signal measurements by modulation code type across constellations.
- **Why**: Different code types (e.g. C = civil/standard, Q = pilot/quadrature, I = in-phase data, P = precision/vendor code) have different chipping structures, symbol rates, and tracking thresholds.
- **SQL**: `docs/EDA/queries/031_cn0_distribution_and_constellations.sql` (§5)
- **Result** [OBSERVED]:

| Constellation | CodeType | Rows ($N$) | Mean $C/N_0$ (dB-Hz) | Median $C/N_0$ (dB-Hz) | StdDev (dB-Hz) | Primary Channel Meaning |
|---|:---:|---:|---:|---:|---:|---|
| **GPS (1)** | C | 1,008,275 | 33.86 | 34.28 | 7.81 | L1 C/A (Legacy Standard) |
| **GPS (1)** | Q | 312,252 | 28.93 | 29.50 | 6.78 | L5 Q (Pilot Channel) |
| **GLONASS (3)**| C | 385,636 | 34.73 | 34.60 | 8.59 | G1 C/A (FDMA Standard) |
| **QZSS (4)** | C | 86,251 | 31.30 | 30.50 | 7.50 | L1 C/A |
| **QZSS (4)** | Q | 38,820 | 24.27 | 24.00 | 5.65 | L5 Q |
| **BeiDou (5)** | C | 513,114 | 37.64 | 38.00 | 5.84 | B1I (Reported as C on Samsung) |
| **BeiDou (5)** | I | 859,647 | 30.66 | 31.20 | 6.46 | B1I (Data channel on MTK/QCOM) |
| **BeiDou (5)** | P | 170,573 | 32.82 | 33.30 | 5.18 | B1C (Vendor labeled P on QCOM CPH2487) |
| **BeiDou (5)** | Q | 492,842 | 27.98 | 28.30 | 6.51 | B2a Q (Pilot Channel) |
| **Galileo (6)**| C | 851,402 | 32.03 | 32.50 | 7.54 | E1 Open Service (Data+Pilot) |
| **Galileo (6)**| Q | 367,811 | 28.04 | 28.10 | 6.59 | E5a Q (Pilot Channel) |
| **NavIC (7)** | C | 185,572 | 33.52 | 33.98 | 5.32 | L5 Standard SPS |

- **Insight**:
  - Across all constellations with pilot channels (GPS, QZSS, BeiDou, Galileo), **CodeType 'Q' is consistently 4 to 7 dB-Hz weaker than CodeType 'C'**.
  - All 170,573 rows with `CodeType = 'P'` belong to BeiDou B1C on the Qualcomm CPH2487 device [OBSERVED, STEP-020].
  - Standard deviations are tightest for pilot channels and NavIC (5.18 – 6.78 dB-Hz), while legacy L1 C/A shows wider spread due to varied multipath environments.
- **ML Relevance**:
  - `CodeType` must be paired with `ConstellationType` as a composite signal identity feature.
  - The systematic 5 dB offset between data (C/I) and pilot (Q) channels implies that models predicting across frequencies without code type awareness will produce large bias errors.

---

## 2. Temporal Persistence and Autocorrelation Analysis

### 2.1 The Baseline Hypothesis: Predicting Future = Current

- **Goal**: Evaluate the temporal stability of $C/N_0$ by computing prediction error for the naive persistence baseline:
  $$\hat{y}_{t+h} = y_t$$
  across prediction horizons $h = 1\text{ s}$, $h = 5\text{ s}$, and $h = 15\text{ s}$.
- **Why**: **This is the central baseline for the entire forecasting project.** If a machine learning model (e.g. LSTM, Transformer, LightGBM) cannot outperform $\hat{y}_{t+h} = y_t$ in MAE, RMSE, or tail error, it provides no practical engineering utility.
- **SQL**: `docs/EDA/queries/033_temporal_persistence_autocorrelation.sql`
  - Windowed lagged self-join partitioned by `(session, ConstellationType, Svid, CarrierFrequencyHz)` with time matching tolerance $\pm 500\text{ ms}$ on all valid signals ($C/N_0 > 0$).
- **Result** [OBSERVED]:

#### Corpus-Wide Persistence Metrics (5.27 Million Signal Transitions)

| Metric | Horizon $h = 1\text{ s}$ | Horizon $h = 5\text{ s}$ | Horizon $h = 15\text{ s}$ (Primary Task) |
|---|---:|---:|---:|
| **Sample Pairs ($N$)** | 5,176,278 | 5,036,489 | **4,828,449** |
| **Pearson Correlation ($r$)** | **0.9943** | **0.9866** | **0.9780** |
| **Mean Absolute Error (MAE)** | **0.437 dB-Hz** | **0.715 dB-Hz** | **0.992 dB-Hz** |
| **StdDev of $|\Delta C/N_0|$** | 0.924 dB-Hz | 1.374 dB-Hz | 1.705 dB-Hz |
| **1st Percentile (p1)** | 0.000 dB-Hz | 0.000 dB-Hz | 0.000 dB-Hz |
| **5th Percentile (p5)** | 0.000 dB-Hz | 0.000 dB-Hz | 0.000 dB-Hz |
| **25th Percentile (p25)** | 0.002 dB-Hz | 0.095 dB-Hz | 0.130 dB-Hz |
| **Median $|\Delta C/N_0|$ (p50)** | **0.099 dB-Hz** | **0.200 dB-Hz** | **0.400 dB-Hz** |
| **75th Percentile (p75)** | 0.500 dB-Hz | 0.761 dB-Hz | 1.063 dB-Hz |
| **95th Percentile (p95)** | 1.989 dB-Hz | 3.050 dB-Hz | **3.931 dB-Hz** |
| **99th Percentile (p99)** | 4.375 dB-Hz | 6.906 dB-Hz | 8.783 dB-Hz |
| **Max Absolute Delta** | 41.69 dB-Hz | 41.89 dB-Hz | 40.39 dB-Hz |
| **Fraction with $|\Delta| > 1.5\text{ dB}$** | **7.28%** | **12.75%** | **17.50%** |
| **Fraction with $|\Delta| > 3.0\text{ dB}$** | **2.31%** | **5.10%** | **7.60%** |

- **Insight**:
  - **Very strong overall persistence**: Even at the 15-second horizon, the Pearson correlation remains **0.978**, and the naive persistence MAE is **less than 1.0 dB-Hz** (0.992 dB-Hz).
  - **Dominance of tiny fluctuations**: The median absolute change at 15 seconds is only **0.40 dB-Hz**. 75% of all signals experience less than a **1.06 dB-Hz** shift over 15 seconds.
  - **Heavy-tailed non-Gaussian distribution**: While 82.5% of 15s transitions stay within $\pm 1.5\text{ dB-Hz}$, the 95th percentile reaches 3.93 dB-Hz, and 7.6% of signals jump by more than 3.0 dB-Hz. The maximum jumps reach ~40 dB-Hz, representing sudden line-of-sight blockages (e.g. phone orientation flips, physical masking).
- **ML Relevance**:
  - **Tough Benchmark to Beat**: Any predictive ML model must achieve an overall $\text{MAE} < 0.99 \text{ dB-Hz}$ and $\text{RMSE} < 1.97 \text{ dB-Hz}$ on the test set to justify deployment over persistence.
  - **Target Formulation**: The regression target should be framed as predicting the **differential change** $\Delta C/N_0(t+15) = C/N_0(t+15) - C/N_0(t)$ rather than the raw level $C/N_0(t+15)$. This forces the model to focus capacity on the volatile 17.5% of signals where $|\Delta| > 1.5\text{ dB-Hz}$.
  - **Evaluation Metric Recommendation**: In addition to global MAE, the evaluation suite must report:
    1. $\text{MAE}_{\text{stable}}$ (on instances where $|\Delta_{\text{true}}| \le 1.5\text{ dB}$)
    2. $\text{MAE}_{\text{volatile}}$ (on instances where $|\Delta_{\text{true}}| > 1.5\text{ dB}$)
    3. F1 score or recall on detecting signal drops $> 3.0\text{ dB}$.

---

## 3. Signal Survivorship and Track Continuity

### 3.1 Horizon Survivorship Rates

- **Goal**: Measure the probability that a satellite signal tracked at epoch $t$ is still present in the measurement stream at $t + h$.
- **Why**: A model forecasting $C/N_0$ 15 seconds ahead can only evaluate predictions for signals that survive. Vanishing signals present an imputation/censoring challenge.
- **SQL**: `docs/EDA/queries/034_signal_survivorship_and_tracks.sql` (§1)
- **Result** [OBSERVED]:

| Horizon | Total Signals at $t$ | Still Tracked at $t+h$ | Survivorship Rate (%) | Dropout Rate (%) |
|---|---:|---:|---:|---:|
| **$h = 1\text{ s}$** | 5,482,734 | 5,368,199 | **97.91%** | 2.09% |
| **$h = 5\text{ s}$** | 5,482,734 | 5,263,128 | **95.99%** | 4.01% |
| **$h = 15\text{ s}$** | 5,482,734 | 5,018,952 | **91.54%** | **8.46%** |

- **Insight**:
  - At the 15-second horizon, **8.46% of observed signals disappear** (~1 in 12 signals).
  - Survivorship is nonlinear with time: the dropout rate increases from 2.1% at 1s to 8.5% at 15s.
- **ML Relevance**:
  - When training multi-step or recurrent architectures, training batches must account for track termination without corrupting loss gradients.
  - In production inference, a secondary classification head predicting $P(\text{Signal Drop within 15s})$ is strongly recommended alongside the regression output.

---

### 3.2 Track Length Distribution

- **Goal**: Quantify how long continuous satellite tracking arcs persist within sessions.
- **Why**: Sequence models (CNNs, RNNs, Transformers) require temporal history windows (e.g. 15 to 60 prior epochs). Short or fragmented tracks restrict available context length.
- **SQL**: `docs/EDA/queries/034_signal_survivorship_and_tracks.sql` (§2)
- **Result** [OBSERVED]:

| Quantile / Stat | Track Length (Epochs / Seconds) |
|---|---:|
| **p1** | 3.0 |
| **p5** | 47.0 |
| **p25** | 685.0 |
| **p50 (Median)** | **798.0** |
| **p75** | 894.0 |
| **p95** | 970.0 |
| **p99** | 2,744.5 |
| **Max Track Length** | 3,206.0 |
| **Mean Track Length** | **749.8** |
| **Total Distinct Tracks** | **7,031** |
| Tracks with Length = 1 | 41 (0.58%) |
| Tracks with Length $\ge 15$ | **6,824 (97.06%)** |

- **Insight**:
  - The vast majority of tracks are long and stable: **97.1% of tracks last at least 15 seconds**, and the median track length is **798 epochs (~13.3 minutes)**.
  - Spurious tracks (lasting only 1 epoch) account for less than 0.6% of tracks.
- **ML Relevance**:
  - The dataset easily supports history lookback windows of 15s to 60s for sequence models without losing more than 3% of available training samples.

---

### 3.3 Signal Gaps and Reappearance Behavior

- **Goal**: Determine whether signals drop out momentarily and reappear, or whether dropouts are permanent.
- **Why**: If dropouts are intermittent (e.g. 1-2 missing epochs due to momentary antenna shadowing), models can benefit from state interpolation or Kalman filtering.
- **SQL**: `docs/EDA/queries/034_signal_survivorship_and_tracks.sql` (§3)
- **Result** [OBSERVED, measured on representative high-density session]:
  - **Total signal epoch-to-epoch transitions**: 206,487
  - **Continuous transitions** ($\Delta t \le 1.5\text{ s}$): **205,570 (99.56%)**
  - **Short gaps** ($1.5\text{ s} < \Delta t \le 5.0\text{ s}$): **511 (0.25%)**
  - **Long gaps** ($\Delta t > 5.0\text{ s}$): **406 (0.20%)**
  - **Mean gap duration (when a gap occurs)**: **22.52 seconds**
  - **Maximum observed gap**: 784.99 seconds (~13 minutes)
- **Insight**:
  - Over **99.5%** of consecutive measurements are seamless 1 Hz steps.
  - When a gap does occur, short micro-fades (1.5–5s) are slightly more frequent than macro-fades, but the average outage duration is ~22.5 seconds. Once a signal drops out for several seconds, it typically remains absent until geometric elevation or device orientation changes significantly.
- **ML Relevance**:
  - Gaps $\le 3\text{ seconds}$ can be safely linear-interpolated for feature extraction; gaps $> 5\text{ seconds}$ should be treated as new tracking track segments with reset hidden states.

---

## 4. Per-Epoch Aggregation and Cross-Satellite Variance

### 4.1 Signal Density per Epoch

- **Goal**: Quantify the number of simultaneously tracked valid signals ($C/N_0 > 0$) per 1 Hz epoch.
- **Why**: Defines the graph size or token length for cross-satellite spatial attention models.
- **SQL**: `docs/EDA/queries/035_per_epoch_behavior.sql`
- **Result** [OBSERVED]:
  - Total 1 Hz epochs in corpus: **87,512**
  - **p5**: 27 signals
  - **p25**: 58 signals
  - **p50 (Median)**: **64 signals**
  - **p75**: 68 signals
  - **p95**: 79 signals
  - **Max**: 86 signals
  - **Mean**: **60.25 signals**
- **Insight**:
  - Smartphone multi-GNSS tracking delivers high channel concurrency, with a median of **64 simultaneous signals** per second across up to 6 constellations.

---

### 4.2 Within-Epoch vs. Between-Epoch Variance Decomposition

- **Goal**: Decompose $C/N_0$ variance into within-epoch (spatial across satellites) and between-epoch (temporal over time) components.
- **Why**: Determines whether receiver-wide global factors (e.g. phone orientation, device AGC, user motion) or satellite-specific channel factors (elevation, multipath) dominate signal variability.
- **SQL**: `docs/EDA/queries/035_per_epoch_behavior.sql`
- **Result** [OBSERVED]:

| Variance Dimension | Metric | Observed Value |
|---|---|---:|
| **Epoch-Mean $C/N_0$ Distribution** | 5th Percentile (p5) | 24.60 dB-Hz |
| | 25th Percentile (p25) | 28.20 dB-Hz |
| | **50th Percentile (Median)** | **31.19 dB-Hz** |
| | 75th Percentile (p75) | 33.37 dB-Hz |
| | 95th Percentile (p95) | 39.62 dB-Hz |
| **Between-Epoch Standard Deviation** | $\sigma_{\text{between}}$ | **4.77 dB-Hz** |
| **Mean Within-Epoch Standard Deviation** | $\bar{\sigma}_{\text{within}}$ (Clean $C/N_0 > 0$) | **6.09 dB-Hz** |
| | Uncleaned (with GLONASS zeros) | 10.1 – 10.6 dB-Hz |

- **Insight**:
  - Within any given second, the spread of $C/N_0$ across the sky is substantial ($\bar{\sigma}_{\text{within}} = 6.09 \text{ dB-Hz}$), reflecting elevation angles spanning $5^\circ$ to $85^\circ$ and differing antenna gain facets.
  - Between epochs, the mean sky signal shifts with $\sigma_{\text{between}} = 4.77 \text{ dB-Hz}$. Device rotation or passing under tree canopy shifts all channels concurrently.
- **ML Relevance**:
  - A two-level feature representation is justified:
    1. **Receiver-level context features**: Current epoch mean $C/N_0$, epoch median $C/N_0$, and IMU attitude to capture common-mode environment shifts.
    2. **Satellite-level relative features**: Deviation from epoch mean $(C/N_0 - \overline{C/N_0}_{\text{epoch}})$, which isolates individual satellite multipath and elevation gains.

---

## 5. Status Table Metadata: Usage, Ephemeris, and Leakage Audit

### 5.1 Status Flag Distributions and Constellation Variations

- **Goal**: Inspect `UsedInFix`, `HasAlmanacData`, and `HasEphemerisData` in `Status.parquet` (6,956,811 rows).
- **Why**: Determine whether these fields indicate tracking lock quality, and assess whether they are suitable for predictive feature engineering.
- **SQL**: `docs/EDA/queries/036_status_flags_and_signalcount.sql` (§1 & §2)
- **Result** [OBSERVED]:

| Constellation | Total Rows ($N$) | Fraction Used in Fix | Fraction Has Almanac | Fraction Has Ephemeris |
|---|---:|---:|---:|---:|
| **Corpus Total** | **6,956,811** | **63.42%** | **25.30%** | **42.35%** |
| **GPS (1)** | 1,544,809 | 72.65% | 24.56% | 50.68% |
| **SBAS (2)** | 112,848 | **0.00%** | **0.00%** | **0.00%** |
| **GLONASS (3)** | 726,219 | 69.74% | 23.18% | 41.73% |
| **QZSS (4)** | 219,290 | 37.47% | 30.45% | 13.11% |
| **BeiDou (5)** | 2,578,923 | 64.27% | 28.75% | 41.86% |
| **Galileo (6)** | 1,488,976 | 62.97% | 26.40% | 48.31% |
| **NavIC (7)** | 285,746 | 37.17% | **4.00%** | 11.35% |

- **Insight**:
  - `HasAlmanacData` (25.3%) and `HasEphemerisData` (42.4%) are **NOT constant**; they fluctuate dynamically as navigation data frames are decoded from the broadcast stream.
  - SBAS channels are never used in PVT fixes (0.00%) and carry no local ephemeris.
  - GPS achieves the highest ephemeris broadcast decode rate (50.68%) and fix inclusion rate (72.65%).

---

### 5.2 UsedInFix vs. Satellite Elevation

- **Goal**: Quantify how satellite elevation dictates PVT fix inclusion.
- **SQL**: `docs/EDA/queries/036_status_flags_and_signalcount.sql` (§3)
- **Result** [OBSERVED]:

| Elevation Bin | Rows ($N$) | Fraction Used In Fix | Mean $C/N_0$ (Status) | Median $C/N_0$ |
|---|---:|---:|---:|---:|
| **$< 0^\circ$** | 37 | **0.00%** | 32.32 dB-Hz | 32.0 dB-Hz |
| **$0^\circ – 10^\circ$** | 951,931 | **18.81%** | 10.37 dB-Hz | 0.0 dB-Hz |
| **$10^\circ – 20^\circ$** | 1,218,480 | **59.20%** | 23.70 dB-Hz | 26.1 dB-Hz |
| **$20^\circ – 30^\circ$** | 1,247,147 | **69.94%** | 29.08 dB-Hz | 31.6 dB-Hz |
| **$30^\circ – 40^\circ$** | 1,268,914 | **75.18%** | 30.24 dB-Hz | 32.4 dB-Hz |
| **$40^\circ – 50^\circ$** | 1,086,046 | **73.18%** | 28.99 dB-Hz | 32.2 dB-Hz |
| **$50^\circ – 60^\circ$** | 551,424 | **80.09%** | 30.35 dB-Hz | 32.4 dB-Hz |
| **$60^\circ – 70^\circ$** | 386,876 | **69.96%** | 28.94 dB-Hz | 32.6 dB-Hz |
| **$70^\circ – 80^\circ$** | 210,451 | **69.61%** | 29.07 dB-Hz | 32.0 dB-Hz |
| **$80^\circ – 90^\circ$** | 35,505 | **90.18%** | 31.54 dB-Hz | 33.6 dB-Hz |

- **Insight**: Below $10^\circ$, fewer than 19% of satellites are incorporated into the navigation solution due to elevation mask thresholds and low $C/N_0$. Above $80^\circ$, over 90% are utilized.

---

### 5.3 SignalCount Consistency and Crucial ML Leakage Warning

- **Goal**: Check whether `SignalCount` in `Status` is consistent within an epoch and matches the actual row count.
- **SQL**: `docs/EDA/queries/036_status_flags_and_signalcount.sql` (§4)
- **Result** [OBSERVED]:
  - `SignalCount` is **constant within an epoch across 99.96% of all epochs**.
  - The mean absolute difference between declared `SignalCount` and actual row count per epoch is **0.19 rows** (median actual = 88, median declared = 88).
- **CRITICAL ML LEAKAGE WARNING**:
  - `UsedInFix` and `SignalCount` in the `Status` table are **POST-PVT outputs** generated by the Android GNSS positioning engine *after* computing the navigation filter solution for that epoch.
  - If a model attempting to forecast whether a signal will be strong or usable at $t+15\text{ s}$ utilizes `UsedInFix(t+15)` or `SignalCount(t+15)`, this constitutes **severe target leakage**.
  - **Permissible usage**: Historical flags from time $t$ (e.g. `UsedInFix(t)`) are valid lagged inputs, but future values $t+h$ must never enter the feature matrix.

---

## 6. Automatic Gain Control (AGC) Dynamics

### 6.1 `Agc.parquet` Table Analysis

- **Goal**: Determine whether the standalone `Agc` record block provides meaningful dynamic RF front-end information.
- **SQL**: `docs/EDA/queries/037_agc_behavior.sql` (§1 & §2)
- **Result** [OBSERVED]:
  - Total rows in `Agc.parquet`: **1,555,526** across **29 sessions** (only recorded on Samsung SM-M146B, Platform 13).
  - In all 66 sessions from OnePlus and OPPO, the `Agc` block is **completely absent (0 rows)**.
  - Across the 1.55M rows where present:
    - **Min**: -1.30 dB
    - **p25**: 3.90 dB
    - **p50 (Median)**: **3.90 dB**
    - **p75**: 3.90 dB
    - **Max**: 5.20 dB
    - **Mean**: 4.07 dB ($\sigma = 0.49 \text{ dB}$)
    - Distinct values: Only **4 distinct values** exist corpus-wide: `{-1.3, 2.6, 3.9, 5.2}` dB.
  - Within any single session, the standard deviation is near zero ($\le 0.25 \text{ dB}$).
- **Insight**: The standalone `Agc` table is device-specific (Samsung only) and severely quantized into static hardware gain states.

---

### 6.2 `Raw.parquet` Inline `AgcDb` Column

- **Goal**: Evaluate the inline `AgcDb` column present in `Raw.parquet` across all four phone models.
- **SQL**: `docs/EDA/queries/037_agc_behavior.sql` (§3)
- **Result** [OBSERVED]:
  - Total `Raw` rows: 5,482,734. Non-null: **5,482,734 (100% populated, 0 nulls)**.
  - However, the behavior is sharply **device-conditional**:

| Phone Model | Chipset | `Raw` Rows | Distinct `AgcDb` Values | Range of `AgcDb` | Dynamic Behavior |
|---|---|---:|---:|---|---|
| **CPH2325** | MediaTek | 327,753 | **1** | Constant (3.90 dB) | Static placeholder |
| **CPH2401** | MediaTek | 2,267,707 | **1** | Constant (6.00 dB) | Static placeholder |
| **CPH2487** | Qualcomm | 1,331,748 | **1,787** | **-66.49 dB to +5.19 dB** | **Truly continuous variable** |
| **SM-M146B**| S.LSI (Samsung) | 1,555,526 | **4** | -1.30 dB to +5.20 dB | Quantized stepped |

  - **Correlation with $C/N_0$** on Qualcomm CPH2487:
    $$\text{Pearson } r(\overline{C/N_0}_{\text{epoch}}, \text{AgcDb}_{\text{epoch}}) = \mathbf{-0.119}$$
- **Insight**:
  - Only the **Qualcomm CPH2487 (OnePlus 11R)** actively reports true continuous RF front-end AGC levels (1,787 distinct values spanning down to -66.5 dB).
  - MediaTek chipsets report hardcoded constants (3.9 dB on A55, 6.0 dB on Nord 2T), while Samsung reports four discrete steps.
  - The correlation with epoch-mean $C/N_0$ is weak negative ($-0.119$). AGC adjusts amplifier gain upward when input power is attenuated, but receiver digital signal processing keeps correlation levels decoupled from raw RF gain changes.
- **ML Relevance**:
  - `AgcDb` **cannot be used as a general feature** across all devices without causing catastrophic cross-phone generalization failures. A model trained on Qualcomm will learn weights for negative values (down to -66 dB) that will produce erroneous predictions when encountering MediaTek's constant +6.0 dB.
  - If used, `AgcDb` must either be device-normalized (z-scored per session/device) or restricted to single-device Qualcomm sub-models.

---

## 7. Synthesis and Strategic Machine Learning Guidelines

### 7.1 Preprocessing Decisions
1. **GLONASS Zero Cleaning**: Filter or mask all rows with $C/N_0 = 0$. In GLONASS, 35.2% of channel entries are unassigned zero-placeholders that will distort regression loss surfaces.
2. **Horizon Target Framing**: Frame the regression task as predicting $\Delta C/N_0 = C/N_0(t+15) - C/N_0(t)$. Persistence predicts $\Delta = 0$ with $\text{MAE} = 0.992 \text{ dB-Hz}$. Predicting the delta focuses model loss directly on genuine signal shifts.
3. **Categorical Encodings**: `ConstellationType`, `CodeType`, and Carrier Frequency Band (`L1` vs `L5`) must be one-hot or embedding-encoded together, preserving the physical $\approx 5 \text{ dB-Hz}$ baseline delta between pilot ('Q') and standard ('C') channels.

### 7.2 Key Baselines to Beat
1. **Persistence Baseline**:
   - $h = 1\text{ s}$: $\text{MAE} = 0.437 \text{ dB-Hz}$, $\text{Median} = 0.099 \text{ dB-Hz}$
   - $h = 5\text{ s}$: $\text{MAE} = 0.715 \text{ dB-Hz}$, $\text{Median} = 0.200 \text{ dB-Hz}$
   - $h = 15\text{ s}$: $\text{MAE} = \mathbf{0.992} \text{ dB-Hz}$, $\text{Median} = \mathbf{0.400} \text{ dB-Hz}$, $\text{Std} = 1.705 \text{ dB-Hz}$
2. **Volatile Subset Benchmark**:
   - On the 17.5% of signals where $|\Delta| > 1.5\text{ dB-Hz}$, persistence produces an average error of $3.08 \text{ dB-Hz}$. ML models should be primarily evaluated on their capability to reduce error on this volatile fraction.
