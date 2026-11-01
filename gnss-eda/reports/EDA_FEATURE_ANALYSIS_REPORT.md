# GNSS Feature Analysis & Exploratory Data Analysis Report
**Project:** ISRO GNSS Orientation & Temporal Optimization Pipeline  
**Dataset Path:** `/home/tempest/Documents/ISRO_Project/gnss-eda/data/parquet/`  
**Analysis Engine:** DuckDB (Zero-RAM streaming execution, < 2GB RAM budget)  
**Reference Document:** `/home/tempest/Documents/ISRO_Project_physics_features/PHYSICS_FEATURES_SUMMARY.md`  
**Date:** September 2026  

---

## Executive Summary

This report delivers a comprehensive Exploratory Data Analysis (EDA) of 6,956,811 GNSS measurements across 14 orientation configurations, 47,888 distinct tracking epochs, and multiple satellite constellations (BeiDou, GPS, Galileo, GLONASS, NavIC, QZSS, SBAS). All analysis was executed directly in **DuckDB** using out-of-core streaming queries without loading raw CSV or Parquet files into memory, guaranteeing safety on 8GB (2-3GB usable) hardware.

The end-to-end operational loop consists of two distinct machine learning models working in continuous closed-loop feedback:
1. **Model 1: Temporal Signal Prediction Model (`02_future_model.py`)**: Continuously monitors the live GNSS stream and predicts whether signal strength ($C/N_0$) will degrade, remain stable, or improve over horizons of $1\text{s}$ to $60\text{s}$. It acts as the trigger every $30\text{--}45\text{s}$. If degradation or stalemate is forecast, it kicks off Model 2.
2. **Model 2: Best Orientation Recommendation Model (`02_orientation_model.py`)**: Evaluates hypothetical candidate phone orientations ($\psi_{\text{cand}} \in [0^\circ, 360^\circ)$) against the currently visible satellite constellation and recommends the orientation that maximizes received signal power across the sky.

### Key Conclusions & Core Directives

1. **Elimination of Single-Device Rig Bias:** The legacy feature set contained categorical variables (`facing="up/down"`, `phone_roll="horizontal/vertical"`, `tilt_angle`, and raw `pitch`/`roll`). Because all training data originated from a single OnePlus device, these features encode OnePlus-specific antenna placement, ground-plane shielding, and test-jig mount geometry. They **must be completely removed**.
2. **Adoption of 3D Vector Incidence Physics:** Replaced body angles with the 3D spatial dot product between the satellite Line-of-Sight (LOS) unit vector $\mathbf{s}_{\text{enu}}$ and the antenna normal boresight unit vector $\mathbf{n}_{\text{enu}}$. Empirical DuckDB analysis proves that moving from negative incidence (chassis-shielded) to $\cos\theta_{\text{inc}} \approx 1.0$ (boresight alignment) delivers an average $+2.5\,\text{dB-Hz}$ gain in $C/N_0$.
3. **Temporal Dynamics Verified:** Autocorrelation of $C/N_0$ remains remarkably high over short intervals ($r = 0.968$ at $5\text{s}$, $r = 0.957$ at $10\text{s}$, $r = 0.931$ at $30\text{s}$). Transitions at $1\text{s}$ and $30\text{s}$ exhibit $\approx 20\%$ degradation, $\approx 60\%$ stability, and $\approx 20\%$ improvement, confirming that signal trend detection via autoregressive lags and velocity derivatives ($\frac{dC/N_0}{dt}$, $\frac{d\text{Elev}}{dt}$) provides high predictive power.
4. **Target Leakage Prohibition in Orientation Model:** Current $C/N_0$ **must never** be passed as an input feature to `02_orientation_model.py`. At inference time, the model evaluates candidate orientations that the user has not rotated to yet; passing current $C/N_0$ would leak the present physical orientation and blind the model to alternative orientations.

---

## 1. Dataset Architecture & DuckDB Analytical Inventory

All data was queried directly from `/home/tempest/Documents/ISRO_Project/gnss-eda/data/parquet/` via DuckDB streaming views:

| Parquet Table | Record Count | Disk Size | Key Fields | Analytical Role |
|---|---|---|---|---|
| **`Status.parquet`** | **6,956,811** | 19 MB | `UnixTimeMillis`, `ConstellationType`, `Svid`, `CarrierFrequencyHz`, `Cn0DbHz`, `ElevationDegrees`, `AzimuthDegrees`, `UsedInFix` | Primary signal measurement corpus (1 Hz) |
| **`OrientationDeg.parquet`** | **11,550,170** | 54 MB | `utcTimeMillis`, `yawDeg`, `pitchDeg`, `rollDeg`, `CalibrationAccuracy` | High-rate IMU orientation angles (~100–200 Hz) |
| **`Raw.parquet`** | **5,482,734** | 278 MB | `utcTimeMillis`, `Svid`, `PseudorangeRateMetersPerSecond`, `MultipathIndicator`, `BiasNanos` | Chipset raw GNSS clock & tracking metrics |
| **`Agc.parquet`** | **1,555,526** | 1.1 MB | `utcTimeMillis`, `AgcDb`, `CarrierFrequencyHz`, `ConstellationType` | Automatic Gain Control levels (sparse: < 1.2% temporal match) |
| **`Fix.parquet`** | **189,986** | 4.2 MB | `UnixTimeMillis`, `LatitudeDegrees`, `LongitudeDegrees`, `AltitudeMeters`, `AccuracyMeters`, `SpeedMps` | PVT (Position, Velocity, Time) navigation fixes |
| **`Accel.parquet`** | **6,369,268** | 45 MB | `utcTimeMillis`, `AccelXMps2`, `AccelYMps2`, `AccelZMps2` | Calibrated accelerometer motion vector |

---

## 2. Exploratory Data Analysis & Empirical Findings

### 2.1 Target Variable Distribution ($C/N_0$)
- **Total Records:** 6,956,811
- **Untracked / Null / Zero Signals:** 1,263,343 (18.16% of total rows where satellites are in ephemeris but signal lock is not achieved).
- **Active Signals ($C/N_0 > 0$):** 5,693,468 rows across 47,888 distinct epochs.
- **Parametric Statistics (Active Signals):**
  - Minimum: $6.0\,\text{dB-Hz}$
  - 25th Percentile ($Q_1$): $26.4\,\text{dB-Hz}$
  - Median ($Q_2$): $32.0\,\text{dB-Hz}$
  - Mean ($\mu$): $31.62\,\text{dB-Hz}$
  - 75th Percentile ($Q_3$): $37.2\,\text{dB-Hz}$
  - Maximum: $54.7\,\text{dB-Hz}$
  - Standard Deviation ($\sigma$): $7.71\,\text{dB-Hz}$

*Implication:* The active signal distribution is approximately Gaussian centered at $32.0\,\text{dB-Hz}$ with a standard deviation of $7.7\,\text{dB-Hz}$. Any filtering in data pipelines must discard $C/N_0 \le 0$ records before model training to avoid zero-inflation artifacts.

---

### 2.2 Constellation Performance Breakdown

Queried across the entire corpus via DuckDB:

| Constellation ID | Constellation Name | Total Signals | Active Signals | Mean $C/N_0$ ($\text{dB-Hz}$) | Std $C/N_0$ | Mean Elev ($^\circ$) | % Used in Fix |
|---|---|---|---|---|---|---|---|
| **5** | **BeiDou (BDS)** | 2,578,923 | 2,108,311 | 31.62 | 7.34 | 34.0° | 64.3% |
| **1** | **GPS** | 1,544,809 | 1,352,630 | 32.35 | 8.09 | 30.6° | 72.7% |
| **6** | **Galileo** | 1,488,976 | 1,272,714 | 30.40 | 7.78 | 31.9° | 63.0% |
| **3** | **GLONASS** | 726,219 | 620,479 | 32.74 | 8.90 | 26.2° | 69.7% |
| **7** | **NavIC / IRNSS** | 285,746 | 182,569 | 33.28 | 5.56 | 35.8° | 37.2% |
| **4** | **QZSS** | 219,290 | 126,730 | 27.73 | 7.93 | 18.1° | 37.5% |
| **2** | **SBAS** | 112,848 | 30,035 | 34.48 | 3.96 | 24.2° | 0.0% |

*Key Insights:*
1. **BeiDou dominates volume:** BeiDou represents 37.1% of all records due to GEO, IGSO, and MEO multi-orbit coverage in the Indian subcontinent.
2. **NavIC has the highest mean signal:** NavIC exhibits a robust $33.28\,\text{dB-Hz}$ average with the tightest standard deviation ($5.56\,\text{dB-Hz}$) owing to high-elevation geostationary orbital slots over India, but only 37.2% are marked `UsedInFix` due to phone receiver firmware constraints.
3. **Bug Audit in Legacy Code:** The legacy codebase checked constellation via `constellation.str.contains("gps")` on integer IDs, zeroing out all constellation features. Mapping must strictly use integer comparison: `(df["ConstellationType"] == 1).astype(float)`.

---

### 2.3 Carrier Frequency & Friis Path Loss Analysis

Signals group into distinct carrier frequencies:

| Carrier Freq (MHz) | Band Identification | Total Records | Active Records | Mean $C/N_0$ ($\text{dB-Hz}$) | Std $C/N_0$ | % Used in Fix |
|---|---|---|---|---|---|---|
| **1575.42** | L1 / E1 / B1C (GPS/Galileo/QZSS/SBAS) | 2,573,085 | 2,017,641 | 32.93 | 8.20 | 56.7% |
| **1176.45** | L5 / E5a (GPS/Galileo/NavIC) | 2,152,076 | 1,989,423 | 29.94 | 7.06 | 72.2% |
| **1561.10** | B1I (BeiDou legacy) | 1,505,451 | 1,065,945 | 31.64 | 7.13 | 59.3% |
| **1602.00 ± Δf** | GLONASS G1 FDMA (1598–1606 MHz) | 689,848 | 614,845 | 33.21 | 8.74 | 70.4% |

*Physical Justification:*
According to the Friis Transmission Equation, free-space path loss is proportional to frequency squared:
$$\Delta L_{\text{freq}} = 20 \log_{10}\left(\frac{f_{\text{carrier}}}{1575.42\,\text{MHz}}\right)$$
The L5 band ($1176.45\,\text{MHz}$) experiences $\approx -2.54\,\text{dB}$ relative free-space path difference compared to L1, and has higher chip rate ($10.23\,\text{MHz}$ vs $1.023\,\text{MHz}$) with distinct antenna gain characteristics on smartphone patch antennas. Frequency must be explicitly encoded via `freq_ghz` and `freq_loss_db`.

---

### 2.4 Elevation Angle vs $C/N_0$ Relationship

Evaluating the physical atmosphere and horizon cutoff via 10° elevation bins:

| Elevation Range | Record Count | Mean $C/N_0$ ($\text{dB-Hz}$) | Median $C/N_0$ | Std $C/N_0$ | % Used in Fix |
|---|---|---|---|---|---|
| **0° – 10°** | 406,503 | 24.29 | 23.2 | 7.72 | 43.7% |
| **10° – 20°** | 1,006,134 | 28.70 | 28.5 | 7.96 | 71.4% |
| **20° – 30°** | 1,107,227 | 32.75 | 33.0 | 7.84 | 78.5% |
| **30° – 40°** | 1,162,308 | 33.01 | 33.3 | 7.42 | 81.9% |
| **40° – 50°** | 948,504 | 33.20 | 33.4 | 6.73 | 83.7% |
| **50° – 60°** | 510,730 | 32.77 | 33.1 | 6.34 | 86.4% |
| **60° – 70°** | 331,079 | 33.82 | 34.0 | 6.53 | 81.7% |
| **70° – 80°** | 188,782 | 32.41 | 33.1 | 7.00 | 77.6% |
| **80° – 90°** | 32,164 | 34.82 | 34.5 | 8.17 | 99.5% |

*Physical Observations:*
- From $0^\circ$ to $25^\circ$, $C/N_0$ undergoes a steep $+8.5\,\text{dB}$ gain as line-of-sight escapes ground clutter, multipath, and thick tropospheric refraction.
- Above $30^\circ$, $C/N_0$ stabilizes between $32.8\text{--}34.8\,\text{dB-Hz}$.
- Simple $1/\sin(\text{elev})$ diverges to infinity at $0^\circ$. The **Kasten-Young airmass model** correctly bounds this behavior:
  $$\text{airmass}_{\text{ky}} = \frac{1}{\sin(\text{El}) + 0.15 \cdot (\text{El} + 3.885)^{-1.25}}$$
  At horizon ($\text{El}=0^\circ$), $\text{airmass}_{\text{ky}} \approx 38.0$ (finite physical atmosphere), whereas at zenith ($\text{El}=90^\circ$), $\text{airmass}_{\text{ky}} = 1.0$.

---

### 2.5 3D Antenna Incidence Vector Validation

Joining `Status.parquet` with `OrientationDeg.parquet` at 1-second synchronization:
- Satellite ENU line-of-sight unit vector:
  $$\mathbf{s} = \begin{bmatrix} \cos(\text{El}) \sin(\text{Az}) \\ \cos(\text{El}) \cos(\text{Az}) \\ \sin(\text{El}) \end{bmatrix}$$
- Phone antenna normal boresight unit vector derived from Euler rotation matrix $R(\psi, \theta, \phi)$:
  $$\mathbf{n} = \begin{bmatrix} \sin\phi \sin\psi + \cos\phi \sin\theta \cos\psi \\ -\sin\phi \cos\psi + \cos\phi \sin\theta \sin\psi \\ \cos\phi \cos\theta \end{bmatrix}$$
- Incidence Angle Cosine: $\cos\theta_{\text{inc}} = \mathbf{s} \cdot \mathbf{n} = s_e n_e + s_n n_n + s_u n_u$.

Empirical results across 200,000 joined records:

| $\cos\theta_{\text{inc}}$ Bin | Incidence Angle Range | Mean $C/N_0$ ($\text{dB-Hz}$) | Physical Antenna State |
|---|---|---|---|
| **[-1.0, -0.6)** | 127° – 180° | 31.12 | Direct chassis / screen blockage (backlobe) |
| **[-0.6, -0.2)** | 102° – 127° | 31.41 | Oblique rear multipath / diffraction |
| **[-0.2, +0.2)** | 78° – 102° | 30.93 | Lateral horizon edge (minimum patch gain) |
| **[+0.2, +0.6)** | 53° – 78° | 31.54 | Forward hemispherical entrance |
| **[+0.6, +0.8)** | 37° – 53° | 32.81 | Strong mainlobe reception |
| **[+0.8, +1.0]** | 0° – 37° | **33.32** | **Direct boresight alignment (peak gain)** |

*Conclusion:* Aligning the phone antenna normal with the satellite direction yields a measurable $+2.4\,\text{dB}$ improvement compared to horizon/edge orientations. This validates $\cos\theta_{\text{inc}}$ as the primary physics driver for `02_orientation_model.py`.

---

### 2.6 Temporal Dynamics & Horizon Autocorrelation

Using DuckDB window functions partitioned by signal key (`ConstellationType, Svid, CarrierFrequencyHz`):

| Temporal Horizon | Metric | Measured Value | Operational Significance |
|---|---|---|---|
| **$\Delta t = 1\text{s}$** | Autocorrelation ($r$) | **0.762** | Captures high-frequency noise & multipath jitter |
| | Std of $\Delta C/N_0$ | $5.44\,\text{dB}$ | Rapid step transitions upon blockage |
| | Degradation (< -1.0 dB) | **19.3%** | Rate of immediate fading |
| | Stable ([-1.0, +1.0] dB) | **61.6%** | Steady tracking condition |
| | Improvement (> +1.0 dB) | **19.1%** | Re-acquisition / clearing obstacle |
| **$\Delta t = 5\text{s}$** | Autocorrelation ($r$) | **0.968** | Stable trend line |
| | Std of $\Delta C/N_0$ | $1.97\,\text{dB}$ | Smoothed tracking error |
| **$\Delta t = 10\text{s}$** | Autocorrelation ($r$) | **0.957** | Predictable medium-range slope |
| | Std of $\Delta C/N_0$ | $2.26\,\text{dB}$ | Clear separation of rising vs setting SVs |
| **$\Delta t = 30\text{s}$** | Autocorrelation ($r$) | **0.931** | **Workflow decision window (30–45s loop)** |
| | Std of $\Delta C/N_0$ | $2.78\,\text{dB}$ | Significant orbital drift & orientation shift |
| | Degradation (< -1.5 dB) | **20.8%** | **Trigger threshold for Orientation Model** |
| | Stable ([-1.5, +1.5] dB) | **58.3%** | Keep current orientation |
| | Improvement (> +1.5 dB) | **20.9%** | Orientation is performing well |

---

## 3. Model 1: Temporal Signal Prediction Model (`02_future_model.py`)

### 3.1 Objective & Architecture
The temporal model evaluates whether currently received GNSS signals will degrade, stay constant, or improve over the next $1\text{s}\text{--}60\text{s}$ (specifically evaluated at $1\text{s}$, $5\text{s}$, $10\text{s}$, and $30\text{s}$). It acts as the gatekeeper:
$$\text{If } \Delta C/N_{0, \text{pred}}(t + 30\text{s}) \le -1.5\,\text{dB} \quad \text{or} \quad \text{Constellation Quality Degrades} \implies \textbf{Trigger Orientation Model}$$

### 3.2 Recommended Feature Set for Temporal Model

| Feature Name | Type | Formula / Origin | Why It Must Be Included |
|---|---|---|---|
| **`Cn0DbHz`** | Autoregressive Base | $C/N_0(t)$ from `Status` | Baseline signal power at current epoch ($r > 0.93$ with future). |
| **`cn0_lag_1s`** | Temporal Lag | $C/N_0(t - 1\text{s})$ | Immediate preceding state. |
| **`cn0_lag_2s`** | Temporal Lag | $C/N_0(t - 2\text{s})$ | 2-second reference point. |
| **`cn0_lag_5s`** | Temporal Lag | $C/N_0(t - 5\text{s})$ | Medium-range reference point. |
| **`cn0_lag_10s`** | Temporal Lag | $C/N_0(t - 10\text{s})$ | Long-range trend anchor. |
| **`cn0_delta_1s`** | Derivative | $C/N_0(t) - C/N_0(t - 1\text{s})$ | Instantaneous rate of change ($\frac{dC/N_0}{dt}$). |
| **`cn0_delta_5s`** | Derivative | $C/N_0(t) - C/N_0(t - 5\text{s})$ | 5-second momentum slope. |
| **`cn0_slope_5s`** | Derivative | $\frac{C/N_0(t) - C/N_0(t - 5\text{s})}{5.0}$ | Velocity of signal change (dB/s). |
| **`ElevationDegrees`** | Geometric State | Direct from `Status` | Primary physical bound on maximum possible signal. |
| **`elevation_sin`** | Trigonometric | $\sin(\text{El})$ | Linearizes zenith geometric path length. |
| **`elevation_delta_1s`** | Trajectory Velocity | $\text{El}(t) - \text{El}(t - 1\text{s})$ | Distinguishes rising satellites ($+$) from setting satellites ($-$). |
| **`elevation_delta_5s`** | Trajectory Velocity | $\text{El}(t) - \text{El}(t - 5\text{s})$ | 5-second orbital elevation rate. |
| **`azimuth_delta_1s`** | Trajectory Velocity | Circular $\Delta(\text{Az}(t), \text{Az}(t - 1\text{s}))$ | Satellite orbital azimuth velocity on the sky. |
| **`airmass_ky`** | Atmospheric Loss | $1 / (\sin(\text{El}) + 0.15(\text{El} + 3.885)^{-1.25})$ | Quantifies atmospheric absorption; critical for low-elevation setting satellites. |
| **`freq_ghz`** | RF Parameter | $\text{CarrierFrequencyHz} / 10^9$ | Separates L1 ($1.575\,\text{GHz}$) and L5 ($1.176\,\text{GHz}$) propagation physics. |
| **`freq_loss_db`** | RF Parameter | $20 \log_{10}(\text{Freq} / 1575.42\text{ MHz})$ | Friis path loss adjustment. |
| **`is_gps`** | Constellation ID | $(\text{ConstellationType} == 1)$ | Constellation-specific transmit power and code structure. |
| **`is_glonass`** | Constellation ID | $(\text{ConstellationType} == 3)$ | FDMA signal characteristics. |
| **`is_beidou`** | Constellation ID | $(\text{ConstellationType} == 5)$ | GEO/IGSO orbital stationary characteristics. |
| **`is_galileo`** | Constellation ID | $(\text{ConstellationType} == 6)$ | High-altitude MEO satellite power. |
| **`is_navic`** | Constellation ID | $(\text{ConstellationType} == 7)$ | Geostationary regional coverage over India. |
| **`UsedInFix`** | Filter Quality | Binary from `Status` | Receiver tracking loop confidence (1 if PVT lock, 0 if tracking only). |
| **`SignalCount`** | Sky Density | `SignalCount` from `Status` | Total number of tracking channels active in current epoch. |

---

## 4. Model 2: Best Orientation Recommendation Model (`02_orientation_model.py`)

### 4.1 Objective & Candidate Evaluation Mechanism
When triggered, `02_orientation_model.py` evaluates candidate phone orientations:
$$\Psi_{\text{cand}} = \{0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ, 315^\circ\}$$
For each candidate yaw $\psi_c$, it computes the hypothetical geometric features for all currently visible satellites and predicts $\widehat{C/N}_0(\psi_c, s_i)$. The system selects the orientation that maximizes the total score:
$$\psi^* = \arg\max_{\psi_c} \sum_{i \in \text{Visible}} w_i \cdot \widehat{C/N}_0(\psi_c, s_i)$$

### 4.2 Prohibited Features (Why They Must Be Excluded)
1. **Current `Cn0DbHz` (Strict Leakage):** Current signal strength represents the phone's *current* position. If included, the model cannot evaluate a hypothetical orientation $\psi_c$ because it does not know what $C/N_0$ would be at $\psi_c$.
2. **`facing`, `phone_roll`, `tilt_angle` (Device Bias):** Categorical metadata from the OnePlus rig. Passing them causes the model to memorize the OnePlus rig rather than learning satellite reception geometry.
3. **Raw `pitchDeg`, `rollDeg`:** Subsumed into the 3D dot product $\cos\theta_{\text{inc}}$.

### 4.3 Recommended Feature Set for Orientation Model (28 Features)

| # | Feature Name | Mathematical Definition | Physical Purpose |
|---|---|---|---|
| 1 | **`cos_inc`** | $\mathbf{s}_{\text{enu}} \cdot \mathbf{n}_{\text{enu}}(\psi_{\text{cand}}, \theta, \phi)$ | **Core Driver:** 3D antenna incidence angle cosine. Boresight alignment yields $+2.5\,\text{dB}$. |
| 2 | **`inc_deg`** | $\arccos(\text{clip}(\cos\theta_{\text{inc}}, -1, 1)) \cdot \frac{180}{\pi}$ | Nonlinear angular deviation from antenna normal (0° to 180°). |
| 3 | **`elevation_sin`** | $\sin(\text{El})$ | Direct zenith projection of satellite elevation. |
| 4 | **`elevation_cos`** | $\cos(\text{El})$ | Horizon projection of satellite elevation. |
| 5 | **`is_above_horizon`**| $(\text{El} > 0)^\circ$ | Binary mask for unmasked satellites. |
| 6 | **`rel_az_sin`** | $\sin(\text{Az}_{\text{sat}} - \psi_{\text{cand}})$ | Cyclic relative azimuth (left-right antenna symmetry). |
| 7 | **`rel_az_cos`** | $\cos(\text{Az}_{\text{sat}} - \psi_{\text{cand}})$ | Cyclic relative azimuth (front-back antenna symmetry). |
| 8 | **`airmass_simple`** | $1 / \sin(\text{El}_{\text{clipped}})$ | First-order atmospheric path length multiplier. |
| 9 | **`airmass_ky`** | $\frac{1}{\sin(\text{El}) + 0.15(\text{El} + 3.885)^{-1.25}}$ | Kasten-Young refraction model (accurate near horizon). |
| 10 | **`freq_ghz`** | $\text{CarrierFrequencyHz} / 10^9$ | Carrier frequency in GHz. |
| 11 | **`freq_loss_db`** | $20 \log_{10}(\text{Freq} / 1575.42\text{ MHz})$ | Friis free-space loss relative to GPS L1. |
| 12 | **`is_gps`** | $(\text{ConstellationType} == 1)$ | GPS indicator. |
| 13 | **`is_sbas`** | $(\text{ConstellationType} == 2)$ | SBAS augmentation indicator. |
| 14 | **`is_glonass`** | $(\text{ConstellationType} == 3)$ | GLONASS indicator. |
| 15 | **`is_qzss`** | $(\text{ConstellationType} == 4)$ | QZSS indicator. |
| 16 | **`is_beidou`** | $(\text{ConstellationType} == 5)$ | BeiDou indicator. |
| 17 | **`is_galileo`** | $(\text{ConstellationType} == 6)$ | Galileo indicator. |
| 18 | **`is_navic`** | $(\text{ConstellationType} == 7)$ | NavIC / IRNSS indicator. |
| 19 | **`rel_az_x_el`** | $\text{rel\_az\_cos} \times \text{elevation\_sin}$ | Couples horizontal orientation with satellite altitude. |
| 20 | **`cos_inc_x_airmass`**| $\cos\theta_{\text{inc}} \times \text{airmass\_ky}$ | Interaction: antenna gain combined with atmospheric attenuation. |
| 21 | **`navic_x_airmass`** | $\text{is\_navic} \times \text{airmass\_ky}$ | NavIC geostationary atmospheric interaction. |
| 22 | **`gps_x_airmass`** | $\text{is\_gps} \times \text{airmass\_ky}$ | GPS atmospheric interaction. |
| 23 | **`galileo_x_airmass`**| $\text{is\_galileo} \times \text{airmass\_ky}$ | Galileo atmospheric interaction. |
| 24 | **`freq_loss_x_elev`** | $\text{freq\_loss\_db} \times \text{elevation\_sin}$ | Frequency attenuation scaled by elevation. |
| 25 | **`elev_x_freq`** | $\text{elevation\_sin} \times \text{freq\_ghz}$ | Interaction of elevation angle and carrier frequency. |
| 26 | **`is_l5_band`** | $(\text{CarrierFrequencyHz} < 1.3 \times 10^9)$ | Binary flag for high-precision L5/E5a/NavIC band. |
| 27 | **`cos_inc_sq`** | $(\cos\theta_{\text{inc}})^2$ | Models quadratic patch antenna gain roll-off. |
| 28 | **`UsedInFix`** | Binary flag from `Status` | Weights satellites that receiver is already utilizing. |

---

## 5. Summary Feature Comparison Matrix

| Feature | Source Table | Temporal Model (`02_future`) | Orientation Model (`02_orientation`) | Real-time Android Feasible? | Primary Rationale |
|---|---|:---:|:---:|:---:|---|
| `Cn0DbHz` | `Status` | **YES** | ❌ **PROHIBITED** | Yes (`GnssStatus.getCn0DbHz`) | Autoregressive anchor in temporal; leakage in orientation. |
| `cn0_lag_1s..10s` | `Status` | **YES** | ❌ **NO** | Yes (ring buffer in app) | Essential for momentum and decay forecasting. |
| `cn0_delta_1s..5s` | `Status` | **YES** | ❌ **NO** | Yes (subtraction) | Derivative $\frac{dC/N_0}{dt}$ captures fading. |
| `cos_inc` | `Status` + `Orientation` | ❌ Optional | **YES (Top 1)** | Yes (Euler transformation) | Primary RF antenna boresight dot product. |
| `inc_deg` | Derived | ❌ Optional | **YES** | Yes ($\arccos$) | Angular off-boresight displacement. |
| `ElevationDegrees` | `Status` | **YES** | **YES** | Yes (`GnssStatus.getElevationDegrees`) | Primary satellite geometry determinant. |
| `elevation_delta_1s` | `Status` | **YES** | ❌ **NO** | Yes (ring buffer) | Orbital velocity (rising vs setting). |
| `AzimuthDegrees` | `Status` | **YES** | **YES** | Yes (`GnssStatus.getAzimuthDegrees`) | Satellite azimuth. |
| `rel_az_sin/cos` | Derived | ❌ Optional | **YES** | Yes ($\text{Az} - \psi_{\text{cand}}$) | Antenna horizontal pattern symmetry. |
| `airmass_ky` | Derived | **YES** | **YES** | Yes (Kasten-Young formula) | Atmospheric refraction loss. |
| `freq_ghz` / `freq_loss_db` | `Status` | **YES** | **YES** | Yes (`GnssStatus.getCarrierFrequencyHz`) | Friis path loss difference across L1/L5. |
| `ConstellationType` | `Status` | **YES** | **YES** | Yes (`GnssStatus.getConstellationType`) | One-hot constellation tracking behavior. |
| `UsedInFix` | `Status` | **YES** | **YES** | Yes (`GnssStatus.usedInFix`) | Navigation filter confidence indicator. |
| `facing` / `phone_roll` | Metadata | ❌ **PURGED** | ❌ **PURGED** | No | Overfits to OnePlus rig; non-existent in wild. |
| `AgcDb` | `Agc` | ❌ Skip | ❌ Skip | No (< 1.2% match rate) | Sparse across Android devices; unneeded. |

---

## 6. Implementation Checklist

1. **`01_merge_sessions.py`:** Update merge logic to use streaming DuckDB queries to join `Status.parquet` and `OrientationDeg.parquet` on synchronized timestamps, producing the combined training corpus with zero RAM spikes.
2. **`02_future_model.py`:** Add `airmass_ky`, `freq_loss_db`, and numeric constellation mappings (`(ConstellationType == 1).astype(float)`).
3. **`02_orientation_model.py`:** Adopt the 28-feature physics set from `feature/physics-orientation-features`, ensuring current $C/N_0$ and OnePlus categorical labels are excluded.
4. **`03_orientation_inference.py`:** Validate candidate yaw rotation grid ($\psi_c \in [0^\circ, 360^\circ)$) using the vectorized `cos_inc` calculation.
