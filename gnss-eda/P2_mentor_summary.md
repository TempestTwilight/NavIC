# Phase 2 Mentor Summary: Smartphone GNSS Signal Optimization

**Target Audience:** Engineering Mentors, Technical Project Leads, and System Architects.  
**Scope:** Synthesis of 95 raw GNSS Logger logs (9.47 GB, 5.5M rows) across 4 phone platforms.  
**Core Objective:** Establish empirical relationships governing GNSS signal strength ($C/N_0$) and design an on-device ML model that beats persistence over a 15-second prediction horizon.

---

## 1. Executive Story: The Physics, Pitfalls, and Machine Learning

Smartphone GNSS positioning suffers from degraded accuracy and frequent multipath fading due to low-cost linearly polarized patch antennas, physical chassis blockage, and antenna pointing changes as users move. To optimize satellite selection on Android devices, machine learning must anticipate signal quality changes over a 15-second planning horizon.

Through exhaustive DuckDB SQL profiling of 5.5 million measurements across 4 smartphone models (OnePlus CPH2401, CPH2487, OPPO CPH2325, Samsung SM-M146B), Phase 2 evaluated the physical mechanisms driving signal variance. We uncovered critical data pitfalls—including a **14.5 dB cross-device C/N0 offset** (confounded with recording date), **Euler yaw/heading divergence** at vertical phone pitch, and **92.4% intra-day satellite constellation overlap**—and demonstrated why standard MSE models collapse into trivial persistence approximations. 

By reformulating the prediction target as **residual fading drift ($\Delta C/N_0$)** and leveraging **rolling 5-second carrier jitter ($\sigma_{5\text{s}}$)**, we unlocked actionable predictive signals that eliminate cross-device covariate shift and reduce satellite selection regret.

---

## 2. Top 8 Foundational Figures & One-Line Takeaways

| Fig # | Artifact Name | One-Line Takeaway |
| :--- | :--- | :--- |
| **01** | `01_gravity_vs_pitch_roll.png` | **Sensor Alignment:** `OrientationDeg.pitchDeg` directly measures physical nose-up pitch tilt ($r > 0.98$), resolving Android documentation ambiguities. |
| **02** | `02_yaw_fop_gimbal_lock.png` | **Euler Yaw Divergence:** Euler yaw and fused heading diverge by $45°–120°$ (median) when pitch exceeds $70^\circ$, likely from both gimbal lock singularities and Android's coordinate frame switching behavior. Quaternion-derived body vectors ($s_{bx}, s_{by}, s_{bz}$) are more stable. |
| **07** | `07_variance_decomposition.png` | **Variance Decomposition:** Device explains 21.3%, elevation 0.75% (after device), epoch common-mode 34.31%, per-satellite residual 65.69%. Sequential ANOVA shares depend on term order; device and epoch-level signal dominate. |
| **10** | `10_device_gain_offset.png` | **Hardware Disparity:** Samsung Exynos reports $\approx 14.5\text{ dB}$ higher nominal $C/N_0$ than MediaTek/Qualcomm devices due to LNA/DSP gain calibration. |
| **12** | `12_antenna_pattern_directivity.png` | **Antenna Directivity:** Android +Z points out of the screen (toward user). Signals along this axis show $+2.8\text{ dB}$ residual gain (10–20° bin); rear hemisphere shows $-4.2\text{ dB}$ attenuation (~7 dB range). Pattern is noisy and device-specific. |
| **18** | `18_cn0_t_vs_t15.png` | **The Persistence Paradox:** Current signal strength explains $90.5\%$ of variance at $t+15\text{s}$ ($r = 0.951$); models must target residual fading drift ($\Delta C/N_0$) to beat persistence. |
| **24** | `24_fading_precursor_roc.png` | **Early Fading Detection:** Past 5-second carrier jitter ($\sigma_{5\text{s}}$) detects impending fading drops 15 seconds in advance ($\text{AUC} = 0.74$). |
| **29** | `29_cross_device_covariate_shift.png` | **Residual Drift Similarity:** While raw signal levels shift by $14.5\text{ dB}$ across devices, 15-second residual drift $\Delta C/N_0$ is observed to be similar: mean $\approx 0.0–0.1\text{ dB}$, std ranges 1.15–3.99 dB. Not formally tested for statistical invariance. |

---

## 3. Top 4 System & Modeling Risks

1. **Dropout Risk in Selection:** Complete satellite tracking loss occurs in $2.6\%$ of candidate epochs (24/922 in table 28). When the chosen satellite drops out, realized C/N0 falls to the sensitivity floor (~12 dB-Hz assumed), generating large regret. Selection policies should use expected value: $\mathbb{E}[\text{C/N0}] = P(\text{survive}) \cdot \hat{C/N_0} + (1 - P) \cdot \text{floor}$.
2. **Orbital Constellation Leakage:** Sessions recorded within 2 hours of each other share **$92.4\%$ of visible satellites** at near-identical sky coordinates. Random train/test splits result in severe geometric data leakage; cross-validation **must be grouped strictly by calendar day**.
3. **Severe Feature Collinearity:** Elevation angle $\sin(\theta_{\text{elev}})$ and Kasten-Young atmospheric airmass $X(\theta)$ exhibit a correlation of $\rho = -0.99$, which previously caused linear weight explosion ($\pm 117$). Airmass must be pruned.
4. **Feature Availability Across Devices:** `BasebandCn0DbHz` is populated on all 4 devices (universal). `Raw.AgcDb` is null on Samsung SM-M146B only; populated on the three OnePlus/OPPO devices. Verified via direct query 2026-11-11.

---

## 4. Completed Experiments & Production Architecture Selection

1. **Experiment A (Baseline Residual ML vs Persistence):** **Adopted (Winner).**
   - Trained Ridge regression and HistGradientBoosting over 5.35M instances across 40 sessions.
   - Persistence selection regret drops from **20.09 dB-Hz** (median 19.80 dB-Hz) to **6.90 dB-Hz** (median 3.80 dB-Hz) with Ridge, and **6.52 dB-Hz** (median 3.50 dB-Hz) with HistGB.
   - Top-1 tracking accuracy improves nearly 9x from 3.16% to 25.29% (Ridge) / 27.52% (HistGB). Top-3 accuracy reaches 47.23% / 51.07%. Ridge CV RMSE is 4.780 ± 0.324 ($R^2 = 0.345$).
   - Ridge selected for mobile deployment: matches tree regret within 0.38 dB-Hz, requires $<5\ \mu\text{s}$ CPU inference, zero tree memory overhead, and straightforward Kotlin porting.

2. **Experiment B (Two-Stage Hurdle Architecture):** **Evaluated & Rejected for Production.**
   - Fading classifier ($|\Delta| > 1.5\text{ dB}$) achieved ROC-AUC 0.67–0.75 across devices, reducing fading RMSE by 2.15–3.08 dB-Hz over persistence.
   - Rejected because single-stage Ridge with momentum features (`cn0_lag_10s`, `cn0_delta_2s`) naturally down-weights decaying signals without the complexity of a dual-model runtime pipeline.

3. **Experiment C (Learning-to-Rank Formulation):** **Evaluated & Rejected.**
   - Epoch-relative ranking target ($C/N_{0, i} - \max_j C/N_{0, j}$) severely degraded selection accuracy. Top-1 tracking accuracy collapsed to **4.4%** on OPPO (vs 31.7% persistence) and **15.4%** on OnePlus CPH2487 (vs 30.3% persistence). Discarded in favor of direct residual regression.

4. **Experiment D (Zero-Shot LODO Cross-Device Transferability):** **Verified & Adopted.**
   - Leave-One-Device-Out cross-validation proved that $\Delta C/N_0$ residual regression transfers zero-shot across unseen phone hardware (regret delta between raw and delta target was $<0.0002\text{ dB-Hz}$). Standard physics features absorb front-end OEM gain discrepancies completely.

5. **Experiment E (Survival-Weighted Utility):** **Evaluated & Rejected.**
   - Weighting predictions by empirical survival probability $P(\text{survive})$ at $\eta = 12^\circ$ elevation produced negligible regret change ($\pm 0.01\text{ dB-Hz}$; e.g., OPPO +0.004 dB-Hz, OnePlus CPH2401 0.000 dB-Hz, Samsung -0.047 dB-Hz) over direct regression.

## 5. Next Steps for Mobile Deployment
1. **Freeze Model Weights:** Export 36 Ridge coefficients, intercept, and `StandardScaler` ($\mu, \sigma$) arrays from `model/temporal_artifacts/` to JSON.
2. **Pure Kotlin Implementation (`RidgeInference.kt`):** Implement standard scalar normalization and dot product in pure Kotlin ($<5\ \mu\text{s}$ per satellite candidate, zero external ML dependencies).
3. **Android Daemon Integration:** Wire into `GnssOptimizerService` for epoch-by-epoch $t+15\text{s}$ residual prediction and candidate constellation ranking.
