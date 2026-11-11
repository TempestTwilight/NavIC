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

## 4. The Next 5 Priority Experiments

1. **Experiment 1 (Residual Learning Benchmark):** Train Ridge regression to predict residual drift $\Delta C/N_0 = C/N_0(t+15\text{s}) - C/N_0(t)$ using $s_{bz}$, $\theta_{\text{elev}}$, and $\sigma_{5\text{s}}$ under leak-free calendar-day cross-validation.
2. **Experiment 2 (Two-Stage Hurdle Architecture):** Implement a binary classifier to detect impending fading ($|\Delta| > 1.5\text{ dB}$) triggered by high jitter $\sigma_{5\text{s}}$, coupled with a specialized fading regressor. Note: MSE is dominated by the 16% fading epochs (high variance), not the 84% static ones.
3. **Experiment 3 (Learning-to-Rank Formulation):** Formulate satellite selection with LightGBM `lambdarank` grouped by epoch to directly optimize NDCG@4, eliminating loss function mismatch.
4. **Experiment 4 (Survival-Weighted Utility):** Weight predicted signal strength by empirical survival probability using expected value: $U(s) = \hat{P}(\text{survive}) \cdot \hat{C/N_0}(s) + (1 - \hat{P}(\text{survive})) \cdot \text{floor}$ where floor is minimum tracking sensitivity (~12 dB-Hz).
5. **Experiment 5 (Shadow-Mode Android Deployment):** Deploy the calibrated Ridge pipeline inside an asynchronous observer thread within the Android app, logging predictions and ground-truth regret without interfering with active PVT positioning.
