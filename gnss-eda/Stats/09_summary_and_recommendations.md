# 09 — Summary and Recommendations

Exploratory Data Analysis summary for the raw Android GNSS Logger dataset (95 sessions, 9.47 GB, 4 phone models: OnePlus CPH2401, Samsung SM-M146B, OnePlus CPH2487, OPPO CPH2325), supporting the ML project to:

- (a) Predict $C/N_0$ 15 seconds ahead to select the best GNSS signal.
- (b) Predict $C/N_0$ as a function of phone orientation to recommend optimal antenna pointing.

---

## 1. Executive Summary

This EDA evaluated **12 raw tables** and **157 columns** across 5.48M `Raw` measurement rows and 6.96M `Status` visibility rows. The analysis revealed that:

1. **The dataset is high quality for orientation-dependent modeling**, with rich multi-constellation coverage (GPS, GLONASS, BeiDou, Galileo, NavIC, QZSS) and dual-frequency tracking (L1/L5).
2. **Missingness is almost entirely structural by device OEM**, not random. Features must be chosen to work across all 4 chipsets or normalized per device.
3. **The persistence baseline is an exceptionally strong competitor** over short horizons (1–15 s). Models predicting raw $C/N_0$ shrink toward the mean and lose to persistence. The target must be formulated as a residual $\Delta C/N_0$ or elevation-anomaly.
4. **Orientation Euler angles suffer from gimbal lock at vertical pitch**, requiring 3D body-frame vectors rather than raw yaw/pitch.

---

## 2. Top 10 Most Important Findings

1. **GLONASS C/N0 = 0 is a MediaTek Vendor Reporting Bug [OBSERVED, Q-001]**

   - 210,539 rows (35.5% of all GLONASS) report $C/N_0 = 0\text{ dB-Hz}$ on OnePlus CPH2401 and OPPO CPH2325, despite the receiver having full tracking lock (`State = 32995`).
   - _Impact:_ These rows are NOT weak signals — they are unmeasured values. Including them in training destroys the loss function. Drop or mask them.

2. **14.5 dB Hardware Gain Spread Across Devices [OBSERVED]**

   - Samsung SM-M146B mean $C/N_0 = 39.17\text{ dB-Hz}$; OPPO CPH2325 mean $C/N_0 = 24.62\text{ dB-Hz}$ under similar open-sky conditions.
   - _Impact:_ An unnormalized model will learn to predict phone brand rather than orientation physics. Target must be epoch-centered or residual.

3. **Sv\* Chipset Columns Are 100% Null on MediaTek [OBSERVED, Q-013]**

   - `SvPositionEcef*` is never provided by OnePlus CPH2401 (41.3% of the corpus).
   - _Impact:_ Do NOT use ECEF coordinates as model inputs. Use `ElevationDegrees` and `AzimuthDegrees` from `Status`, which are 100% populated on all devices.

4. **Persistence Beats Raw Regression Over Short Horizons [OBSERVED]**

   - Signal change over 15s is $< 1.5\text{ dB-Hz}$ in >75% of epochs. A model predicting absolute future $C/N_0$ suffers from L2 shrinkage toward the dataset mean ($31.8\text{ dB-Hz}$) and loses to $C/N_0(t+\Delta t) = C/N_0(t)$.
   - _Impact:_ The target must be $\Delta C/N_0 = C/N_0(t+\Delta t) - C/N_0(t)$. This collapses to persistence as its mathematical floor.

5. **Euler Angle Gimbal Lock Diverges by >100° at Vertical Pitch [OBSERVED, Q-006]**

   - When phone pitch exceeds 70° (e.g. handheld upright), `OrientationDeg.yawDeg` exhibits up to 124° median error vs the true magnetic heading.
   - _Impact:_ Compute 3D body-frame satellite vectors $\mathbf{s}_{\text{body}} = \mathbf{R} \cdot \mathbf{s}_{\text{enu}}$ using rotation quaternions from `FOP` or `GameRotationVector`, bypassing Euler angles entirely.

6. **Orientation Session Names Are Rig-Relative, Not Compass North [OBSERVED, Q-007]**

   - Session folder names like `90_front_vertical` encode the phone's position on the experimental mounting rig, not its heading relative to true north.
   - _Impact:_ Never use folder names as ground-truth target headings. Use calibrated sensor measurements.

7. **UsedInFix and SignalCount Are Post-PVT Target Leaks [OBSERVED, Q-015]**

   - `Status.UsedInFix` is determined by the position engine AFTER the measurement epoch. Using it as a feature is a forward-looking data leak.
   - _Impact:_ Exclude `UsedInFix`, `SignalCount`, `HasAlmanacData`, and `HasEphemerisData` from feature sets.

8. **Collinear Feature Pairs Exploded Ridge Coefficients to ±117 [OBSERVED]**

   - `freq_ghz` vs `freq_loss_db` ($r = 0.99997$) and `airmass_simple` vs `airmass_ky` ($r = 0.9896$) caused extreme matrix ill-conditioning.
   - _Impact:_ Keep strictly one representation per physical phenomenon: `freq_loss_db`, `airmass_ky`, $\sin(\text{elevation})$, and $\cos(\theta_{\text{inc}})$.

9. **Calendar-Day Grouping Is Essential for Honest Validation [OBSERVED]**

   - Satellites move at ~15°/hour. Sessions recorded within minutes of each other share nearly identical sky constellations. Random or time-gap splits allow the test fold to memorize the constellation.
   - _Impact:_ Group cross-validation folds strictly by **calendar recording day** (`GroupKFold` on `recording_day`).

10. **Duplicate Signals Represent Separate Code Channels (I/Q) [OBSERVED, Q-002]**
    - Multi-frequency signals (GPS L5, Galileo E5a) appear twice per epoch with `CodeType = 'I'` and `'Q'`.
    - _Impact:_ Group signal tracks by `(ConstellationType, Svid, CarrierFrequencyHz, CodeType)` to avoid combining independent tracking channels.

---

## 3. Risks for Machine Learning

| Risk                            | Severity | Root Cause                                      | Mitigation                                                                         |
| ------------------------------- | -------- | ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Hardware Gain Bias**          | HIGH     | 14.5 dB LNA gain spread across phone models     | Epoch-relative target centering: $y = C/N_{0,i} - \overline{C/N_{0,\text{epoch}}}$ |
| **Constellation Sky Leakage**   | HIGH     | Slow satellite orbital motion (~15°/hr)         | Group CV folds by calendar date, not by 10s session splits                         |
| **Gimbal Lock Error**           | HIGH     | Euler yaw/pitch singularity near ±90° pitch     | Rotate line-of-sight unit vector into body coordinates: $s_{bx}, s_{by}, s_{bz}$   |
| **Collinear Weight Explosion**  | MEDIUM   | Dual airmass and dual frequency representations | Prune redundant features ($r > 0.95$) before fitting Ridge                         |
| **Post-PVT Data Leakage**       | MEDIUM   | Features computed after position solution       | Strip `UsedInFix`, `SignalCount`, and `Fix.ExtraKeyValuePairs`                     |
| **GLONASS C/N0=0 Corruption**   | MEDIUM   | MediaTek HAL reporting bug                      | Drop rows with $C/N_0 = 0$ from training targets                                   |
| **Non-Causal Timestamp Jitter** | LOW      | `merge_asof` with `direction="nearest"`         | Enforce strictly backward lags (`direction="backward"`)                            |

---

## 4. Suggested Preprocessing Decisions (Do Not Implement Yet)

1. **Target Definitions:**

   - **Temporal Model:** Predict residual $\Delta C/N_0(t+15\text{s}) = C/N_0(t+15\text{s}) - C/N_0(t)$.
   - **Orientation Model:** Predict epoch-centered relative gain $C/N_{0,i}(t) - \overline{C/N_{0,\text{epoch}}}(t)$, then search candidate yaw angles for maximum expected gain.

2. **Feature Selection (Keep Strictly Orthogonal Physics):**

   - Line of sight: $s_{\text{body},x}, s_{\text{body},y}, s_{\text{body},z}$ (from 3D rotation matrix).
   - Sky clearance: $\sin(\text{elevation})$, $\cos(\theta_{\text{inc}})$.
   - Atmospheric path: Kasten-Young airmass ($\text{airmass}_{\text{ky}}$).
   - Frequency: Relative Friis path loss ($\text{freq\_loss\_db}$).
   - Constellation: One-hot encoded indicators (GPS, GLONASS, BeiDou, Galileo, NavIC, QZSS, SBAS).
   - Dynamics: Causal backward lags ($t-1\text{s}, t-5\text{s}, t-15\text{s}$) and rolling $\sigma(C/N_0)_{5\text{s}}$.

3. **Data Cleaning Rules:**

   - Drop rows where $C/N_0 = 0.0$ (GLONASS MTK bug).
   - Clamp $\text{ElevationDegrees}$ to $[0, 90]$.
   - Clip $\text{PseudorangeRate}$ to $[-1200, +1200]\text{ m/s}$.
   - Canonicalize quaternions: if $w < 0$, multiply $(x,y,z,w)$ by $-1$.
   - Drop `Agc` table; use `Raw.AgcDb` only for Qualcomm/MTK sessions.

4. **Validation Architecture:**
   - 4-fold `GroupKFold` grouped by calendar recording day (`UnixTimeMillis // 86400000`).
   - Leave-one-device-out cross-validation to test cross-chipset generalization.

---

## 5. Top 5 Most Important Experiments to Run Next

1. **Experiment 1: Elevation-Dependent Nominal Baseline Subtraction**

   - Fit an open-sky polynomial $C/N_{0,\text{nom}}(\theta_{\text{el}}) = a + b \sin(\theta_{\text{el}}) + c \sin^2(\theta_{\text{el}})$.
   - Train the orientation model on the residual $C/N_0 - C/N_{0,\text{nom}}$.
   - _Success Criterion:_ Residual $R^2$ improves by $>0.05$ as elevation variance is removed.

2. **Experiment 2: Antenna Non-Linear Directivity Polynomials**

   - Add $s_{bz}^2$, off-axis radius $r_{xy} = \sqrt{s_{bx}^2 + s_{by}^2}$, and hemisphere mask $\mathbb{I}(s_{bz}>0)$ to the Ridge feature vector.
   - _Success Criterion:_ Captures the physical cardioid radiation lobe without requiring decision trees; verified via Python-Kotlin parity test.

3. **Experiment 3: Causal Rolling Fading-Precursor Features**

   - Compute rolling 5s standard deviation $\sigma(C/N_0)_{5\text{s}}$ and 5s linear slope $d(C/N_0)/dt$.
   - Test on fading epochs ($|\Delta C/N_0| > 1.5\text{ dB}$).
   - _Success Criterion:_ Temporal model achieves $>20\%$ lower MAE than persistence on the fading subset.

4. **Experiment 4: Learning-to-Rank (LambdaMART) for Satellite Selection**

   - Train LightGBM with `objective="lambdarank"` using future $C/N_0$ relevance buckets, grouped by `(session, epoch)`.
   - Compare Top-1/Top-3 hit rate and regret against regression-then-argmax.
   - _Success Criterion:_ Mean regret drops from $6.90\text{ dB-Hz}$ to $<5.0\text{ dB-Hz}$.

5. **Experiment 5: Cross-Device Gain Normalization**
   - Compute median high-elevation ($>60^\circ$) GPS L1 $C/N_0$ per device model; subtract as a static front-end gain offset.
   - Evaluate leave-one-device-out generalization error.
   - _Success Criterion:_ Cross-device transfer MAE decreases by $>2.0\text{ dB-Hz}$.
