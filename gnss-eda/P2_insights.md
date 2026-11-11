# P2 Insights & Empirical Findings

Comprehensive running findings from the Phase 2 Relationship-focused EDA. Each insight links directly to an exported figure and tabular evidence, details statistical confidence, outlines concrete implications for machine learning model design, and evaluates alignment or contradictions with Phase 1 conclusions.

---

### INS-01: OrientationDeg.pitchDeg Conclusively Mapped to Physical Gravity Tilt
- **Statement:** The logged column `OrientationDeg.pitchDeg` directly measures device elevation tilt (rotation around the lateral $X$-axis) bounded within $[-89°, +89°]$, perfectly matching accelerometer gravity tilt $\arctan2(-a_y, \sqrt{a_x^2 + a_z^2})$.
- **Evidence:** Figure `01_pitch_roll_convention.png` and Table `01_pitch_roll_convention.csv`. Across 4 phone models, data points cluster tightly along the $y = x$ identity line ($r > 0.98$).
- **Confidence:** HIGH (verified across 4 hardware chipsets and 95 sessions).
- **Implication for ML:** The column name `pitchDeg` in `OrientationDeg.parquet` is physically accurate. There is no axis flip with `rollDeg`.
- **Phase 1 Alignment:** Confirmed Phase 1 finding Q-006 and corrected the initial external documentation hypothesis.

### INS-02: Euler Yaw and FOP Heading Diverge by 45–120° at High Pitch
- **Statement:** When absolute phone pitch exceeds $70°$, Euler angle yaw representation diverges by a median of $45°–120°$ (and up to $175°$ at the 95th percentile) from fused quaternion heading (`FOP.headingDegrees`).
- **Evidence:** Figure `02_yaw_fop_gimbal_lock.png` and Table `02_yaw_fop_gimbal_lock.csv`. Error is $< 4°$ when flat ($0°–30°$) but curves upward asymptotically beyond $60°$.
- **Confidence:** HIGH for the divergence observation. The cause is likely a combination of Euler gimbal lock singularities and Android's documented behavior of changing the heading reference axis when the phone is upright. The 45–120° gap at high pitch may partly reflect a coordinate frame definition difference, not only mathematical singularity.
- **Implication for ML:** Handheld smartphones are frequently held at $60°–85°$ pitch. Models using Euler yaw/pitch features will experience degradation in real-world use. 3D body-frame unit vectors derived from quaternions ($\mathbf{s}_{\text{body}} = \mathbf{R} \cdot \mathbf{s}_{\text{enu}}$) are more stable.
- **Phase 1 Alignment:** Confirms and empirically quantifies Phase 1 Finding 5; adds coordinate-frame caveat.

### INS-03: Experimental Orientation Space Has Major Oblique Blind Spots
- **Statement:** The experimental training corpus does not cover the orientation sphere uniformly; it is heavily concentrated in flat horizontal ($|pitch| < 15°$), steep vertical ($-70°$ to $-85°$), and four discrete rig-aligned azimuth corridors.
- **Evidence:** Figure `03_orientation_coverage.png` and Table `03_orientation_coverage.csv`. Oblique pitches ($30°–60°$) and positive pitches have near-zero sample density.
- **Confidence:** HIGH.
- **Implication for ML:** Unbounded orientation recommendation models may output arbitrary predictions in these unconstrained regions. Model inference must restrict recommendations to the convex hull of observed training attitudes.
- **Phase 1 Alignment:** Expands Phase 1 session inventory into a quantitative 2D density map.

### INS-04: Candidate Signal Keys Require CodeType to Prevent Multi-Channel Fan-Out
- **Statement:** Dual-frequency wideband signals (GPS L5, Galileo E5a) broadcast simultaneous in-phase ($I$) and quadrature ($Q$) channels. The distribution mode shifts from 62 to 66 candidates per epoch when accounting for CodeType.
- **Evidence:** Figure `04_candidates_per_epoch_codetype.png` and Table `04_candidates_per_epoch_codetype.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** The composite primary key for any per-signal prediction, join, or lag must be `(session, epoch, ConstellationType, Svid, CarrierFrequencyHz, CodeType)`. Omitting `CodeType` causes duplicate joins or accidental aggregation.
- **Phase 1 Alignment:** Confirmed Phase 1 Q-002.

### INS-05: Device Model Is Strictly Confounded with Calendar Recording Date
- **Statement:** Sessions for each device model are clustered on specific calendar days (e.g. Samsung SM-M146B on Oct 25–27, OnePlus CPH2401 on June 4 and Oct 19–21).
- **Evidence:** Figure `05_session_timeline.png` and Table `05_session_timeline.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** Cross-validation folds must be grouped by calendar day (`GroupKFold` on recording date). Standard random or session-split CV will leak identical satellite geometries and atmospheric conditions between train and validation.
- **Phase 1 Alignment:** Confirms Phase 1 Finding 9.

### INS-06: 56.4% of C/N0 Variance Governed by Static Physics and Hardware
- **Statement:** A sequential ANOVA variance decomposition reveals that device hardware gain explains 21.3% of total $C/N_0$ variance, satellite elevation explains 0.75% (when added after device), constellation type explains 0.4%, and epoch common-mode signal level explains 34.31%. Per-satellite spatial and multipath residuals comprise 65.69%.
- **Evidence:** Figure `07_variance_decomposition.png` and Table `07_variance_decomposition.csv`.
- **Confidence:** HIGH. Note: The sequential ANOVA shares depend on term order. When device is entered first, it absorbs variance that elevation would otherwise explain.
- **Implication for ML:** Device LNA offset and epoch-level common-mode signal explain most of the variance (55.6% combined). Elevation alone explains <1% after accounting for device. The addressable signal for orientation optimization and antenna directivity is within the 65.69% per-satellite residual. Per-epoch centering removes 34.31% of variance but preserves the signal needed for satellite ranking.
- **Phase 1 Alignment:** Revises Phase 1 expectations with exact quantitative sequential ANOVA shares. Elevation's low incremental contribution after device suggests strong confounding.

### INS-07: Device Model Shows 14.5 dB C/N0 Offset, Confounded with Recording Date
- **Statement:** Under identical clean conditions (GPS L1 satellites at $>60°$ elevation), Samsung SM-M146B maintains a median of 42.4 dB-Hz, compared to 28.1 dB-Hz on OPPO CPH2325—a clean 14.3 dB gap with $< 1.8\text{ dB}$ intra-device session spread.
- **Evidence:** Figure `10_device_gain_benchmark.png` and Table `10_device_gain_benchmark.csv`.
- **Confidence:** HIGH for the observation; LOW for causality attribution. Device model is strictly confounded with calendar recording date (INS-05). This offset could reflect hardware LNA/DSP gain calibration differences, but could also reflect atmospheric conditions, satellite constellation geometry, or other date-correlated factors.
- **Implication for ML:** Unnormalized cross-device models are invalid. A single per-device scalar offset or epoch-relative target centering ($y = C/N_0 - \overline{C/N_{0,\text{epoch}}}$) removes this gap empirically, regardless of root cause. Claiming intrinsic hardware causation requires controlled same-day cross-device comparisons.
- **Phase 1 Alignment:** Confirms Phase 1 Finding 2 observation; adds confounding caveat not present in Phase 1.

### INS-08: BasebandCn0DbHz Is Redundant with Cn0DbHz ($r > 0.995$)
- **Statement:** Antenna `Cn0DbHz` is strictly linear with `BasebandCn0DbHz`, separated by a fixed $3.1–4.8\text{ dB}$ DSP correlator processing gain.
- **Evidence:** Figure `09_baseband_vs_antenna_cn0.png` and Table `09_baseband_vs_antenna_cn0.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** Never include both `Cn0DbHz` and `BasebandCn0DbHz` as linear features; their collinearity causes matrix ill-conditioning. Use the difference as a tracking quality indicator.
- **Phase 1 Alignment:** Confirms Phase 1 collinearity warnings.

---

### INS-09: Smartphone Antenna Directivity Shows ~7 dB Boresight-to-Back Attenuation
- **Statement:** Android body-frame +Z axis points **out of the screen** (toward the user's face when viewing the phone). Signals incident along this axis show +2.8 dB residual gain (10–20° off-axis bin) relative to elevation baseline; signals from the rear hemisphere (130–150° off-axis) show -4.2 dB attenuation (a ~7 dB dynamic range).
- **Evidence:** Figure `12_antenna_pattern_directivity.png`, Table `12_antenna_pattern_body.csv`.
- **Confidence:** MEDIUM. The pattern is noisy and not strictly monotonic; peak gain appears in the 10–20° bin rather than exactly at boresight (0°). Data is from a single device (OnePlus CPH2401) during controlled June orientation sessions.
- **Implication for ML:** Validates using $s_{bz} = \cos(\alpha)$ (dot product with sky-to-satellite ENU unit vector) as a directivity feature. The non-zero peak suggests either antenna element offset from the geometric center, or measurement noise in the orientation quaternion.
- **Phase 1 Alignment:** Partial confirmation; corrects +Z axis description (screen-facing, not rear-facing).

---

### INS-10: Epoch-Mean Signal Variance Accounts for Only 20.5% of Orientation Variance
- **Statement:** Across 14 orientation sessions, the variance of epoch-mean $C/N_0$ is 6.20 $\text{dB}^2$ (standard deviation 2.49 dB), whereas the variance of per-satellite deviations within an epoch is 24.12 $\text{dB}^2$ (standard deviation 4.91 dB).
- **Evidence:** Figure `14_epoch_mean_vs_orientation.png`, Table `14_epoch_mean_vs_orientation.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** If models are trained on epoch-centered residuals $C/N_0 - \overline{C/N_0}_{\text{epoch}}$, they lose the common-mode attenuation from phone tilting/blocking, but preserve 79.5% of the total variance that discriminates individual satellites. For satellite ranking and selection, per-satellite relative residuals are the optimal target.
- **Phase 1 Alignment:** EXTENDS Phase 1 findings on common-mode noise.

---

### INS-11: Severe Collinearity Between Elevation Representations ($\rho = -0.99$)
- **Statement:** Spearman correlation between $\sin(\theta_{\text{elev}})$ and Kasten-Young airmass $X(\theta)$ is $-0.99$, and simple airmass is $-0.97$.
- **Evidence:** Figure `16_feature_correlations.png`, Table `16_feature_correlations.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** Linear models (Ridge) will suffer variance inflation and unstable weights if both $\sin(\theta)$ and airmass are included. Airmass must be dropped in favor of $\sin(\theta)$.
- **Phase 1 Alignment:** Explains historical coefficient instability observed during initial Ridge regressions.

---

### INS-12: Mean Reversion Slope of 0.92–0.94 Governs 15-Second Signal Dynamics
- **Statement:** Across all four devices, the linear regression slope of conditional future signal $\mathbb{E}[C/N_0(t+15\text{s}) \mid C/N_0(t)]$ is strictly less than 1.0 (ranging from $0.915$ on OPPO to $0.941$ on OnePlus CPH2487).
- **Evidence:** Figure `18_temporal_15s_persistence.png`, Table `18_temporal_15s_persistence.csv`.
- **Confidence:** HIGH. Validated across tens of thousands of lagged track pairs.
- **Implication for ML:** Pure persistence is mathematically biased on extreme values. Adding a simple mean-reversion linear term $\hat{\Delta} = -\beta (C/N_0(t) - \mu_{\text{elev}})$ with $\beta \approx 0.08$ immediately reduces RMSE.
- **Phase 1 Alignment:** Confirms and formalizes Phase 1 Finding 3.

---

### INS-13: Over 84% of Epochs Are Static; Fading Concentrates at Low Elevation (<20°)
- **Statement:** More than $84\%$ of signal tracks experience changes $|C/N_0(t+15\text{s}) - C/N_0(t)| \le 1.5\text{ dB-Hz}$. For high-elevation satellites ($\ge 45^\circ$), fading occurs in only $13.4\%–21.6\%$ of epochs, whereas low-elevation satellites ($<20^\circ$) experience fading in $24.2\%–45.0\%$ of epochs (table 19, fading_pct column).
- **Evidence:** Figure `19_temporal_15s_fading_shares.png`, Table `19_temporal_15s_fading_shares.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** Standard MSE loss is dominated by the large squared errors on the 16% fading epochs, not the 84% static ones. Fading events have high variance (std 3.99–4.54 dB vs 0.43–1.21 dB for static), so their squared errors dominate the loss surface. Models must use sample weighting, separate fading classifiers, or robust loss functions to avoid fitting only the fading tail.
- **Phase 1 Alignment:** Confirms Phase 1 fading shares; corrects loss-gradient reasoning.

---

### INS-14: Track Dropouts Create 7.6% Survivorship Bias at Low Elevation
- **Statement:** At high elevation ($\ge 45^\circ$), track survival probability over 15 seconds is $99.8\%$. At low elevation ($<20^\circ$), survival drops to $92.4\%$ at 15s and $88.1\%$ at 30s.
- **Evidence:** Figure `21_temporal_track_survival.png`, Table `21_temporal_track_survival.csv`.
- **Confidence:** HIGH.
- **Implication for ML:** Evaluating models purely on retained pairs hides signal loss. Selection policies must incorporate an elevation threshold or survival probability penalty.
- **Phase 1 Alignment:** Directly addresses Open Question Q-008.

---

### INS-15: Past 5-Second Jitter ($\sigma_{5\text{s}}$) Predicts Fading (AUC = 0.74), Elevation Confounding Unknown
- **Statement:** The past 5-second standard deviation of signal strength $\sigma_{5\text{s}}$ shifts from a median of $0.18\text{ dB-Hz}$ in stable epochs to $0.46\text{ dB-Hz}$ prior to fading events, achieving a classification AUC of $0.74$ in a small sample (4 hand-picked sessions, NB04).
- **Evidence:** Figure `24_temporal_precursor_fading.png`, Table `24_temporal_precursor_fading.csv`.
- **Confidence:** MEDIUM. Low-elevation signals are both noisier (higher $\sigma_{5\text{s}}$) and fade more often (INS-13). The AUC may partly reflect elevation confounding rather than independent predictive value. Comparison of elevation-only vs elevation+$\sigma_{5\text{s}}$ on held-out sessions is needed (Q-026).
- **Implication for ML:** Rolling dispersion features may give models predictive foresight beyond static persistence, but the incremental value over elevation must be quantified before claiming independent utility.
- **Phase 1 Alignment:** Confirms Recommendation R-03 from Phase 1; adds confounding caveat.

---

### INS-16: Selection Headroom (Best vs Median) Is 9–11 dB; Mean Regret vs Persistence Is 0.16 dB
- **Statement:** Across tracked candidate pools averaging $60–66$ satellites per epoch, the gap between the best future satellite and the **median** candidate is $9.2\text{ to }9.7\text{ dB}$ (table 25). However, a persistence policy (always pick the current strongest signal) achieves **0.16 dB mean regret** and 0.0 dB median regret on a single held-out test session (table 26).
- **Evidence:** Figure `25_selection_headroom.png`, `26_selection_regret_benchmarks.png`; Tables `25_selection_headroom.csv`, `26_selection_regret_benchmarks.csv`.
- **Confidence:** HIGH for headroom measurement; MEDIUM for regret (single-session test). The 9+ dB headroom is relative to the median candidate (rank ~30); the ML challenge is beating the **already-strong** persistence baseline by closing the 0.16 dB mean regret (driven by rare fading events).
- **Implication for ML:** Establishes that satellite selection has large total dynamic range (9+ dB), but persistence captures most of it. ML models must demonstrate <0.16 dB mean regret with confidence intervals across multiple held-out days, not just beat random selection.
- **Phase 1 Alignment:** Refines Phase 1 selection motivation with headroom vs regret distinction.

---

### INS-17: Persistence Achieves 0.16 dB Mean Regret on a Single Held-Out Test Session
- **Statement:** On a held-out test session (922 epochs), picking the highest current signal (Persistence) achieves $0.16\text{ dB}$ mean regret, $0.0\text{ dB}$ median regret, and a $94.8\%$ hit rate within 0.5 dB of optimal future signal. A simple hybrid (Persistence + Elevation) achieves $0.15\text{ dB}$ mean regret and $92.4\%$ top-1 accuracy (table 26).
- **Evidence:** Figure `26_selection_regret_benchmarks.png`, Table `26_selection_regret_benchmarks.csv`.
- **Confidence:** MEDIUM (single session, 922 epochs). Confidence intervals and replication across multiple held-out days are needed before generalizing.
- **Implication for ML:** Persistence is a very strong baseline. ML models must demonstrate statistically significant reduction below 0.16 dB mean regret across multiple held-out calendar days with reported confidence intervals. Single-session or hand-picked test sets will overstate model value.
- **Phase 1 Alignment:** Refines Phase 1 LTR hypothesis with actual measured regret and strong baseline performance.

---

### INS-18: Signal Dropout Contribution to Regret Requires Expected-Value Formulation
- **Statement:** Table 28 shows 898 continuously-tracked epochs with 141.3 dB total regret (0.16 dB mean), and 24 dropout epochs with 0.0 dB regret. Dropouts represent 2.6% of epochs (24/922), not 0.8%. The regret calculation assumed the chosen satellite survived; if it dropped out, the actual realized C/N0 would fall to a sensitivity floor (~12 dB-Hz assumed), generating large regret.
- **Evidence:** Figure `28_regret_dropout_decomposition.png`, Table `28_regret_dropout_decomposition.csv`.
- **Confidence:** MEDIUM. The "16.4% of regret" claim requires an assumed floor value (12 dB-Hz) and an expected-value calculation not present in table 28.
- **Implication for ML:** Selection policies should incorporate survival probability via expected value: $\mathbb{E}[\text{realized C/N0}] = P(\text{survive}) \cdot \hat{C/N_0} + (1 - P(\text{survive})) \cdot \text{floor}$, where the floor is the minimum tracking sensitivity. The dropout share of regret depends on the choice of floor.
- **Phase 1 Alignment:** Supports Phase 1 survivorship warnings; clarifies expected-value formulation.

---

### INS-19: Residual Drift Shows Similar Distributions Across Devices (Not Formally Tested for Invariance)
- **Statement:** While raw signal levels exhibit an extreme $14.5\text{ dB}$ hardware gap across phones (Samsung $32.8\text{ dB-Hz}$ vs OPPO $24.2\text{ dB-Hz}$ mean in table 29), the 15-second residual change $\Delta C/N_0$ is **observed to be similar** across manufacturers: mean $\approx 0.0–0.1\text{ dB}$, std ranges 1.145 dB (CPH2487) to 3.993 dB (CPH2325). Mean-reversion also pulls toward each device's own baseline level, so the conditional delta is not strictly device-invariant.
- **Evidence:** `figures/29_cross_device_covariate_shift.png`, `tables/29_cross_device_covariate_shift.csv`.
- **Confidence:** HIGH for the observation; claim of "statistical invariance" is not supported by formal hypothesis testing (no Kolmogorov-Smirnov, permutation test, or confidence intervals reported).
- **Implication for Modeling:** Residual drift formulation reduces but does not eliminate cross-device covariate shift. Using C/N0 **relative to per-device baseline** (e.g., $C/N_0 - \mu_{\text{device},\text{elev}}$) as a feature may improve zero-shot transferability. Leave-one-device-out validation is still required.
- **Phase 1 Alignment:** Reinforces Recommendation 2 from Phase 1; softens "statistically invariant" claim to "observed similar".

---

### INS-20: Satellite Constellation Overlap Mandates Calendar-Day Grouped Cross-Validation
- **Statement:** Sessions recorded within 2 hours of each other share $92.4\%$ of visible satellites in identical sky coordinates. Random train/test splits result in catastrophic data leakage from orbital geometric memory.
- **Evidence:** `figures/30_sky_overlap_leakage.png`, `tables/30_sky_overlap_leakage.csv`.
- **Confidence:** HIGH. Grounded in orbital mechanics and verified via pair-wise Jaccard constellation similarity.
- **Implication for Modeling:** All cross-validation must be grouped strictly by calendar day (`GroupKFold` on recording date). Random shuffling or session-level random splits will drastically overstate real-world accuracy.
- **Phase 1 Alignment:** Confirms Recommendation 8 from Phase 1.

---

### INS-21: BasebandCn0DbHz Is Universal; AgcDb Is Null on Samsung Only
- **Statement:** `BasebandCn0DbHz` is **non-null and populated on all 4 devices** (OnePlus CPH2401, CPH2487, OPPO CPH2325, Samsung SM-M146B), verified via direct DuckDB query on Raw.parquet 2026-11-11. `Raw.AgcDb` is **null on Samsung SM-M146B only**; it is populated (with device-specific distributions) on the three OnePlus/OPPO devices. A separate `Agc` **table** (not column) exists only in Samsung logs.
- **Evidence:** Direct query results (2026-11-11) on `gnss-eda/Stats/scratch/typed/Raw.parquet` joined with `sessions_meta.csv`. Figure `31_device_feature_availability.png` and table may reflect an earlier or incomplete analysis.
- **Confidence:** HIGH. Contradicts the earlier summary and INS-21 draft, which reversed the availability.
- **Implication for Modeling:** `BasebandCn0DbHz` can be used across all devices. `Raw.AgcDb` should be excluded or imputed for Samsung. The previous claim that both features are "Samsung-only" was incorrect and would have caused models to discard a universal feature.
- **Phase 1 Alignment:** Contradicts an error in Phase 1 or Phase 2 draft feature matrices; corrected via re-verification.



