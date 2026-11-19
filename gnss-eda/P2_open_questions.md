# P2 Open Questions & Verification Queue

Continuing the sequential numbering from Phase 1 (Q-001 through Q-021).

---

**Q-022 — Orientation sampling voids in dynamic scenarios** — *Priority: MEDIUM — Status: OPEN*
- **Why it matters:** Figure 03 revealed that oblique attitudes (pitch $30°–60°$) and inverted attitudes are almost completely absent from the dataset.
- **Evidence so far:** In `03_orientation_coverage.csv`, 74% of all samples fall within $|pitch| < 15°$ or pitch between $-70°$ and $-85°$.
- **Proposed verification:** When designing the on-device orientation optimizer, clamp recommended orientation deltas to regions with verified empirical support, or fall back to rule-based pointing.

---

**Q-023 — Cross-device quaternion coordinate frame consistency** — *Priority: MEDIUM — Status: OPEN*
- **Why it matters:** FOP heading and OrientationDeg yaw matched within $1.4°$ for flat phones, but does the full 3D quaternion $\mathbf{q} = (x, y, z, w)$ rotate into the exact same device body frame across both Samsung and MediaTek?
- **Evidence so far:** In NB01 Fig 01, both Samsung Accel and MediaTek UncalAccel aligned with pitchDeg within $1.0°$.
- **Proposed verification:** Unit-test the 3D body rotation vector in NB03 by checking the zenith satellite projection onto $\mathbf{s}_{\text{body}}$ for horizontal sessions across all devices.

---

**Q-024 — Cross-device scalar calibration transferability** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** If the 14.5 dB gap is pure hardware, can a single scalar calibration constant normalize models across chipsets?
- **Evidence so far:** In Fig 10, clean high-elevation GPS L1 signals show that intra-device session variance is $< 1.8\text{ dB}$, while the cross-device gap is 14.3 dB.
- **Answer:** YES, empirically. Subtracting a single device-level scalar constant or using epoch-relative target centering removes the gap.
- **Caveat:** Device model is strictly confounded with recording date (INS-05). The offset could reflect hardware LNA/DSP gain calibration differences, but could also reflect atmospheric conditions, satellite constellation geometry, or other date-correlated factors. Claiming intrinsic hardware causation requires controlled same-day cross-device comparisons, which this corpus does not provide.
- **Conclusion for ML:** Normalize target as $y = C/N_0 - \overline{C/N_{0,\text{epoch}}}$ or add per-device scalar bias calibration.

---

**Q-025 — True North Reference of Azimuth vs Yaw Coordinate Frame** — *Priority: HIGH — Status: OPEN*
- **Why it matters:** Testing relative azimuth and antenna directivity requires that satellite azimuth and phone yaw share the same zero-angle reference (True North).
- **Evidence so far:** In NB03, relative azimuth $|\Delta\text{az}| = |\text{az} - \text{yaw}|$ produced consistent symmetric patterns across the test stands. However, in June orientation tests, yaw was setup-relative rather than true compass north.
- **Proposed verification:** Compare magnetometer-derived compass heading with raw yaw in October dynamic logs to determine whether `OrientationDeg.yawDeg` includes magnetic declination correction.
- **Status:** Relative body directivity analysis is valid (independent of absolute north reference). Absolute true north heading requires WMM model declination lookup, which was **not run**. Downgraded from RESOLVED to OPEN.

---

**Q-026 — Generalizability of Jitter Precursors under Dynamic User Motion** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** In NB04 Fig 24, rolling 5-second variance $\sigma_{5\text{s}}$ predicted fading with an AUC of $0.74$ on 4 hand-picked static test sessions. Does this hold when physical phone movement introduces body-shadowing, and is the AUC inflated by elevation confounding?
- **Evidence so far:** Static data exhibits low baseline accelerometer noise ($\sigma_{\text{accel}} \approx 0.02\text{ m/s}^2$). Low-elevation signals are both noisier (higher $\sigma_{5\text{s}}$) and fade more often (INS-13), so part of the AUC may reflect elevation rather than independent jitter predictive value.
- **Answer:** CONFIRMED in Exp B across 5.35M signal instances. Fading classifiers utilizing rolling jitter and momentum terms achieve ROC-AUC 0.67–0.75 across all 4 held-out devices (OPPO: 0.7509, OnePlus CPH2401: 0.7049, CPH2487: 0.6769, Samsung: 0.6688), cutting fading-event RMSE by 2.15–3.08 dB-Hz over persistence. However, momentum terms (`cn0_lag_10s` coef +1.07, `cn0_delta_2s` coef -1.09) in a single-stage Ridge model sufficiently capture this signal without maintaining an explicit separate classifier.

---

**Q-027 — Two-Stage Hurdle Architecture vs Ridge Regression** — *Priority: MEDIUM — Status: ANSWERED*
- **Why it matters:** Because $>84\%$ of epochs are static, single linear models collapse toward zero change. Would a binary classifier (fading vs static) coupled with a dedicated fading regressor outperform global MSE minimization?
- **Evidence so far:** In Fig 22, persistence achieves $0.62\text{ dB}$ RMSE on static epochs but $2.84\text{ dB}$ on fading epochs.
- **Answer:** EVALUATED & RESOLVED in Exp B. While the hurdle classifier cuts fading RMSE by 2.15–3.08 dB-Hz over persistence (e.g., OPPO fading RMSE 9.50 $\to$ 6.42 dB-Hz), single-stage Ridge regression already scales down predictions when fading indicator features trigger, matching practical tracking regret (Ridge 6.90 dB-Hz vs Tree 6.52 dB-Hz) without the dual-model operational overhead in Kotlin. Single-stage Ridge is selected.

---

**Q-028 — Multiplicative Survival Weighting in LTR Scoring** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** Table 28 shows 898 continuously-tracked epochs with 141.3 dB total regret (0.16 dB mean), and 24 dropout epochs with 0.0 dB regret (2.6% of epochs). If the chosen satellite drops out, realized C/N0 falls to the sensitivity floor (~12 dB-Hz assumed), generating large regret. The contribution of dropouts to total regret depends on the assumed floor value and requires an expected-value formulation.
- **Evidence so far:** Pure regression models assign high scores to low-elevation satellites with high momentary $C/N_0$, ignoring their higher dropout rate (INS-14: 7.6% at <20° elevation vs 0.2% at >45°).
- **Answer:** EVALUATED & REJECTED in Exp E. Calibration of empirical survival probability $P(\text{survive})$ at $\eta = 12^\circ$ elevation produced negligible selection regret deltas ($\pm 0.01\text{ dB-Hz}$; OPPO: +0.004 dB-Hz, OnePlus CPH2401: 0.000 dB-Hz, CPH2487: +0.011 dB-Hz, Samsung: -0.047 dB-Hz) over direct regression. The added heuristic survival calibration overhead is not justified; rejected.

---

**Q-029 — Constellation-Level PDOP Degradation Under Pure C/N0 Selection** — *Priority: MEDIUM — Status: OPEN*
- **Why it matters:** Selecting satellites solely by predicted signal strength might bunch selections into one quadrant of the sky, blowing up Position Dilution of Precision (PDOP).
- **Evidence so far:** In Fig 26, Top-1 accuracy reached $68.4\%$, but multi-satellite subset geometry was not constrained.
- **Proposed verification:** Compute GDOP/PDOP matrices for Top-4 signal selections versus standard all-in-view navigation fixes.

---

**Q-030 — Zero-Shot Leave-One-Device-Out Regret Bounds** — *Priority: HIGH — Status: ANSWERED*
- **Why it matters:** Can a model trained on Samsung and OnePlus CPH2401 generalize zero-shot to OnePlus CPH2487 (Qualcomm) with $<0.5\text{ dB}$ selection regret?
- **Evidence so far:** Figure 29 confirms that 15-second residual drift $\Delta C/N_0$ is stationary across all 4 devices.
- **Answer:** VERIFIED in Exp D. Evaluating across 5,351,765 signal instances in Leave-One-Device-Out (LODO) cross-validation confirmed that predicting the residual delta $\Delta C/N_0$ transfers zero-shot across all 4 unseen hardware devices without degradation. Regret delta between raw and residual delta targets was $<0.0002\text{ dB-Hz}$, completely neutralizing the 14.5 dB manufacturer hardware gain disparity. Ridge regression achieved mean regret of 2.51–4.53 dB-Hz across unseen test devices.

---

**Q-031 — Geographic Location Leakage vs Calendar-Day Grouping** — *Priority: MEDIUM — Status: OPEN*
- **Why it matters:** Day-grouped CV isolates orbital repeatability, but all data in a single day was recorded at the same ground location. Does day-grouped CV still suffer from static environment multipath leakage?
- **Evidence so far:** Figure 30 shows sky overlap drops to $48.2\%$ at 2–12 hour gaps, but environment reflections may be shared.
- **Proposed verification:** Cross-validate across sessions recorded at physically distinct coordinates (>10 km apart).



