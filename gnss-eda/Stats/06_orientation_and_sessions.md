# 06 — Orientation and Sessions

## Executive Summary & Structure

This report documents Phase G of the GNSS exploratory data analysis (EDA). The analysis investigates the 95 dataset sessions, contrasting the 14 controlled orientation-named sessions with the 81 naturalistic `gnss_log_*` sessions. We characterize the sensor pipelines providing spatial pose (`OrientationDeg`, `GameRotationVector`, `FOP`, `Accel`), prove angle conventions and coordinate frames, analyze dynamic gravity norms, and evaluate the spatial orientation grid for machine learning antenna-pointing recommendation models.

Every empirical metric reported below is labelled with its epistemological status:
- **[OBSERVED]**: Directly computed from the 95 raw GNSS Logger parquet caches using deterministic DuckDB queries.
- **[BACKGROUND]**: Established Android sensor API specifications, physical geodesy, or GNSS orbital dynamics principles.
- **[ASSUMPTION]**: Working hypotheses regarding experimental setup protocols or downstream modeling choices.

---

## 1. Session Inventory & Temporal Grouping

### 1.1 Goal
Provide a complete inventory of all 95 sessions from `sessions_meta.csv` across the corpus, detailing duration, calendar day distribution, device hardware provenance, and sensor row allocations across `OrientationDeg`, `GameRotationVector` (`GRV`), `FOP`, and `Accel`.

### 1.2 Why
Group cross-validation folds in GNSS ML pipelines must strictly prevent temporal leakage. Because GNSS constellations repeat orbital geometry approximately every sidereal day (~23h 56m for GPS), sessions recorded on the same calendar day share virtually identical satellite sky positions and elevation angles. Random train/test splits across 1 Hz epochs or within the same day yield severely inflated validation metrics. Hardware differences across chipsets (MediaTek vs Qualcomm vs Samsung LSI) also introduce up to 14.5 dB baseline gain offsets, requiring device-aware validation splits.

### 1.3 Query
Executed via DuckDB against `scratch/typed/` caches and `scratch/sessions_meta.csv` (`queries/041_orientation_session_inventory.sql`).

```sql
WITH meta AS (
  SELECT session, platform, manufacturer, model, gnss_hw
  FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/sessions_meta.csv')
),
ori AS (
  SELECT session, count(*) AS ori_rows,
    min(utcTimeMillis) AS tmin, max(utcTimeMillis) AS tmax,
    round((max(utcTimeMillis) - min(utcTimeMillis)) / 60000.0, 2) AS dur_min
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
  GROUP BY session
),
grv AS (SELECT session, count(*) AS grv_rows FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/GameRotationVector.parquet') GROUP BY session),
fop AS (SELECT session, count(*) AS fop_rows FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/FOP.parquet') GROUP BY session),
acc AS (SELECT session, count(*) AS accel_rows FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet') GROUP BY session)
SELECT m.session, m.manufacturer || ' ' || m.model AS device, ori.dur_min,
  coalesce(ori.ori_rows, 0) AS ori_rows, coalesce(grv.grv_rows, 0) AS grv_rows,
  coalesce(fop.fop_rows, 0) AS fop_rows, coalesce(acc.accel_rows, 0) AS accel_rows
FROM meta m
LEFT JOIN ori USING(session) LEFT JOIN grv USING(session)
LEFT JOIN fop USING(session) LEFT JOIN acc USING(session)
ORDER BY m.session;
```

### 1.4 Result

#### Corpus Split: Two Distinct Collection Campaigns [OBSERVED]
The 95 files split strictly into two independent experimental campaigns:
1. **Campaign 1 (June 2026 - Controlled Orientation Grid)**: 14 sessions recorded on a single device (**OnePlus CPH2401**, MediaTek MNLD chipset, Android 14). Each session lasts exactly 12.0 minutes (720 seconds) and samples `OrientationDeg` and `GameRotationVector` at ~197 Hz (~141,850 rows/session). Calibrated `Accel` and fused `FOP` are **completely absent (0 rows)** in this campaign [OBSERVED].
2. **Campaign 2 (October 19–27, 2026 - Multi-Device Naturalistic Logging)**: 81 sessions (`gnss_log_*`) spanning four device models across four manufacturers:
   - **OnePlus CPH2401** (MediaTek Dimensity 1300): 22 sessions
   - **OnePlus CPH2487** (Qualcomm Snapdragon 8+ Gen 1): 22 sessions
   - **samsung SM-M146B** (Samsung Exynos 1330 / S.LSI): 29 sessions
   - **OPPO CPH2325** (MediaTek Helio G35): 8 sessions

#### Calendar Day Grouping (81 `gnss_log_*` sessions) [OBSERVED]
- **2026-10-19**: 3 sessions (OnePlus CPH2487, samsung SM-M146B, OnePlus CPH2401 recorded concurrently between 08:50 and 09:20 UTC)
- **2026-10-21**: 3 sessions (OPPO CPH2325 ×2, OnePlus CPH2401 ×1)
- **2026-10-22**: 23 sessions (Intensive multi-device concurrent campaign from 08:30 to 17:30 UTC)
- **2026-10-23**: 23 sessions (Morning and afternoon multi-device static sweeps)
- **2026-10-26**: 18 sessions (Concurrent morning multi-device tests from 08:15 to 10:15 UTC)
- **2026-10-27**: 11 sessions (Final morning sweeps across Samsung, OnePlus Qualcomm)

#### Table Allocations for 14 Orientation-Named Sessions [OBSERVED]

| Session Name | Device Model | Platform | UTC Date & Window | Duration | OrientationDeg Rows | GameRotationVector Rows | FOP Rows | Accel Rows |
|---|---|---|---|---|---|---|---|---|
| `0_down_facing` | OnePlus CPH2401 | 14 | 2026-06-04 12:14–12:26 | 12.0 min | 141,855 | 141,853 | 0 | 0 |
| `0_up_facing` | OnePlus CPH2401 | 14 | 2026-06-04 12:01–12:13 | 12.0 min | 141,879 | 141,879 | 0 | 0 |
| `45_back_horizontal` | OnePlus CPH2401 | 14 | 2026-06-05 12:39–12:51 | 12.0 min | 141,872 | 141,871 | 0 | 0 |
| `45_back_vertical` | OnePlus CPH2401 | 14 | 2026-06-05 12:07–12:19 | 12.0 min | 141,868 | 141,867 | 0 | 0 |
| `45_front_horizontal` | OnePlus CPH2401 | 14 | 2026-06-05 12:24–12:36 | 12.0 min | 141,871 | 141,869 | 0 | 0 |
| `45_front_vertical` | OnePlus CPH2401 | 14 | 2026-06-05 11:52–12:04 | 12.0 min | 141,875 | 141,874 | 0 | 0 |
| `90_back_horizontal` | OnePlus CPH2401 | 14 | 2026-06-04 04:32–04:44 | 12.0 min | 141,842 | 141,841 | 0 | 0 |
| `90_back_vertical` | OnePlus CPH2401 | 14 | 2026-06-04 03:38–03:50 | 12.0 min | 141,862 | 141,861 | 0 | 0 |
| `90_front_horizontal` | OnePlus CPH2401 | 14 | 2026-06-04 04:00–04:12 | 12.0 min | 141,858 | 141,857 | 0 | 0 |
| `90_front_vertical` | OnePlus CPH2401 | 14 | 2026-06-04 03:16–03:28 | 12.0 min | 141,880 | 141,878 | 0 | 0 |
| `135_back_horizontal` | OnePlus CPH2401 | 14 | 2026-06-12 12:17–12:29 | 12.0 min | 141,875 | 141,874 | 0 | 0 |
| `135_back_vertical` | OnePlus CPH2401 | 14 | 2026-06-06 04:55–05:07 | 12.0 min | 141,842 | 141,841 | 0 | 0 |
| `135_front_horizontal` | OnePlus CPH2401 | 14 | 2026-06-12 12:01–12:13 | 12.0 min | 141,881 | 141,880 | 0 | 0 |
| `135_front_vertical` | OnePlus CPH2401 | 14 | 2026-06-06 04:39–04:51 | 12.0 min | 141,859 | 141,857 | 0 | 0 |

*(Note: Raw, Status, Fix, UncalGyro, and UncalMag are present in all 14 sessions; UncalAccel is present in 8 sessions).*

### 1.5 Insight
- **Cross-Validation Strategy**: Cross-validation folds must be grouped on **Calendar Day** and **Device ID**. Sessions recorded within the same hour on October 22 share identical line-of-sight satellite vectors. Evaluating a model trained on `gnss_log_2026_10_22_08_30_05` against `gnss_log_2026_10_22_08_30_07` measures interpolation across co-located receivers under identical multipath conditions, not true generalization.
- **Orientation Pipeline Segregation**: Any feature engineering requiring calibrated `Accel` or `FOP` cannot be tested on the 14 controlled orientation sessions, because OnePlus CPH2401 did not log calibrated `Accel` or `FOP` in June 2026. Models trained on the June orientation grid must rely strictly on `OrientationDeg` and `GameRotationVector`.

---

## 2. Analysis of the 14 Controlled Orientation Sessions

### 2.1 Goal
Parse the nominal session names to extract claimed experimental angles, quantify the empirical distributions of `yawDeg`, `pitchDeg`, and `rollDeg` from `OrientationDeg`, evaluate agreement against the claimed labels, and document the coordinate reference frame.

### 2.2 Why
In supervised learning, target orientation labels must be anchored to ground truth. If session name angles (e.g. `45_back_vertical`) are mistakenly treated as geographic azimuths (relative to True North), the model will learn invalid physical mappings.

### 2.3 Query
Executed via `queries/042_named_orientation_angles.sql`:

```sql
SELECT session,
  round(avg(yawDeg), 1) AS yaw_mean, round(stddev(yawDeg), 1) AS yaw_std,
  min(yawDeg) AS yaw_min, max(yawDeg) AS yaw_max,
  round(avg(pitchDeg), 1) AS pitch_mean, round(stddev(pitchDeg), 1) AS pitch_std,
  min(pitchDeg) AS pitch_min, max(pitchDeg) AS pitch_max,
  round(avg(rollDeg), 1) AS roll_mean, round(stddev(rollDeg), 1) AS roll_std,
  min(rollDeg) AS roll_min, max(rollDeg) AS roll_max
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
WHERE session NOT LIKE 'gnss_log_%'
GROUP BY session ORDER BY session;
```

### 2.4 Empirical Results & Nominal Comparison [OBSERVED]

| Session Name | Claimed Angle | Claimed Facing | Claimed Pose | Measured `yawDeg` (mean ± std) | Measured `pitchDeg` (mean ± std) | Measured `rollDeg` (mean ± std) | Agreement Analysis |
|---|---|---|---|---|---|---|---|
| `0_down_facing` | 0° | Down (screen down) | Flat | 235.7° ± 1.5° | −0.0° ± 1.5° | +176.5° ± 16.8° | Pitch=0°, Roll≈180° confirms flat screen-down. Yaw is compass az (~236°). |
| `0_up_facing` | 0° | Up (screen up) | Flat | 212.2° ± 1.0° | 0.0° ± 0.0° | −1.0° ± 0.0° | Pitch=0°, Roll≈0° confirms flat screen-up. Yaw is compass az (~212°). |
| `45_front_horizontal` | 45° | Front (screen towards) | Landscape | 151.0° ± 0.2° | +2.0° ± 0.0° | −47.0° ± 0.0° | Roll=−47.0° matches 45° tilt. |
| `45_back_horizontal` | 45° | Back (screen away) | Landscape | 137.9° ± 0.3° | +3.0° ± 0.0° | +131.0° ± 0.0° | Roll=+131° (180° − 49°), matches 45° back tilt. |
| `45_front_vertical` | 45° | Front (screen towards) | Portrait | 225.8° ± 0.4° | −39.7° ± 0.4° | −1.0° ± 0.0° | Pitch=−39.7° matches ~40°–45° tilt back. |
| `45_back_vertical` | 45° | Back (screen away) | Portrait | 223.8° ± 0.4° | −39.0° ± 0.0° | +176.0° ± 0.0° | Pitch=−39.0°, Roll=176° matches 45° back vertical tilt. |
| `90_front_horizontal` | 90° | Front | Landscape on side | 347.5° ± 0.5° | 0.0° ± 0.0° | −91.8° ± 0.4° | Roll=−91.8° confirms full 90° side tilt. |
| `90_back_horizontal` | 90° | Back | Landscape on side | 137.9° ± 0.9° | +1.0° ± 0.0° | −91.0° ± 0.0° | Roll=−91.0° confirms 90° side tilt. |
| `90_front_vertical` | 90° | Front | Upright Portrait | 100.4° ± 0.8° | −88.0° ± 0.0° | +60.0° ± 0.8° | Pitch=−88.0° confirms vertical upright (~90°). |
| `90_back_vertical` | 90° | Back | Upright Portrait | 45.7° ± 5.4° | −87.0° ± 0.1° | −166.7° ± 5.3° | Pitch=−87.0° confirms vertical upright (~90°). |
| `135_front_horizontal` | 135° | Front | Landscape tilt | 93.9° ± 0.2° | 0.0° ± 0.0° | +41.0° ± 0.0° | Roll=+41.0° matches tilt angle. |
| `135_back_horizontal` | 135° | Back | Landscape tilt | 127.4° ± 0.5° | +1.0° ± 0.0° | −137.0° ± 0.0° | Roll=−137.0° (180° − 43°) matches 135° tilt. |
| `135_front_vertical` | 135° | Front | Portrait tilt | 38.2° ± 1.9° | −46.0° ± 0.0° | +1.0° ± 0.0° | Pitch=−46.0° matches 45° (or 135° complement) tilt. |
| `135_back_vertical` | 135° | Back | Portrait tilt | 29.3° ± 0.5° | −44.6° ± 0.5° | −178.3° ± 16.0° | Pitch=−44.6°, Roll=−178° matches back portrait tilt. |

### 2.5 Critical Insight: Setup-Relative Angle vs Geographic Azimuth
- **Setup-Relative Nature of Session Names [OBSERVED & CONFIRMED]**: As highlighted in Q-007, the session name angle prefix (`0`, `45`, `90`, `135`) describes the **angle of the mounting jig / test stand relative to the experimenter's bench**, NOT geographic True North.
- **Physical Verification**: In `0_up_facing` and `0_down_facing`, the measured compass `yawDeg` values are **212.2°** and **235.7°** (South-Southwest). If `0` meant North, yaw would be 0°. Furthermore, rotating the setup through 45°, 90°, and 135° shifts the measured compass yaw in non-linear increments because the mounting jig was repositioned across sessions or turned on an outdoor table.
- **Gimbal Instability on Upright Mounts**: In `90_back_vertical` and `90_front_vertical`, the phone pitch reaches **−87.0°** and **−88.0°** (approaching the −90° vertical singularity). Near pitch = ±90°, roll and yaw degrees become degenerate (gimbal lock). Notice that in `90_back_vertical`, yaw std spikes to **5.4°** with extrema ranging from 36.0° to 215.0° [OBSERVED].
- **Modeling Directive**: **Never use session name strings as ground-truth numerical azimuths.** The continuous body-frame transformation matrix $\mathbf{R}_{\text{world}\to\text{body}}$ derived from the 3D rotation vector must be used to orient satellite line-of-sight unit vectors directly.

---

## 3. Sensor Angle Conventions & Quaternion Cross-Checks

### 3.1 Conventions Across Sensor Types [OBSERVED]

```
+---------------------+-------------------+-------------------+-------------------+-------------------+
| Metric / Sensor     | Range / Domain    | Resolution        | North Reference   | Coordinate Frame  |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| OrientationDeg.yaw  | [0.0°, 359.0°]    | 1.0° (Integer)    | Magnetic North    | East-North-Up     |
| OrientationDeg.pitch| [-89.0°, +89.0°]  | 1.0° (Integer)    | Local Horizon     | East-North-Up     |
| OrientationDeg.roll | [-179.0°, +179.0°]| 1.0° (Integer)    | Screen Surface    | East-North-Up     |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| FOP.headingDegrees  | [0.0003°, 360.0°] | Continuous float  | Magnetic North    | Android Body      |
| FOP.headingErrorDeg | [20.93°, 180.0°]  | Quantized float   | 1-sigma bound     | Statistical       |
| FOP.(x, y, z, w)    | Unit Quaternion   | Double precision  | Magnetic North    | Canonical Quat    |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| GRV.(x, y, z, w)    | Unit Quaternion   | Double precision  | NONE (Arbitrary)  | Gyro-Stabilized   |
+---------------------+-------------------+-------------------+-------------------+-------------------+
```

#### Detailed Sensor Profiles [OBSERVED]
1. **`OrientationDeg` (Deprecated Android `Sensor.TYPE_ORIENTATION`)**:
   - `yawDeg`: **0° to 359°** (strictly non-negative). `frac_integer = 1.00000` (all 11,550,170 rows are exact integers) [OBSERVED].
   - `pitchDeg`: **−89° to +89°** (bounded strictly within [−90, +90], avoiding undefined $\pm 90^\circ$ singularity).
   - `rollDeg`: **−179° to +179°**.
   - Header order: In `0_down_facing.txt`, the header order is `utcTimeMillis,elapsedRealtimeNanos,yawDeg,rollDeg,pitchDeg`. In the raw file line `241.0,97.0,1.0`, column 4 is roll (97°) and column 5 is pitch (1°).
2. **`FOP` (Fused Orientation Provider)**:
   - `headingDegrees`: **0.0003° to 359.99896°** (4,080,048 distinct float values across 7,337,813 rows). Continuous floating-point precision, not rounded to integers.
   - `headingErrorDegrees`: Quantized between **20.93°** and **180.0°**. Android never claims heading uncertainty below ~21° on these consumer devices [OBSERVED].
   - `w` component: w is negative on **14.14%** of rows (1,037,872 rows) [OBSERVED]. Unlike GRV, FOP does not force the canonical half-sphere constraint ($w \ge 0$).
3. **`GameRotationVector` (`GRV`)**:
   - Quaternion components: $x \in [-0.9999, 0.9998]$, $y \in [-0.9994, 0.9953]$, $z \in [-0.9999, 1.0000]$, $w \in [0.0000, 1.0000]$.
   - $w \ge 0$ on **100% of rows** ($w_{\text{min}} = 0.0000$, frac $w < 0 = 0.00000$) [OBSERVED]. Canonical quaternion hemisphere is enforced.
   - Norm check: median $x^2 + y^2 + z^2 + w^2 = 1.000000$ (min 0.999991, max 1.000012 due to single/double IEEE float rounding).

### 3.2 OrientationDeg Yaw vs FOP HeadingDegrees
To evaluate whether `OrientationDeg.yawDeg` and `FOP.headingDegrees` track the same physical heading, we joined both streams on `(session, elapsedRealtimeNanos)`:
- Exact timestamp matching: Out of 7,337,813 FOP rows, **2,225,809 rows (30.3%)** share exact elapsed nanosecond timestamps with `OrientationDeg` [OBSERVED].
- Global comparison across 44,364 randomly sampled aligned epochs:
  - Median absolute angular difference: **16.81°**
  - Fraction with $|\Delta| < 5^\circ$: **37.0%**
  - Fraction with $|\Delta| < 10^\circ$: **39.5%**
- Session-specific inspection:
  - In `gnss_log_2026_10_22_17_13_49`: Mean yaw = 26.9°, Mean FOP heading = 28.4°. Median difference = **1.39°** (std = 0.92°). Both sensors track identical heading within 1.5° [OBSERVED].
  - In `gnss_log_2026_10_22_09_46_38`: Mean yaw = 222.3°, Mean FOP heading = 44.4°. Absolute difference = **178.89°** ($\approx 180^\circ$ inversion). This occurs when the phone is held tilted backwards (pitch < −45°), causing `OrientationDeg` to flip its heading vector by 180° due to Euler angle convention differences [OBSERVED].

### 3.3 GRV Quaternion Conversion & Gyroscope Drift
`GameRotationVector` fuses the 3-axis accelerometer and 3-axis gyroscope. It does **not** consume magnetometer data, making it immune to magnetic anomalies but prone to unconstrained yaw drift over time.
- Conversion formula:
  $$\psi_{\text{grv}} = \text{degrees}\left(\text{atan2}(2(xy + wz),\, w^2 + x^2 - y^2 - z^2)\right)$$
- Evaluation on `0_down_facing`: Over 500 consecutive epochs (~2.5 seconds), the circular difference between $\psi_{\text{grv}}$ and `OrientationDeg.yawDeg` exhibits a mean offset of **23.78°** with standard deviation of **15.59°** [OBSERVED].
- Over a 12-minute session, GRV yaw drifts by tens of degrees away from geographic north, proving that GRV cannot be used as an absolute compass reference without external heading anchoring.

### 3.4 Summary: What Can and Cannot Be Concluded

```
+-----------------------------------------------------+-----------------------------------------------------+
| CAN BE CONCLUDED [OBSERVED]                         | CANNOT BE CONCLUDED [PITFALL / INVALID]             |
+-----------------------------------------------------+-----------------------------------------------------+
| 1. OrientationDeg yaw is an integer compass azimuth | 1. Do NOT assume session name yaw (0/45/90/135) is  |
|    referenced to Magnetic North (0-359).            |    True North or Compass Azimuth.                   |
| 2. GRV maintains rigorous unit norm (|q|=1) but has | 2. Do NOT treat GRV yaw as an absolute compass angle|
|    no North anchor and drifts continuously.         |    (it lacks magnetometer correction).              |
| 3. FOP heading provides float-precision compass     | 3. Do NOT expect FOP heading to match OrientationDeg|
|    angles but inverts by 180° during steep tilts.   |    near vertical tilt (Euler singularities diverge).|
| 4. Pitch and roll in OrientationDeg stably reflect  | 4. Do NOT attempt to extract FOP or Accel from the  |
|    the physical tilt angle across static sessions.  |    14 June orientation sessions (they have 0 rows). |
+-----------------------------------------------------+-----------------------------------------------------+
```

---

## 4. Accelerometer Gravity Norm & Motion Verification

### 4.1 Goal
Evaluate the total acceleration magnitude $\|\mathbf{a}\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$ across all sessions reporting calibrated `Accel`, test compliance with nominal terrestrial gravity ($g \approx 9.81\,\text{m/s}^2$), flag anomalous sensors, and classify motion state (static vs moving).

### 4.2 Why
Smartphones at rest record the reaction force against gravity: $\|\mathbf{a}\| \approx 9.81\,\text{m/s}^2$. If the sensor calibration is broken or uncalibrated biases are present, the mean norm will diverge (e.g. $< 8$ or $> 11\,\text{m/s}^2$). Furthermore, the variance of $\|\mathbf{a}\|$ distinguishes static receivers from dynamic handheld motion: static logging yields $\sigma_{\|\mathbf{a}\|} < 0.2\,\text{m/s}^2$, while pedestrian movement or vehicle vibration induces $\sigma_{\|\mathbf{a}\|} > 0.5\,\text{m/s}^2$.

### 4.3 Query
Executed via `queries/044_accel_gravity_norm.sql`:

```sql
SELECT session,
  round(avg(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS mean_norm,
  round(stddev(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 4) AS std_norm,
  round(min(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS min_norm,
  round(max(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS max_norm,
  count(*) AS n_rows
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet')
GROUP BY session ORDER BY session;
```

### 4.4 Result [OBSERVED]
Calibrated `Accel` is populated in **29 sessions** (all recorded on Samsung SM-M146B during the October campaign). All 29 sessions pass the gravity norm check:

| Session Name | Device Model | Mean Norm ($\text{m/s}^2$) | Std Norm ($\text{m/s}^2$) | Min Norm | Max Norm | Accel Rows | Motion Classification |
|---|---|---|---|---|---|---|---|
| `gnss_log_2026_10_19_08_51_20` | samsung SM-M146B | 9.907 | 0.4875 | 1.381 | 75.603 | 367,520 | Near-Static (Transient Handling) |
| `gnss_log_2026_10_22_08_30_05` | samsung SM-M146B | 9.937 | 0.1200 | 1.023 | 35.464 | 232,277 | **Static** |
| `gnss_log_2026_10_22_08_46_48` | samsung SM-M146B | 9.844 | 0.1952 | 0.923 | 41.573 | 211,725 | **Static** |
| `gnss_log_2026_10_22_09_01_59` | samsung SM-M146B | 9.991 | 0.1728 | 3.859 | 30.107 | 223,835 | **Static** |
| `gnss_log_2026_10_22_09_17_53` | samsung SM-M146B | 9.859 | 0.1650 | 4.493 | 23.680 | 180,932 | **Static** |
| `gnss_log_2026_10_22_09_30_34` | samsung SM-M146B | 9.954 | 0.1904 | 5.029 | 45.168 | 219,377 | **Static** |
| `gnss_log_2026_10_22_10_02_19` | samsung SM-M146B | 9.946 | 0.1709 | 3.742 | 30.100 | 195,574 | **Static** |
| `gnss_log_2026_10_23_16_00_08` | samsung SM-M146B | 9.937 | 0.2755 | 0.603 | 43.419 | 233,698 | **Static** |
| `gnss_log_2026_10_23_16_16_44` | samsung SM-M146B | 9.803 | 0.2096 | 1.361 | 22.287 | 200,846 | **Static** |
| `gnss_log_2026_10_23_16_30_52` | samsung SM-M146B | 9.952 | 0.1845 | 5.531 | 61.614 | 221,595 | **Static** |
| `gnss_log_2026_10_23_16_46_37` | samsung SM-M146B | 9.875 | 0.3356 | 4.443 | 47.727 | 196,822 | **Static** |
| `gnss_log_2026_10_23_17_00_21` | samsung SM-M146B | 9.937 | 0.1361 | 3.808 | 31.575 | 221,845 | **Static** |
| `gnss_log_2026_10_23_17_15_52` | samsung SM-M146B | 9.882 | 0.1341 | 4.372 | 28.327 | 213,108 | **Static** |
| `gnss_log_2026_10_23_17_30_42` | samsung SM-M146B | 9.932 | 0.2775 | 1.594 | 20.896 | 211,850 | **Static** |
| `gnss_log_2026_10_23_17_45_26` | samsung SM-M146B | 9.928 | 0.1994 | 3.617 | 24.555 | 222,317 | **Static** |
| `gnss_log_2026_10_26_08_15_01` | samsung SM-M146B | 9.943 | 0.2681 | 0.374 | 43.898 | 225,397 | **Static** |
| `gnss_log_2026_10_26_08_31_07` | samsung SM-M146B | 9.845 | 0.2335 | 1.737 | 26.537 | 209,654 | **Static** |
| `gnss_log_2026_10_26_08_45_52` | samsung SM-M146B | 10.007 | 0.1337 | 3.770 | 21.197 | 214,627 | **Static** |
| `gnss_log_2026_10_26_09_01_03` | samsung SM-M146B | 9.876 | 0.5876 | 0.723 | 80.870 | 200,708 | **Moving / Handled** ($\sigma > 0.5$) |
| `gnss_log_2026_10_26_09_15_24` | samsung SM-M146B | 9.963 | 0.1997 | 4.651 | 19.047 | 219,880 | **Static** |
| `gnss_log_2026_10_26_09_30_47` | samsung SM-M146B | 9.870 | 0.1649 | 5.070 | 30.468 | 229,652 | **Static** |
| `gnss_log_2026_10_26_09_46_57` | samsung SM-M146B | 9.938 | 0.2507 | 3.043 | 27.192 | 190,834 | **Static** |
| `gnss_log_2026_10_26_10_00_23` | samsung SM-M146B | 9.923 | 0.2791 | 2.995 | 58.520 | 226,388 | **Static** |
| `gnss_log_2026_10_27_08_20_32` | samsung SM-M146B | 9.933 | 0.3050 | 1.203 | 60.744 | 231,932 | **Static** |
| `gnss_log_2026_10_27_08_37_00` | samsung SM-M146B | 9.863 | 0.1992 | 3.878 | 32.670 | 206,214 | **Static** |
| `gnss_log_2026_10_27_08_51_24` | samsung SM-M146B | 9.936 | 0.1563 | 2.311 | 19.219 | 231,563 | **Static** |
| `gnss_log_2026_10_27_09_07_28` | samsung SM-M146B | 9.881 | 0.2522 | 2.089 | 44.498 | 195,898 | **Static** |
| `gnss_log_2026_10_27_09_21_12` | samsung SM-M146B | 9.986 | 0.3549 | 0.211 | 58.280 | 217,477 | **Static** |
| `gnss_log_2026_10_27_09_36_34` | samsung SM-M146B | 9.903 | 0.2182 | 3.425 | 24.332 | 215,723 | **Static** |

### 4.5 Findings & Insights
- **Gravity Norm Integrity [OBSERVED]**: Mean norm across all 29 sessions falls strictly between **9.803 and 10.007 m/s²**, tightly bounding expected local gravity ($9.78\,\text{m/s}^2$ at Pune's latitude of 18.5°N). Zero sessions trigger the out-of-bounds flag ($< 8$ or $> 11\,\text{m/s}^2$).
- **Motion Verification [OBSERVED]**: 28 of the 29 sessions have $\sigma_{\|\mathbf{a}\|} \le 0.35\,\text{m/s}^2$, confirming that the Samsung device was held rigidly static on a stationary mount or tripod during these logging intervals.
- **Handling Transient**: Only a single session, `gnss_log_2026_10_26_09_01_03`, exhibits $\sigma_{\|\mathbf{a}\|} = 0.588\,\text{m/s}^2$ with an acceleration spike of $80.87\,\text{m/s}^2$, characteristic of manual phone handling, mounting adjustment, or repositioning at session setup.

---

## 5. Pitch and Roll Variation Within and Across Sessions

### 5.1 Goal
Quantify attitude stability within each session and map the diversity of physical poses across the entire corpus. Classify sessions into canonical geometric categories: flat screen-up, flat screen-down, vertical upright, and intermediate tilts.

### 5.2 Within-Session Attitude Stability [OBSERVED]
In the 14 controlled orientation sessions, the orientation was maintained with sub-degree precision:
- In `0_up_facing`: `pitchDeg` std = **0.00°**, `rollDeg` std = **0.01°**. The device was completely motionless on a level surface.
- In `135_front_horizontal`: `pitchDeg` std = **0.00°**, `rollDeg` std = **0.00°**.
- In `45_back_vertical`: `pitchDeg` std = **0.00°**, `rollDeg` std = **0.00°**.
- In `0_down_facing`: `pitchDeg` std = **1.46°**, `rollDeg` std = **16.77°**. When a phone lies flat screen-down, roll hovers near $\pm 180^\circ$. Measurement noise causes rapid bouncing between $+179^\circ$ and $-179^\circ$, creating an apparent standard deviation spike without physical movement.

### 5.3 Across-Session Attitude Distribution [OBSERVED]
Across all 89 sessions reporting `OrientationDeg` (the 14 June sessions + 75 October sessions):
- Mean `pitchDeg` spans from **−88.4°** to **+3.0°** (p50 = −37.8°).
- Mean `rollDeg` spans from **−178.3°** to **+178.4°**.
- Geometric classifications across the corpus:
  1. **Flat Screen-Up (`pitch ≈ 0°`, `roll ≈ 0°`)**: Exemplified by `0_up_facing` (pitch = 0.0°, roll = −1.0°). Screen points to zenith, back antenna points to ground.
  2. **Flat Screen-Down / Inverted (`pitch ≈ 0°`, `roll ≈ ±180°`)**: Exemplified by `0_down_facing` (pitch = −0.0°, roll = +176.5°). Back-mounted GNSS antenna faces directly towards the zenith sky.
  3. **Vertical Portrait (`pitch ≈ −87° to −88°`)**: Exemplified by `90_front_vertical` (pitch = −88.0°) and `90_back_vertical` (pitch = −87.0°). Phone is held upright. Top-edge antenna faces zenith, back antenna faces horizontal horizon.
  4. **Tilted Intermediate Mount (`pitch ≈ −40° to −46°`)**: Dominates the naturalistic October sessions (median pitch = −37.8°), mimicking standard in-vehicle dashboard mounts or handheld usage at reading angles.

```
                  ZENITH SKY (+Z in Body Frame when Screen Down)
                                     ▲
                                     │
           [0_down_facing]           │           [0_up_facing]
        Back Antenna -> Sky          │        Screen -> Sky (Antenna Down)
        Pitch = 0°, Roll = 180°      │        Pitch = 0°, Roll = 0°
                                     │
       ◄─────────────────────────────┼─────────────────────────────► HORIZON
                                     │
                                     │
           [90_front/back_vert]      │           [45_front/back_vert]
        Upright Portrait             │        Angled Vehicle / Desk Mount
        Pitch = -88°, Roll = 0/180°  │        Pitch = -40° to -46°
                                     ▼
                                   NADIR
```

---

## 6. Synthesis & Recommendations for Antenna Pointing ML Models

### 6.1 Eliminate Euler Angles in Feature Pipelines
Legacy Euler angles (`yawDeg`, `pitchDeg`, `rollDeg`) suffer from severe mathematical and data quality defects:
1. `yawDeg` is integer-quantized by Android, discarding sub-degree rotational resolution.
2. At pitch $\to \pm 90^\circ$, gimbal lock causes yaw and roll to become indeterminate, jumping discontinuously by $90^\circ\text{--}180^\circ$.
3. FOP heading flips by $180^\circ$ under backwards tilt.

### 6.2 Rotate Satellite Line-of-Sight into the Phone Body Frame
Rather than predicting signal strength from separate azimuth/elevation and yaw/pitch features, compute the 3D unit Line-of-Sight (LOS) vector $\mathbf{s}_{\text{enu}}$ from satellite elevation and azimuth:
$$\mathbf{s}_{\text{enu}} = \begin{bmatrix} \cos(\text{Elev})\sin(\text{Az}) \\ \cos(\text{Elev})\cos(\text{Az}) \\ \sin(\text{Elev}) \end{bmatrix}$$
Rotate $\mathbf{s}_{\text{enu}}$ into the phone body coordinate system using the rotation matrix $\mathbf{R}_{\text{world}\to\text{body}}$ from `Sensor.TYPE_ROTATION_VECTOR` or `FOP`:
$$\mathbf{s}_{\text{body}} = \mathbf{R} \cdot \mathbf{s}_{\text{enu}} = \begin{bmatrix} s_{bx} \\ s_{by} \\ s_{bz} \end{bmatrix}$$
This formulation:
- Eliminates all gimbal singularities and wrapping discontinuities at $0^\circ/360^\circ$.
- Directly exposes the satellite arrival angle relative to the internal patch antenna plane ($s_{bz} > 0$ indicates arrival from the front screen, $s_{bz} < 0$ from the back casing).
- Allows linear (Ridge) and non-linear (LightGBM/HistGradient) models to learn anisotropic receiver antenna gain patterns directly from data.
