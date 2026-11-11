# 00 — Index and One-Paragraph Summary

**Dataset:** 95 raw Android GNSS Logger v3.1.1.3 files (9.47 GB) from 4 phones (OnePlus CPH2401, samsung SM-M146B, OnePlus CPH2487, OPPO CPH2325), collected in **two campaigns**: 14 orientation-named files recorded ≈ 2026-06-04 (OnePlus CPH2401, MediaTek) and 81 `gnss_log_*` files recorded 2026-10-19 → 2026-10-27 (all four devices) [OBSERVED]. Each file a "session" that mixes up to 17 record types: raw GNSS measurements (`Raw`), satellite status (`Status`), position fixes (`Fix`), accelerometer/gyro/magnetometer (calibrated + uncalibrated), orientation (`OrientationDeg`, `GameRotationVector`, `FOP`), barometer (`Pressure`, empty), exercise distances (empty), navigation-message bytes (`Nav`), and a clock/AGC block (`Agc`). The EDA documents **every column of every present record type** (157 columns, 100% coverage), data quality, cross-table time alignment, C/N0 signal behaviour, phone-orientation conventions, verifies the reference doc claim-by-claim (37 confirmed, 4 contradicted), and collects 21 answered questions for the C/N0-forecasting project.

**Status:** EDA complete (2026-11-11).

| File | Content |
|---|---|
| [01_environment_and_inventory.md](01_environment_and_inventory.md) | machine limits, file inventory, format, parsing approach |
| [02_table_Raw.md](02_table_Raw.md) | `Raw` — per-signal GNSS measurement record type |
| [02_table_Status.md](02_table_Status.md) | `Status` — per-signal satellite status |
| [02_table_Fix.md](02_table_Fix.md) | `Fix` — phone position fixes |
| [02_table_Agc.md](02_table_Agc.md) | `Agc` — clock + AGC block |
| [02_table_OrientationDeg.md](02_table_OrientationDeg.md) | Euler angles |
| [02_table_GameRotationVector.md](02_table_GameRotationVector.md) | quaternion (accel+gyro) |
| [02_table_FOP.md](02_table_FOP.md) | fused orientation + heading |
| [02_table_Accel.md](02_table_Accel.md) | calibrated accelerometer |
| [02_table_UncalAccel.md](02_table_UncalAccel.md) | uncalibrated accelerometer |
| [02_table_UncalGyro.md](02_table_UncalGyro.md) | uncalibrated gyroscope |
| [02_table_UncalMag.md](02_table_UncalMag.md) | uncalibrated magnetometer |
| [02_table_Nav.md](02_table_Nav.md) | navigation-message byte records |
| [03_data_quality.md](03_data_quality.md) | nulls, constants, duplicates, outliers, timing |
| [04_cross_table_and_time_alignment.md](04_cross_table_and_time_alignment.md) | join keys, sampling rates, alignment strategy |
| [05_signal_behavior.md](05_signal_behavior.md) | C/N0 behaviour, persistence, survivorship (Phase F) |
| [06_orientation_and_sessions.md](06_orientation_and_sessions.md) | sessions, orientation grid, angle conventions |
| [07_reference_doc_verification.md](07_reference_doc_verification.md) | every `Data_Obs.md` claim: CONFIRMED/CONTRADICTED/UNVERIFIABLE |
| [08_open_questions_and_experiments.md](08_open_questions_and_experiments.md) | verification queue Q-001… |
| [09_summary_and_recommendations.md](09_summary_and_recommendations.md) | findings, ML risks, preprocessing decisions, top-10 |
| [execution_log.md](execution_log.md) | append-only log of every action |
| [column_coverage.csv](column_coverage.csv) | one row per table.column, documentation status |
| `queries/NNN_*.sql` | every SQL query used |
| `scratch/` | raw caches (parquet) built from the read-only raw files; sizes reported in the log |

## Record types present vs declared

Declared by the logger header: 17. Actually present in ≥1 file: **12** (`Raw`, `Status`, `Fix`, `Agc`, `Nav`, `OrientationDeg`, `GameRotationVector`, `FOP`, `Accel`, `UncalAccel`, `UncalGyro`, `UncalMag`). Never present anywhere (0 rows in all 95 files): `Mag`, `Gyro`, `Pressure`, `ExerciseTotalDist`, `ExerciseDeltaDist` [OBSERVED, STEP-004].
