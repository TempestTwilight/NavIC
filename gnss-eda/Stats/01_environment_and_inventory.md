# 01 — Environment, File Inventory, and Parsing Approach

*All statements in this report are labelled: **[OBSERVED]** (seen in data, query/step ID in `execution_log.md`), **[REFERENCE]** (claimed by `docs/Data_Obs.md`), **[BACKGROUND]** (general GNSS/Android knowledge), **[ASSUMPTION]**.*

---

## 1. Machine and tool limits [OBSERVED, STEP-000]

| Item | Value |
|---|---|
| CPU count | 4 logical cores |
| Total RAM | 5.7 GiB (only ~1.0 GiB *available* at start — the machine is already heavily loaded; swap in use) |
| Disk | 228 GB volume, **38 GB free** at start of EDA |
| DuckDB | **2.0.0**, installed during this EDA (`pip install --user --break-system-packages duckdb`) |
| psutil | available (used for RSS checks) |

**Resource policy (followed in every DuckDB connection):**

```sql
SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;
SET temp_directory='docs/EDA/scratch';
```

Because the OS itself had only ~1.0 GB free, DuckDB's 1 GB limit plus the Python process must coexist with the rest of the system; all heavy work is therefore done by *streaming* DuckDB scans with results reduced to small aggregates (≤50 rows), never by materialising data in Python.

## 2. What is in `data/Raw_Logs/` [OBSERVED, STEP-001, STEP-010]

**262 files, 10.0 GB total.** Three groups:

### 2.1 Group 1 — the main corpus: 95 Android GNSS Logger `.txt` files (9.47 GB)

| Sub-group | Count | Naming | Meaning |
|---|---|---|---|
| Orientation-named | 14 | `NN_front/back_horizontal/vertical.txt`, `0/90_up_facing.txt` | A designed grid of phone orientations (0°/45°/90°/135° × front/back × horizontal/vertical, plus 0°/90° up-facing) — **[ASSUMPTION about design intent, from names only]** |
| Timestamp-named | 81 | `gnss_log_2026_10_XX_HH_MM_SS.txt` | Sessions named by UTC start time (matches the version header inside) |

All 95 files are **Android GNSS Logger v3.1.1.3** files [OBSERVED]. The header line inside each file identifies the device:

| Manufacturer / Model | Platform | GNSS chipset string | Files |
|---|---|---|---|
| OnePlus CPH2401 | Android 14 | MTK `MNL_VER_24081501ALPS05_6.7o_00` | 36 |
| samsung SM-M146B | Android 13 | S.LSI `K056` | 29 |
| OnePlus CPH2487 | Android 16 | Qualcomm `MPSS.DE.3.0.5-01458.1-WAIPIO` (two fw builds: …310 ×17, …312 ×5) | 22 |
| OPPO CPH2325 | Android 13 | MTK (one `…20040301` build ×4, one `default` ×2) | 6 |
| OnePlus CPH2401 | Android 14 | MTK `default` | 2 |

So this is a **4-device, multi-chipset corpus** (MediaTek ×44, Samsung Exynos ×29, Qualcomm ×22). Model marketing names were **not** taken from the files themselves; the `Data Collection` subfolders name them *Galaxy M14 5G*, *Oppo A55*, *OnePlus 11R*, *OnePlus Nord 2T 5G* [OBSERVED] — the mapping model-code↔name is plausible but **[ASSUMPTION]** (verified by timestamp matching only for a handful of files; see `06_orientation_and_sessions.md`).

### 2.2 Group 2 — subfolder collections `Data Collection -20261101T065940Z-1-001…005` (~520 MB)

Organised `device / day / orientation /` [OBSERVED]. Contains *derived conversions* of the same sessions, not new raw measurements:

| Ext | Files | What it is | Evidence |
|---|---|---|---|
| `.26o` | 80, 279 MB | RINEX 2.01-style observation files converted *from* GnssLogger (header: `GnssLogger samsung 13 … PGM/RUN BY/DATE`) [OBSERVED] | conversion of group 1 sessions |
| `.nmea` | 78, 210 MB | NMEA sentences, each line suffixed with `,utcTimeMillis` [OBSERVED] | same |
| `.26g` / `.26n` | 1 + 1 | GLONASS / GPS **broadcast ephemeris** files (header `CCRINEXG V1.4 UX CDDIS`, `IGS BROADCAST EPHEMERIS FILE`) [OBSERVED] | external reference data downloaded from CDDIS, *not* phone data |
| `.csv` | 1, 30 MB | a CSV export of the `Raw` record block (header identical to the Raw header) [OBSERVED] | processed export of a session already present as `.txt` |
| `.sh` | 1 | the script that moved all `.txt` files to the top level (`find … -name "*.txt" -exec mv`) [OBSERVED] | explains why `.txt` sits flat at top level |

### 2.3 Group 3 — analysis outputs for one session (`gnss_log_2026_10_22_08_30_05.*`) [OBSERVED]

`.derived` (7.9 MB), `-Adr.kml`, `-rawPr.kml`, `-rawKfPr.kml`, `-Adr.nmea`, `-rawPr.nmea`, `-rawKfPr.nmea`, `-param.mat`. These are outputs of the Google **GNSS Measurement and Simulation/Analysis tools** (KML truth tracks, KF pseudorange analysis). **Per the task definition, processed artefacts are out of scope**: they are inventoried here but excluded from column-level EDA.

## 3. Format of the main corpus `.txt` files [OBSERVED, STEP-002/005]

Each file is plain text:

1. `#` comment block: version line, then one **header line per record type** (comma-separated column names).
2. Then data lines, each starting with the **record-type label as the first CSV field**, e.g. `Raw,1780575270999,849959000000,18,…`.

All 95 files declare **exactly the same 17 record types with identical column orders** [OBSERVED — script `head_inv.py`, 1 distinct header signature]. This makes a uniform parser safe.

| # | Record type | Declared columns (after the label) | One row represents |
|---|---|---|---|
| 1 | `Raw` | 53 columns | one satellite signal measurement in one epoch |
| 2 | `UncalAccel` | 9 | uncalibrated accelerometer triad + bias est. |
| 3 | `Accel` | 6 | calibrated accelerometer triad |
| 4 | `UncalGyro` | 9 | uncalibrated gyroscope triad + drift est. |
| 5 | `Gyro` | 6 | calibrated gyroscope triad |
| 6 | `UncalMag` | 9 | uncalibrated magnetometer triad + bias est. |
| 7 | `Mag` | 6 | calibrated magnetometer triad |
| 8 | `Pressure` | 4 | barometer reading (hPa) |
| 9 | `OrientationDeg` | 6 | Euler angles `yawDeg,rollDeg,pitchDeg` (note order!) + accuracy |
| 10 | `Fix` | 17 | one position fix from a location provider |
| 11 | `Nav` | 6 declared, **variable** | one decoded navigation-message word (byte array) |
| 12 | `Status` | 13 | per-signal status snapshot (azimuth/elevation/C/N0/used-in-fix) |
| 13 | `Agc` | 13 | one clock+AGC snapshot (per constellation/carrier) |
| 14 | `GameRotationVector` | 6 | orientation quaternion (accel+gyro only) |
| 15 | `ExerciseTotalDist` | 8 | cumulative exercise distance |
| 16 | `ExerciseDeltaDist` | 10 | incremental exercise distance |
| 17 | `FOP` | 9 | fused orientation pose quaternion + heading |

**Format traps found before parsing anything [OBSERVED]:**

- `Nav` lines have a comma-separated **byte array** of variable length after 5 declared columns → a naive `read_csv` with a fixed schema would either error or misassign columns. Strategy: treat Nav's 6th field onward as *data text* (see §4).
- `Status` lines begin `Status,,87,0,…` in the inspected file → the declared first column `UnixTimeMillis` is **empty on many rows**; column profiling must expect nulls.
- `Fix`'s last column `ExtraKeyValuePairs` uses `|` between key=value pairs (e.g. `maxCn0=41|meanCn0=28|satellites=37`) — no embedded commas observed, so comma-splitting is safe **[OBSERVED in samples; full-file check pending in Phase C]**.
- `OrientationDeg` header order is `yawDeg, rollDeg, pitchDeg` — **different from the order in `docs/Data_Obs.md`** (which lists yaw, pitch, roll) and from the sample value `241,97,1`, where 97° is impossible as a roll for a phone held flat-ish. Flagged as **Q-006**.
- Record-type *presence* varies per file/device (e.g. `0_down_facing.txt` has no `Accel`/`Mag`/`Agc`/`Pressure`/`FOP` rows at all) [OBSERVED, STEP-005] → every per-type profile must report "number of files containing this type".

## 4. Parsing approach and why [OBSERVED, STEP-011…]

**Chosen strategy — two-stage lazy pipeline, DuckDB only:**

- **Stage 1 ("lines cache", per file):** one `read_csv` pass per file over the raw text into a single `VARCHAR` column; derive `label = split_part(line, ',', 1)`; write `COPY (… ) TO scratch/lines/<file>.parquet` with columns `(session, label, line)`. This is a *raw cache* (documented, not processed data): it preserves every byte of every data line, drops only `#` comment lines.
- **Stage 2 ("typed caches", per record type):** for each of the 16 fixed-arity record types, one scan over `scratch/lines/*.parquet` filtered by label, each field extracted by position with `split_part` (or list indexing) and cast to the declared type, `COPY` to `scratch/typed/<type>.parquet`. `Nav` gets a special cache: 5 declared fields + the raw remainder as one text column + a field count.
- All analysis then runs on **views** over these typed caches, with aggregate-only results.

**Why this approach (plain language):** the files are 9.5 GB of mixed-schema text. Reading them once into a byte-preserving cache means (a) we never re-parse the slow text format again, (b) each record type can be scanned independently (DuckDB reads only the needed parquet), (c) memory stays bounded because every step streams. Keeping the full `line` string in Stage 1 is the safety net: if a field mapping is ever wrong, we can re-derive it without touching the raw read-only files.

**What could go wrong (honest list):**
1. A line with **fewer fields than declared** (e.g. a truncated trailing line) → `split_part` returns `''`, which must be treated as NULL, not as 0. Casting `''` to a number fails, so casts must handle empty strings explicitly.
2. A line with **more fields than declared** (Nav byte arrays, a stray comma in a future field) → extra fields silently land in the last column. Guard: assert arity per label (`array_length(str_split(line,',')…)`) and record deviations in the quality report.
3. Scientific notation or locale issues (none observed in samples).
4. **DuckDB CSV-reader bugs found while building Stage 1 [OBSERVED, STEP-006]:** (a) with `auto_detect=false` and an explicit `columns={'line':'VARCHAR'}`, DuckDB 2.0 silently **truncates each line at the first comma** even with `strict_mode=false` — every data line lost all fields after the label. Fix: use a delimiter that never occurs in the files, `delim=E'\x01'`, so each full line is one field. (b) With the sniffer enabled (`auto_detect` default), dialect detection **fails** on one file (`gnss_log_2026_10_27_09_36_34.txt`, "Expected Number of Columns: 1 Found: 2"), so the sniffer is never used. Both fixes are in `scripts/stage1_lines.py`.
5. **Killing a background `COPY` mid-write leaves a truncated parquet** ("No magic bytes found at end of file", observed after a timeout killed Stage 2). Rule: never terminate a job during `COPY`; after every build, verify with a full `count(*)` and cross-check against the independent awk survey before trusting the cache.
6. Duplicate `utcTimeMillis` collisions across devices are *expected* (different phones can share wall-clock); session id is therefore part of every key.
7. The two-stage caches use disk; total cache size is reported in the log (lines cache 1.7 GB + typed caches 2.8 GB ≈ 4.5 GB, within the 38 GB free) [OBSERVED, STEP-007/009].

## 5. Memory safety checklist for every step

- Every query ends in `GROUP BY`/aggregate or `LIMIT ≤50`.
- `approx_count_distinct` / `approx_quantile` on full columns; exact quantiles only on small aggregates.
- RSS logged before/after heavy steps (psutil); abort-and-shrink if RSS > 1.5 GB.
- No `SELECT *` on full data; no `.df()`/`.fetchall()` on anything unbounded; per-session processing in loops for anything that could blow up.

---

**Peak RSS observed so far:** 206 MB across profiling [STEP-011]; ~640 MB peak during per-session gap sorting (Phase B, UncalGyro/GameRotationVector) — both well under the 1.5 GB abort threshold. Final value also in `09_summary_and_recommendations.md`.
