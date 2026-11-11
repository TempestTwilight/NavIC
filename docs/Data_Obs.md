# Android GNSS Logger Dataset: Column Reference

## 0. Background

**GNSS** (Global Navigation Satellite System) is the umbrella term for GPS (US), GLONASS (Russia), Galileo (EU), BeiDou (China), QZSS (Japan), SBAS (augmentation) and IRNSS/NavIC (India).

**How positioning works:** each satellite broadcasts its position and the time it sent the signal. The phone measures when the signal arrived, so `distance = (arrival time - transmit time) × speed of light`. With distances to 4+ satellites, the phone solves for (x, y, z, clock error). This is called **trilateration**.

**Common terms**

| Term | Meaning |
|---|---|
| ECEF | Earth-Centered, Earth-Fixed 3D Cartesian frame (origin at Earth's center) |
| Pseudorange | Measured distance to a satellite, "pseudo" because it includes the receiver clock error |
| Cn0 (C/N0) | Carrier-to-noise density ratio in dB-Hz: signal strength (higher is better; ~20 is weak, ~45+ is strong) |
| Svid | Satellite Vehicle ID, unique only *within* a constellation |
| Epoch | One timestamp at which a batch of measurements was taken |

**Constellation type codes (`ConstellationType`)**

| Code | Constellation |
|---|---|
| 1 | GPS |
| 2 | SBAS |
| 3 | GLONASS |
| 4 | QZSS |
| 5 | BeiDou |
| 6 | Galileo |
| 7 | IRNSS (NavIC) |

**Time columns used across tables**

| Column | What it is |
|---|---|
| `utcTimeMillis` / `UnixTimeMillis` | Wall-clock time in milliseconds since 1970-01-01 UTC. Used for joining tables and aligning with ground truth |
| `elapsedRealtimeNanos` | Nanoseconds since phone boot (monotonic, never jumps). Best for aligning **sensors** to each other because it isn't affected by clock changes |

---

## 1. `Accel`: Accelerometer

Measures the **acceleration of the phone along its own 3 axes**, including gravity (a phone lying flat reads about 9.81 on Z).

| Column | Description |
|---|---|
| `Accel` | Record-type label (constant, 1 unique value) |
| `utcTimeMillis` | UTC timestamp of the reading |
| `elapsedRealtimeNanos` | Time since boot, nanosecond precision |
| `AccelXMps2` | Acceleration along the phone's X axis (left to right across the screen), m/s² |
| `AccelYMps2` | Acceleration along Y (bottom to top of screen), m/s² |
| `AccelZMps2` | Acceleration along Z (out of the screen), m/s² |
| `CalibrationAccuracy` | Android sensor accuracy level (0 = unreliable ... 3 = high). Constant here |

**Why:** detects motion between GNSS fixes, step/turn/stop detection, and dead reckoning (estimating position when GPS is weak).
**How:** integrate acceleration to get velocity and position (drifts quickly), or use as ML features (e.g., mean/variance over a window for activity recognition). Rotate into the world frame using the orientation data below.

---

## 2. `Raw`: Raw GNSS Measurements

The most detailed table: **one row per satellite signal per epoch**. This is what you need for custom positioning algorithms (e.g., WLS, Kalman filter, RTK/PPP).

### 2.1 Receiver clock and time

| Column | Description |
|---|---|
| `Raw` | Record-type label |
| `utcTimeMillis` | UTC timestamp |
| `TimeNanos` | Receiver's internal hardware clock time at the measurement (ns) |
| `TimeOffsetNanos` | Per-measurement offset added to `TimeNanos` (ns). Usually 0 |
| `FullBiasNanos` | Difference between the receiver hardware clock and true GPS time (ns). Large negative number |
| `BiasNanos` | Sub-nanosecond correction to `FullBiasNanos` |
| `BiasUncertaintyNanos` | 1-sigma uncertainty of the clock bias |
| `DriftNanosPerSecond` | Rate at which the receiver clock drifts relative to GPS time |
| `DriftUncertaintyNanosPerSecond` | Uncertainty of the drift |
| `TimeUncertaintyNanos` | Uncertainty of `TimeNanos` |
| `LeapSecond` | Current GPS-UTC leap second count |
| `HardwareClockDiscontinuityCount` | Increments when the receiver clock jumps/resets. A change means the bias history is no longer continuous |
| `ChipsetElapsedRealtimeNanos` | Time-since-boot as reported by the GNSS chipset itself |

**How it's used:** GPS time of reception is
`t_rx = TimeNanos + TimeOffsetNanos - (FullBiasNanos + BiasNanos)`

### 2.2 Satellite signal measurement

| Column | Description |
|---|---|
| `ConstellationType` | Which constellation (see table above) |
| `Svid` | Satellite ID within the constellation |
| `CarrierFrequencyHz` | Signal frequency (e.g., 1575.42 MHz = GPS L1, 1176.45 MHz = L5). The ~15 unique values are different bands |
| `CodeType` | Signal code, e.g., `C` (civilian), `I`/`Q` (in-phase/quadrature channels), `P`. Tells which signal on the frequency |
| `State` | Bitmask of how far the receiver has locked on to the signal (code lock, bit sync, subframe sync, time-of-week decoded...). Used to filter out unreliable measurements |
| `ReceivedSvTimeNanos` | Satellite transmit time (ns) as decoded from the signal |
| `ReceivedSvTimeUncertaintyNanos` | Uncertainty of that transmit time |
| `Cn0DbHz` | Signal strength (dB-Hz) |
| `BasebandCn0DbHz` | Signal strength measured at the baseband stage, slightly different from `Cn0DbHz` as it's measured at a different point in the receiver chain |
| `MultipathIndicator` | Whether the signal is suspected to be reflected (multipath). Constant here, so not informative |
| `AgcDb` | Automatic Gain Control level. Reflects amplifier gain. Changes in it can indicate **interference or jamming** |
| `SnrInDb` | Signal-to-noise ratio (dB) |

**Pseudorange:** `ρ = (t_rx - t_tx) × c`, where `t_tx` comes from `ReceivedSvTimeNanos` and `c` is the speed of light.

### 2.3 Doppler (velocity information)

| Column | Description |
|---|---|
| `PseudorangeRateMetersPerSecond` | Rate of change of distance to the satellite (m/s), measured via the Doppler shift. Used to estimate **receiver velocity** |
| `PseudorangeRateUncertaintyMetersPerSecond` | Uncertainty of the above |

### 2.4 Carrier phase (high-precision tracking)

| Column | Description |
|---|---|
| `AccumulatedDeltaRangeMeters` | Accumulated change in carrier phase, converted to meters. Very precise (cm level) but ambiguous by an unknown constant |
| `AccumulatedDeltaRangeUncertaintyMeters` | Uncertainty |
| `AccumulatedDeltaRangeState` | Bitmask: 1 = valid, 2 = reset, 4 = cycle slip (tracking glitch). **Only use rows where valid** |
| `CarrierPhase`, `CarrierCycles`, `CarrierPhaseUncertainty` | Older carrier-phase fields. **100% null**, so drop them |

**Why:** carrier-phase measurements enable centimeter/decimeter-level positioning (RTK/PPP) and smoothing of noisy pseudoranges (Hatch filter).

### 2.5 Satellite position, velocity, clock (chipset-provided)

| Column | Description |
|---|---|
| `SvPositionEcefXMeters`, `...YMeters`, `...ZMeters` | Satellite position in ECEF (m) at transmit time |
| `SvVelocityEcefXMetersPerSecond`, `...Y...`, `...Z...` | Satellite velocity in ECEF (m/s) |
| `SvClockBiasMeters` | Satellite clock error expressed as a distance (m). Must be corrected for |
| `SvClockDriftMetersPerSecond` | Rate of change of satellite clock error |
| `IsFullTracking` | Whether the receiver was in full tracking mode (not duty-cycled to save power) |

**Why:** you need satellite positions to do trilateration. About 57% null here, so for those rows you must compute satellite positions yourself from ephemeris (navigation) files.

### 2.6 Atmospheric and hardware bias corrections

| Column | Description |
|---|---|
| `KlobucharAlpha0-3`, `KlobucharBeta0-3` | Coefficients of the **Klobuchar ionospheric model** broadcast by GPS. The ionosphere delays signals, and these let you estimate and subtract that delay (~50% of the error) |
| `FullInterSignalBiasNanos` | Hardware delay difference between signals/frequencies in the receiver |
| `FullInterSignalBiasUncertaintyNanos` | Uncertainty of the above |
| `SatelliteInterSignalBiasNanos` | Satellite-side delay between signals (e.g., L1 vs L5) |
| `SatelliteInterSignalBiasUncertaintyNanos` | Uncertainty of the above |

---

## 3. GNSS clock summary table (the 2nd block in your CSV)

This block has `Agc`, `TimeNanos`, `FullBiasNanos`, `BiasNanos`, `LeapSecond`, `ConstellationType`, etc., with 1,555,526 rows, the same count as the nulls in `Raw`'s `TimeUncertaintyNanos`. It looks like a **subset/earlier version of `Raw`** (likely a different log source or file) that contains only clock and AGC fields. Its columns mean the same as in section 2. `TimeUncertaintyNanos`, `DriftUncertaintyNanosPerSecond` and `DriftNanosPerSecond` are 100% null here, so drop them.

---

## 4. `Fix`: Position Fixes from the Phone

The **location estimates the phone itself computed** (what Google Maps would show). Useful as a baseline and for comparison against your own algorithm.

| Column | Description |
|---|---|
| `Fix` | Record-type label |
| `Provider` | Source of the fix: `GPS` (GNSS chip only), `FLP` (Fused Location Provider: GNSS + WiFi + sensors), `NLP` (Network Location Provider: WiFi/cell). 3 unique values |
| `LatitudeDegrees` | Latitude (°) |
| `LongitudeDegrees` | Longitude (°) |
| `AltitudeMeters` | Altitude above the WGS84 ellipsoid (m) |
| `SpeedMps` | Ground speed (m/s) |
| `BearingDegrees` | Direction of travel, 0-360° clockwise from true north |
| `AccuracyMeters` | Estimated horizontal accuracy (radius, ~68% confidence) |
| `VerticalAccuracyMeters` | Estimated vertical accuracy |
| `SpeedAccuracyMps` | Estimated speed error |
| `BearingAccuracyDegrees` | Estimated bearing error |
| `UnixTimeMillis` | UTC timestamp of the fix |
| `elapsedRealtimeNanos` | Time since boot |
| `MockLocation` | Whether the location was faked by an app (constant here) |
| `ExtraKeyValuePairs` | Extra provider metadata as key=value text (about 54% null) |
| `NumberOfUsedSignals`, `SolutionType`, `VerticalSpeedAccuracyMps` | 100% null, so drop them |

**Note:** `BearingDegrees` is mostly null (72%), probably because the phone is stationary and has no heading when not moving.

---

## 5. `Status`: Satellite Visibility and Quality

**One row per visible satellite signal per epoch**, derived from Android's `GnssStatus`.

| Column | Description |
|---|---|
| `Status` | Record-type label |
| `UnixTimeMillis` | UTC timestamp (about 0.12% null) |
| `SignalCount` | Number of signals in this epoch's batch |
| `SignalIndex` | Index of this signal within the batch |
| `ConstellationType` | Constellation (see table above) |
| `Svid` | Satellite ID |
| `CarrierFrequencyHz` | Signal frequency |
| `Cn0DbHz` | Signal strength |
| `BasebandCn0DbHz` | Baseband signal strength |
| `AzimuthDegrees` | Satellite compass direction from the receiver (0-360°) |
| `ElevationDegrees` | Satellite angle above the horizon (0-90°) |
| `UsedInFix` | Whether the phone used this satellite for its position fix (boolean) |
| `HasAlmanacData` | Whether the phone has coarse orbit data for this satellite |
| `HasEphemerisData` | Whether the phone has precise orbit data for this satellite |

**Why:**
- Azimuth and elevation give **sky geometry** (used to compute DOP, i.e. how good the satellite spread is).
- Low-elevation signals suffer more from multipath and atmospheric error, so they're often filtered or down-weighted.
- Great for **NLOS (non-line-of-sight) detection** in urban canyons: low Cn0 plus a low elevation angle suggests a reflected signal.

---

## 6. `FOP`: Fused Orientation / Heading

> I'm fairly (not 100%) sure this is the Android fused orientation/heading sensor output (similar to Android 14's `TYPE_HEADING`).

| Column | Description |
|---|---|
| `FOP` | Record-type label |
| `utcTimeMillis`, `elapsedRealtimeNanos` | Timestamps |
| `x`, `y`, `z`, `w` | Orientation as a **quaternion** |
| `headingDegrees` | Compass heading (°) relative to true north |
| `headingErrorDegrees` | Estimated heading error (°) |
| `conservativeHeadingErrorDegrees` | A more cautious (larger) error bound for the heading |

**Why:** gives the direction the phone is facing. Fuse it with GNSS to improve heading and with accelerometer data for dead reckoning.

---

## 7. `GameRotationVector`: Orientation Without Magnetometer

Android's `TYPE_GAME_ROTATION_VECTOR`: orientation from **accelerometer and gyroscope only** (no magnetometer), so it is smooth and not disturbed by magnets or metal but **drifts in yaw over time** and has no absolute north reference.

| Column | Description |
|---|---|
| `GameRotationVector` | Record-type label |
| `utcTimeMillis`, `elapsedRealtimeNanos` | Timestamps |
| `x`, `y`, `z`, `w` | Orientation quaternion (unit quaternion: x² + y² + z² + w² = 1) |

**Why quaternions?** They represent 3D rotation without gimbal lock and are easy to interpolate and compose. You can convert them to a rotation matrix to transform phone-frame acceleration into the world frame.

---

## 8. `OrientationDeg`: Euler Angles

Human-readable orientation derived from the rotation sensors.

| Column | Description |
|---|---|
| `OrientationDeg` | Record-type label |
| `utcTimeMillis`, `elapsedRealtimeNanos` | Timestamps |
| `yawDeg` | Rotation around the vertical axis (azimuth / compass heading) |
| `pitchDeg` | Rotation around the lateral axis (tilting forward/back), range about -180 to 180 |
| `rollDeg` | Rotation around the longitudinal axis (tilting sideways) |
| `CalibrationAccuracy` | Sensor accuracy level (constant here) |

**Why:** easy to interpret and plot, and useful as features. Euler angles suffer from gimbal lock, so use the quaternion for computation.

---

## 9. Data-Cleaning Notes

| Issue | Columns | Action |
|---|---|---|
| 100% null | `CarrierPhase`, `CarrierCycles`, `CarrierPhaseUncertainty`, `NumberOfUsedSignals`, `SolutionType`, `VerticalSpeedAccuracyMps`, and (in the clock-summary table) `TimeUncertaintyNanos`, `DriftUncertaintyNanosPerSecond`, `DriftNanosPerSecond` | Drop |
| Constant (nunique = 1) | `Accel`, `Raw`, `Fix`, `FOP`, `CalibrationAccuracy`, `MockLocation`, `TimeOffsetNanos`, `MultipathIndicator`, `Status` | Drop (no information) |
| Partially null, structural | `Sv*` columns (~57%), `Klobuchar*` (~41%), `Bearing*` (~72%) | Check *why* before imputing (often missing by design, not random) |
| Different sampling rates | Sensors (~100 Hz) vs. GNSS (~1 Hz) | Align via timestamps (`merge_asof` in pandas) |