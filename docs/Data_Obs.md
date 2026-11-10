# Combined Data Observation
---

## 1. Raw.csv — Column Overview

The `Raw.csv` dataset contains raw GNSS measurement and navigation data collected from the device. Its columns cover satellite observations, timing information, signal quality, carrier measurements, receiver clock parameters, and satellite positioning data.

### Dataset Columns

#### 1. Time & Clock Measurements

Receiver timing, clock offsets, uncertainty and synchronization.

```
utcTimeMillis
TimeNanos
LeapSecond
TimeUncertaintyNanos
FullBiasNanos
BiasNanos
BiasUncertaintyNanos
DriftNanosPerSecond
DriftUncertaintyNanosPerSecond
HardwareClockDiscontinuityCount
TimeOffsetNanos
ChipsetElapsedRealtimeNanos
```

#### 2. Satellite Identification & Tracking

Satellite identifiers, constellation information and tracking status.

```
Svid
ConstellationType
State
IsFullTracking
CodeType
Raw
```

#### 3. Signal Strength & Quality

Measurements describing received signal strength and quality.

```
Cn0DbHz
BasebandCn0DbHz
SnrInDb
AgcDb
MultipathIndicator
```

#### 4. Pseudorange & Signal Propagation

Satellite signal reception timing and propagation measurements.

```
ReceivedSvTimeNanos
ReceivedSvTimeUncertaintyNanos
PseudorangeRateMetersPerSecond
PseudorangeRateUncertaintyMetersPerSecond
```

#### 5. Carrier Phase & Frequency

Carrier-wave measurements used for precise GNSS positioning.

```
CarrierFrequencyHz
CarrierCycles
CarrierPhase
CarrierPhaseUncertainty
AccumulatedDeltaRangeState
AccumulatedDeltaRangeMeters
AccumulatedDeltaRangeUncertaintyMeters
```

#### 6. Inter-Signal Bias Measurements

Bias corrections associated with satellite and signal timing.

```
FullInterSignalBiasNanos
FullInterSignalBiasUncertaintyNanos
SatelliteInterSignalBiasNanos
SatelliteInterSignalBiasUncertaintyNanos
```

#### 7. Satellite Position & Velocity (ECEF)

Satellite coordinates and velocity components in Earth-Centered, Earth-Fixed reference frame.

```
SvPositionEcefXMeters
SvPositionEcefYMeters
SvPositionEcefZMeters
SvVelocityEcefXMetersPerSecond
SvVelocityEcefYMetersPerSecond
SvVelocityEcefZMetersPerSecond
```

#### 8. Satellite Clock Corrections

```
SvClockBiasMeters
SvClockDriftMetersPerSecond
```

#### 9. Ionospheric Correction Parameters (Klobuchar Model)

Parameters used to estimate ionospheric signal propagation delays.

```
KlobucharAlpha0
KlobucharAlpha1
KlobucharAlpha2
KlobucharAlpha3
KlobucharBeta0
KlobucharBeta1
KlobucharBeta2
KlobucharBeta3
```

Total: 52 columns, including the `Raw` field.