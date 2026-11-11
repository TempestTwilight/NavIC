"""Stage 2: build one typed parquet cache per record type from the stage-1
lines caches. Each field is extracted positionally with split_part and cast
with try_cast (empty string -> NULL via nullif). The `line` column is kept in
every typed cache as a safety net for spot-checks.

This is a RAW CACHE (format conversion only, no values altered), not
processed data. Sizes are reported in execution_log.md.
"""

import duckdb
import os
import psutil
import time

OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
LINES = os.path.join(OUT, "scratch", "lines")
TYPED = os.path.join(OUT, "scratch", "typed")
os.makedirs(TYPED, exist_ok=True)
GLOB = os.path.join(LINES, "*.parquet")

I, D, S = "BIGINT", "DOUBLE", "VARCHAR"

# field maps: (label, [(column, type), ...]) -- position 1 = right after the label
MAPS = {
    "Raw": [
        ("utcTimeMillis", I),
        ("TimeNanos", I),
        ("LeapSecond", I),
        ("TimeUncertaintyNanos", D),
        ("FullBiasNanos", I),
        ("BiasNanos", D),
        ("BiasUncertaintyNanos", D),
        ("DriftNanosPerSecond", D),
        ("DriftUncertaintyNanosPerSecond", D),
        ("HardwareClockDiscontinuityCount", I),
        ("Svid", I),
        ("TimeOffsetNanos", D),
        ("State", I),
        ("ReceivedSvTimeNanos", I),
        ("ReceivedSvTimeUncertaintyNanos", I),
        ("Cn0DbHz", D),
        ("PseudorangeRateMetersPerSecond", D),
        ("PseudorangeRateUncertaintyMetersPerSecond", D),
        ("AccumulatedDeltaRangeState", I),
        ("AccumulatedDeltaRangeMeters", D),
        ("AccumulatedDeltaRangeUncertaintyMeters", D),
        ("CarrierFrequencyHz", I),
        ("CarrierCycles", D),
        ("CarrierPhase", D),
        ("CarrierPhaseUncertainty", D),
        ("MultipathIndicator", I),
        ("SnrInDb", D),
        ("ConstellationType", I),
        ("AgcDb", D),
        ("BasebandCn0DbHz", D),
        ("FullInterSignalBiasNanos", D),
        ("FullInterSignalBiasUncertaintyNanos", D),
        ("SatelliteInterSignalBiasNanos", D),
        ("SatelliteInterSignalBiasUncertaintyNanos", D),
        ("CodeType", S),
        ("ChipsetElapsedRealtimeNanos", I),
        ("IsFullTracking", I),
        ("SvPositionEcefXMeters", D),
        ("SvPositionEcefYMeters", D),
        ("SvPositionEcefZMeters", D),
        ("SvVelocityEcefXMetersPerSecond", D),
        ("SvVelocityEcefYMetersPerSecond", D),
        ("SvVelocityEcefZMetersPerSecond", D),
        ("SvClockBiasMeters", D),
        ("SvClockDriftMetersPerSecond", D),
        ("KlobucharAlpha0", D),
        ("KlobucharAlpha1", D),
        ("KlobucharAlpha2", D),
        ("KlobucharAlpha3", D),
        ("KlobucharBeta0", D),
        ("KlobucharBeta1", D),
        ("KlobucharBeta2", D),
        ("KlobucharBeta3", D),
    ],
    "Status": [
        ("UnixTimeMillis", I),
        ("SignalCount", I),
        ("SignalIndex", I),
        ("ConstellationType", I),
        ("Svid", I),
        ("CarrierFrequencyHz", I),
        ("Cn0DbHz", D),
        ("AzimuthDegrees", D),
        ("ElevationDegrees", D),
        ("UsedInFix", I),
        ("HasAlmanacData", I),
        ("HasEphemerisData", I),
        ("BasebandCn0DbHz", D),
    ],
    "Fix": [
        ("Provider", S),
        ("LatitudeDegrees", D),
        ("LongitudeDegrees", D),
        ("AltitudeMeters", D),
        ("SpeedMps", D),
        ("AccuracyMeters", D),
        ("BearingDegrees", D),
        ("UnixTimeMillis", I),
        ("SpeedAccuracyMps", D),
        ("BearingAccuracyDegrees", D),
        ("elapsedRealtimeNanos", I),
        ("VerticalAccuracyMeters", D),
        ("MockLocation", I),
        ("NumberOfUsedSignals", I),
        ("VerticalSpeedAccuracyMps", D),
        ("SolutionType", S),
        ("ExtraKeyValuePairs", S),
    ],
    "Accel": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("AccelXMps2", D),
        ("AccelYMps2", D),
        ("AccelZMps2", D),
        ("CalibrationAccuracy", I),
    ],
    "UncalAccel": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("UncalAccelXMps2", D),
        ("UncalAccelYMps2", D),
        ("UncalAccelZMps2", D),
        ("BiasXMps2", D),
        ("BiasYMps2", D),
        ("BiasZMps2", D),
        ("CalibrationAccuracy", I),
    ],
    "UncalGyro": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("UncalGyroXRadPerSec", D),
        ("UncalGyroYRadPerSec", D),
        ("UncalGyroZRadPerSec", D),
        ("DriftXRadPerSec", D),
        ("DriftYRadPerSec", D),
        ("DriftZRadPerSec", D),
        ("CalibrationAccuracy", I),
    ],
    "UncalMag": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("UncalMagXMicroT", D),
        ("UncalMagYMicroT", D),
        ("UncalMagZMicroT", D),
        ("BiasXMicroT", D),
        ("BiasYMicroT", D),
        ("BiasZMicroT", D),
        ("CalibrationAccuracy", I),
    ],
    "OrientationDeg": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("yawDeg", D),
        ("rollDeg", D),
        ("pitchDeg", D),
        ("CalibrationAccuracy", I),
    ],
    "FOP": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("x", D),
        ("y", D),
        ("z", D),
        ("w", D),
        ("headingDegrees", D),
        ("headingErrorDegrees", D),
        ("conservativeHeadingErrorDegrees", D),
    ],
    "GameRotationVector": [
        ("utcTimeMillis", I),
        ("elapsedRealtimeNanos", I),
        ("x", D),
        ("y", D),
        ("z", D),
        ("w", D),
    ],
    "Agc": [
        ("utcTimeMillis", I),
        ("TimeNanos", I),
        ("LeapSecond", I),
        ("TimeUncertaintyNanos", D),
        ("FullBiasNanos", I),
        ("BiasNanos", D),
        ("BiasUncertaintyNanos", D),
        ("DriftNanosPerSecond", D),
        ("DriftUncertaintyNanosPerSecond", D),
        ("HardwareClockDiscontinuityCount", I),
        ("AgcDb", D),
        ("CarrierFrequencyHz", I),
        ("ConstellationType", I),
    ],
}

con = duckdb.connect(database=":memory:")
con.execute(
    "SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false;"
)
con.execute(f"SET temp_directory='{OUT}/scratch';")

t0 = time.time()
for label, cols in MAPS.items():
    dest = os.path.join(TYPED, label + ".parquet")
    if os.path.exists(dest):
        continue
    sel = []
    for k, (cname, ctype) in enumerate(cols, start=2):  # position 1 is the label
        sel.append(
            f"try_cast(nullif(split_part(line, ',', {k}), '') AS {ctype}) AS {cname}"
        )
    nfields = len(cols) + 1
    q = f"""
        COPY (
            SELECT session,
                   {", ".join(sel)},
                   split_part(line, ',', {nfields + 1}) AS _extra_field_1,
                   line
            FROM read_parquet('{GLOB}', union_by_name=false)
            WHERE label = '{label}'
        ) TO '{dest}' (COMPRESSION zstd, ROW_GROUP_SIZE 100000)
    """
    rss0 = psutil.Process().memory_info().rss / 1e6
    con.execute(q)
    rss1 = psutil.Process().memory_info().rss / 1e6
    print(f"{label}: rss {rss0:.0f}->{rss1:.0f} MB", flush=True)

# Nav: variable arity. Store the 5 declared fields + count + remainder-as-text.
dest = os.path.join(TYPED, "Nav.parquet")
if not os.path.exists(dest):
    q = f"""
        COPY (
            SELECT session,
                   try_cast(nullif(split_part(line, ',', 2), '') AS BIGINT) AS Svid,
                   try_cast(nullif(split_part(line, ',', 3), '') AS BIGINT) AS Type,
                   try_cast(nullif(split_part(line, ',', 4), '') AS BIGINT) AS Status,
                   try_cast(nullif(split_part(line, ',', 5), '') AS BIGINT) AS MessageId,
                   try_cast(nullif(split_part(line, ',', 6), '') AS BIGINT) AS SubMessageId,
                   array_length(str_split(line, ','), 1) - 6 AS nDataBytes,
                   substr(line, length('Nav,' || split_part(line,',',2) || ',' ||
                          split_part(line,',',3) || ',' || split_part(line,',',4) || ',' ||
                          split_part(line,',',5) || ',' || split_part(line,',',6) || ',') + 1)
                     AS DataBytes,
                   line
            FROM read_parquet('{GLOB}')
            WHERE label = 'Nav'
        ) TO '{dest}' (COMPRESSION zstd, ROW_GROUP_SIZE 100000)
    """
    con.execute(q)
    print("Nav done", flush=True)
print("stage2 finished in", round(time.time() - t0, 1), "s")
