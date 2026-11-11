
-- Compute gravity-derived pitch and roll vs logged pitchDeg and rollDeg across sessions
WITH accel_unified AS (
    -- Samsung calibrated Accel
    SELECT 
        session, utcTimeMillis,
        AccelXMps2 as ax, AccelYMps2 as ay, AccelZMps2 as az
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Accel.parquet')
    UNION ALL
    -- Other devices UncalAccel (subtracting bias)
    SELECT 
        session, utcTimeMillis,
        (UncalAccelXMps2 - BiasXMps2) as ax,
        (UncalAccelYMps2 - BiasYMps2) as ay,
        (UncalAccelZMps2 - BiasZMps2) as az
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/UncalAccel.parquet')
),
sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
),
merged AS (
    SELECT 
        o.session,
        s.device_name,
        o.pitchDeg as logged_pitch,
        o.rollDeg as logged_roll,
        -- Gravity-derived pitch: tilt around X-axis
        DEGREES(ATAN2(-a.ay, SQRT(a.ax * a.ax + a.az * a.az))) as grav_pitch,
        -- Gravity-derived roll: bank around Y-axis
        DEGREES(ATAN2(a.ax, a.az)) as grav_roll
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/OrientationDeg.parquet') o
    JOIN sessions s ON o.session = s.session
    ASOF JOIN accel_unified a
      ON o.session = a.session
     AND o.utcTimeMillis >= a.utcTimeMillis
    WHERE (o.utcTimeMillis - a.utcTimeMillis) <= 200
)
SELECT 
    device_name,
    FLOOR(grav_pitch / 5.0) * 5.0 as grav_pitch_bin,
    FLOOR(logged_pitch / 5.0) * 5.0 as logged_pitch_bin,
    COUNT(*) as count
FROM merged
GROUP BY device_name, grav_pitch_bin, logged_pitch_bin
HAVING count >= 10
ORDER BY device_name, grav_pitch_bin, logged_pitch_bin;
