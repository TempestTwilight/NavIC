
-- Compute body-frame satellite vectors and elevation residual C/N0
WITH base AS (
    SELECT 
        r.session,
        'OnePlus CPH2401' as device_name,
        r.Cn0DbHz,
        st.AzimuthDegrees as az,
        st.ElevationDegrees as el,
        -- Nominal elevation baseline: 20 + 20 * sin(el)
        r.Cn0DbHz - (20.0 + 20.0 * SIN(RADIANS(st.ElevationDegrees))) as cn0_residual,
        o.yawDeg as yaw,
        o.pitchDeg as pitch,
        o.rollDeg as roll
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, AzimuthDegrees, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
    ) st ON r.session = st.session
       AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    ASOF JOIN read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/OrientationDeg.parquet') o
      ON r.session = o.session
     AND r.utcTimeMillis >= o.utcTimeMillis
    WHERE r.session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
      AND r.Cn0DbHz > 0 AND st.ElevationDegrees >= 0
      AND (r.utcTimeMillis - o.utcTimeMillis) <= 200
)
SELECT 
    device_name,
    az, el, yaw, pitch, roll, cn0_residual
FROM base
LIMIT 20000;
