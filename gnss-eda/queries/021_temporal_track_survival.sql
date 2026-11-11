-- Compute track survival rates over 1 to 30s horizons by elevation band
WITH session_tracks AS (
    SELECT 
        ConstellationType,
        Svid,
        CarrierFrequencyHz,
        CodeType,
        CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        Cn0DbHz
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_22_08_30_05' AND Cn0DbHz > 0
),
epoch_set AS (
    SELECT DISTINCT sec_epoch, ConstellationType, Svid, CarrierFrequencyHz, CodeType
    FROM session_tracks
),
survival_eval AS (
    SELECT 
        e.sec_epoch,
        e.ConstellationType,
        e.Svid,
        e.CarrierFrequencyHz,
        e.CodeType,
        st.ElevationDegrees as el,
        -- Check presence at t+15 and t+30
        CASE WHEN e15.sec_epoch IS NOT NULL THEN 1 ELSE 0 END as survives_15s,
        CASE WHEN e30.sec_epoch IS NOT NULL THEN 1 ELSE 0 END as survives_30s
    FROM epoch_set e
    JOIN (
        SELECT DISTINCT CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session = 'gnss_log_2026_10_22_08_30_05'
    ) st ON e.sec_epoch = st.sec_epoch
       AND e.ConstellationType = st.ConstellationType
       AND e.CarrierFrequencyHz = st.CarrierFrequencyHz
    LEFT JOIN epoch_set e15 
      ON e.ConstellationType = e15.ConstellationType
     AND e.Svid = e15.Svid
     AND e.CarrierFrequencyHz = e15.CarrierFrequencyHz
     AND e.CodeType = e15.CodeType
     AND e15.sec_epoch = e.sec_epoch + 15
    LEFT JOIN epoch_set e30 
      ON e.ConstellationType = e30.ConstellationType
     AND e.Svid = e30.Svid
     AND e.CarrierFrequencyHz = e30.CarrierFrequencyHz
     AND e.CodeType = e30.CodeType
     AND e30.sec_epoch = e.sec_epoch + 30
    WHERE e.sec_epoch <= (SELECT MAX(sec_epoch) - 30 FROM epoch_set)
)
SELECT 
    CASE 
        WHEN el < 20 THEN 'Low (<20 deg)'
        WHEN el < 45 THEN 'Mid (20-45 deg)'
        ELSE 'High (>=45 deg)'
    END as elevation_band,
    COUNT(*) as total_epochs,
    ROUND(AVG(survives_15s) * 100.0, 2) as survival_rate_15s_pct,
    ROUND(AVG(survives_30s) * 100.0, 2) as survival_rate_30s_pct
FROM survival_eval
GROUP BY elevation_band
ORDER BY elevation_band;
