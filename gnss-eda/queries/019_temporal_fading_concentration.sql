-- Compute 15s delta distribution binned by delta, elevation, and constellation
WITH target_sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    WHERE session IN (
        'gnss_log_2026_10_22_08_30_05',
        'gnss_log_2026_10_19_08_51_20',
        'gnss_log_2026_10_22_08_30_07',
        'gnss_log_2026_10_21_09_04_50'
    )
),
base_signals AS (
    SELECT 
        r.session,
        s.device_name,
        r.ConstellationType,
        r.Svid,
        r.CarrierFrequencyHz,
        r.CodeType,
        CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
        r.Cn0DbHz as cn0_t,
        st.ElevationDegrees as el
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN target_sessions s ON r.session = s.session
    JOIN (
        SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
    ) st ON r.session = st.session
       AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    WHERE r.Cn0DbHz > 0
),
lagged_pairs AS (
    SELECT 
        t1.device_name,
        t1.ConstellationType,
        t1.el,
        t1.cn0_t,
        (t2.cn0_t - t1.cn0_t) as delta_15s,
        CASE WHEN ABS(t2.cn0_t - t1.cn0_t) > 1.5 THEN 1 ELSE 0 END as is_fading
    FROM base_signals t1
    JOIN base_signals t2 
      ON t1.session = t2.session
     AND t1.ConstellationType = t2.ConstellationType
     AND t1.Svid = t2.Svid
     AND t1.CarrierFrequencyHz = t2.CarrierFrequencyHz
     AND t1.CodeType = t2.CodeType
     AND t2.sec_epoch = t1.sec_epoch + 15
)
SELECT 
    device_name,
    CASE 
        WHEN el < 20 THEN 'Low (<20 deg)'
        WHEN el < 45 THEN 'Mid (20-45 deg)'
        ELSE 'High (>=45 deg)'
    END as elevation_band,
    COUNT(*) as total_obs,
    SUM(is_fading) as fading_obs,
    ROUND(AVG(is_fading) * 100.0, 2) as fading_pct,
    ROUND(AVG(delta_15s), 3) as mean_delta,
    ROUND(STDDEV(delta_15s), 3) as std_delta
FROM lagged_pairs
GROUP BY device_name, elevation_band
ORDER BY device_name, elevation_band;
