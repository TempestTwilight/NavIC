-- Compute marginal quantiles of features and targets across devices
WITH target_sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    WHERE session IN (
        'gnss_log_2026_10_22_08_30_05', -- Samsung
        'gnss_log_2026_10_19_08_51_20', -- OnePlus CPH2401
        'gnss_log_2026_10_22_08_30_07', -- OnePlus CPH2487
        'gnss_log_2026_10_21_09_04_50'  -- OPPO CPH2325
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
        WHERE session IN ('gnss_log_2026_10_22_08_30_05', 'gnss_log_2026_10_19_08_51_20', 'gnss_log_2026_10_22_08_30_07', 'gnss_log_2026_10_21_09_04_50')
    ) st ON r.session = st.session
       AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    WHERE r.session IN ('gnss_log_2026_10_22_08_30_05', 'gnss_log_2026_10_19_08_51_20', 'gnss_log_2026_10_22_08_30_07', 'gnss_log_2026_10_21_09_04_50')
      AND r.Cn0DbHz > 0
),
lagged_pairs AS (
    SELECT 
        t1.device_name,
        t1.el,
        t1.cn0_t,
        (t2.cn0_t - t1.cn0_t) as delta_15s
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
    COUNT(*) as n_obs,
    -- Cn0 level
    ROUND(AVG(cn0_t), 2) as mean_cn0,
    ROUND(MEDIAN(cn0_t), 2) as median_cn0,
    ROUND(STDDEV(cn0_t), 2) as std_cn0,
    -- Delta 15s
    ROUND(AVG(delta_15s), 3) as mean_delta_15s,
    ROUND(MEDIAN(delta_15s), 3) as median_delta_15s,
    ROUND(STDDEV(delta_15s), 3) as std_delta_15s,
    -- Elevation
    ROUND(AVG(el), 2) as mean_elevation,
    ROUND(MEDIAN(el), 2) as median_elevation
FROM lagged_pairs
GROUP BY device_name;
