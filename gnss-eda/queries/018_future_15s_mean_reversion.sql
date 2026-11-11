-- Compute paired C/N0(t) and C/N0(t+15s) across sessions
WITH target_sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    WHERE session IN (
        'gnss_log_2026_10_22_08_30_05', -- Samsung Oct 22
        'gnss_log_2026_10_19_08_51_20', -- OnePlus CPH2401 Oct 19
        'gnss_log_2026_10_22_08_30_07', -- OnePlus CPH2487 Oct 22
        'gnss_log_2026_10_21_09_04_50'  -- OPPO CPH2325 Oct 21
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
        r.Cn0DbHz as cn0_t
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN target_sessions s ON r.session = s.session
    WHERE r.Cn0DbHz > 0
),
lagged_pairs AS (
    SELECT 
        t1.device_name,
        t1.cn0_t,
        t2.cn0_t as cn0_t_plus_15,
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
    FLOOR(cn0_t / 2.0) * 2.0 as cn0_t_bin,
    COUNT(*) as n_pairs,
    ROUND(AVG(cn0_t_plus_15), 2) as mean_future_cn0,
    ROUND(MEDIAN(cn0_t_plus_15), 2) as median_future_cn0,
    ROUND(AVG(delta_15s), 2) as mean_drift,
    ROUND(MEDIAN(delta_15s), 2) as median_drift,
    ROUND(QUANTILE_CONT(cn0_t_plus_15, 0.25), 2) as p25_future,
    ROUND(QUANTILE_CONT(cn0_t_plus_15, 0.75), 2) as p75_future
FROM lagged_pairs
GROUP BY device_name, cn0_t_bin
HAVING COUNT(*) >= 20
ORDER BY device_name, cn0_t_bin;
